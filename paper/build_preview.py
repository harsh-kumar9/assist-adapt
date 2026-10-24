"""Readable preview of paper/main.tex without a TeX install: LaTeX subset -> HTML -> PDF (headless Chrome).

  python paper/build_preview.py            (writes paper/preview.html and paper/preview.pdf)

Handles only the constructs main.tex uses. Overleaf (or any TeX install) is the real build.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# ---------------------------------------------------------------- bibliography

def parse_bib(path):
    text = open(path).read()
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{(.*?)\},?\s*$", m.group(3), re.M):
            fields[f.group(1).lower()] = f.group(2)
        out[m.group(2).strip()] = fields
    return out


def clean_tex(s):
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\c c", "ç").replace("\\i", "ı").replace('\\"O', "Ö").replace("\\H o", "ő")
    s = re.sub(r"\\c\s*c", "ç", s)
    s = s.replace("--", "–").replace("\\&", "&")
    return s


def authors(fields):
    names = [a.strip() for a in fields.get("author", "").split(" and ")]
    last = []
    for n in names:
        if n == "others":
            last.append("others")
        elif n.startswith("{"):
            last.append(clean_tex(n))
        else:
            last.append(clean_tex(n.split(",")[0]))
    return last


def short_cite(fields):
    a = authors(fields)
    a = [x for x in a if x != "others"] if len(a) <= 2 else a
    if len(a) == 1:
        name = a[0]
    elif len(a) == 2 and "others" not in a:
        name = f"{a[0]} and {a[1]}"
    else:
        name = f"{a[0]} et al."
    return name, fields.get("year", "")


def full_ref(fields):
    names = []
    for n in fields.get("author", "").split(" and "):
        n = n.strip()
        if n == "others":
            names.append("et al.")
        elif n.startswith("{"):
            names.append(clean_tex(n))
        elif "," in n:
            l, f = n.split(",", 1)
            names.append(f"{clean_tex(l)}, {clean_tex(f.strip())}")
        else:
            names.append(clean_tex(n))
    venue = fields.get("journal") or fields.get("booktitle") or ""
    bits = [", ".join(names).replace(", et al.", " et al."), f"({fields.get('year', '')}).",
            clean_tex(fields.get("title", "")) + ".", f"<em>{clean_tex(venue)}</em>"]
    if fields.get("volume"):
        bits[-1] += f", {fields['volume']}"
    if fields.get("pages"):
        bits[-1] += f", {clean_tex(fields['pages'])}"
    bits[-1] += "."
    if fields.get("note"):
        bits.append(clean_tex(fields["note"]) + ".")
    return " ".join(bits)


# ---------------------------------------------------------------- brace helpers

def take_group(s, i):
    """s[i] == '{'; return (content, index after the closing brace)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces near: " + s[i:i + 80])


def take_opt(s, i):
    """Optional [..] argument at s[i]; returns (content or None, new index)."""
    if i < len(s) and s[i] == "[":
        j = s.index("]", i)
        return s[i + 1:j], j + 1
    return None, i


# ---------------------------------------------------------------- math

GREEK = {"rho": "ρ", "phi": "φ", "beta": "β", "lambda": "λ", "times": "×", "approx": "≈", "geq": "≥",
         "leq": "≤", "mapsto": "↦", "in": "∈", "pm": "±", "ldots": "…", "cdot": "·", "sum": "Σ"}


def math_html(m):
    m = m.strip()
    out, i = "", 0
    while i < len(m):
        c = m[i]
        if c == "\\":
            name = re.match(r"\\([A-Za-z]+)", m[i:])
            if name:
                n = name.group(1)
                i += len(n) + 1
                if n == "text":
                    arg, i = take_group(m, i)
                    out += f'<span class="mt">{arg}</span>'
                elif n == "frac":
                    a, i = take_group(m, i)
                    b, i = take_group(m, i)
                    out += f"{math_html(a)}/{math_html(b)}"
                elif n in GREEK:
                    out += GREEK[n]
                else:
                    out += n
                continue
            out += m[i + 1]
            i += 2
            continue
        if c in "_^":
            tag = "sub" if c == "_" else "sup"
            if m[i + 1] == "{":
                arg, i = take_group(m, i + 1)
            else:
                arg, i = m[i + 1], i + 2
            out += f"<{tag}>{math_html(arg)}</{tag}>"
            continue
        if c in "{}":
            i += 1
            continue
        if c.isalpha():
            out += f"<i>{c}</i>"
        else:
            out += {"-": "−", "<": "&lt;", ">": "&gt;"}.get(c, c)
        i += 1
    return out


# ---------------------------------------------------------------- inline text

class Doc:
    def __init__(self, bib):
        self.bib, self.labels, self.footnotes = bib, {}, []

    def cite(self, keys, mode, pre=None, post=None):
        parts = []
        for k in [k.strip() for k in keys.split(",")]:
            name, year = short_cite(self.bib[k])
            parts.append(f"{name} ({year})" if mode == "t" else f"{name}, {year}")
        if mode == "t":
            return "; ".join(parts)
        inner = "; ".join(parts)
        if pre:
            inner = f"{pre} {inner}"
        if post:
            inner = f"{inner}, {post}"
        return f"({inner})"

    def inline(self, s):
        out, i = "", 0
        while i < len(s):
            c = s[i]
            if c == "$":
                j = s.index("$", i + 1)
                out += f'<span class="m">{math_html(s[i + 1:j])}</span>'
                i = j + 1
                continue
            if c == "\\":
                m = re.match(r"\\([A-Za-z]+)", s[i:])
                if not m:
                    nxt = s[i + 1]
                    out += {"%": "%", "&": "&amp;", "_": "_", ",": "&thinsp;", "\\": "<br>"}.get(nxt, nxt)
                    i += 2
                    continue
                n = m.group(1)
                i += len(n) + 1
                if n in ("emph", "textit"):
                    a, i = take_group(s, i)
                    out += f"<em>{self.inline(a)}</em>"
                elif n == "textbf":
                    a, i = take_group(s, i)
                    out += f"<strong>{self.inline(a)}</strong>"
                elif n == "pending":
                    a, i = take_group(s, i)
                    out += f'<span class="pending">[{self.inline(a)}]</span>'
                elif n in ("citet", "citep", "citealp"):
                    pre, i = take_opt(s, i)
                    post, i = take_opt(s, i)
                    if pre is not None and post is None:
                        post, pre = pre, None
                    keys, i = take_group(s, i)
                    if n == "citep" and post is not None and pre is not None:
                        out += self.cite(keys, "p", pre=pre or None, post=post or None)
                    else:
                        out += self.cite(keys, "t" if n == "citet" else "p", post=post or None)
                elif n == "ref":
                    a, i = take_group(s, i)
                    out += self.labels.get(a, "??")
                elif n == "footnote":
                    a, i = take_group(s, i)
                    self.footnotes.append(self.inline(a))
                    k = len(self.footnotes)
                    out += f'<sup class="fn"><a href="#fn{k}" id="fr{k}">{k}</a></sup>'
                elif n == "H":
                    a, i = take_group(s, i)
                    out += {"o": "ő"}.get(a, a)
                elif n in ("small", "centering", "label"):
                    if n == "label":
                        _, i = take_group(s, i)
                elif n in ("toprule", "midrule", "bottomrule"):
                    pass
                else:
                    out += ""
                continue
            if s.startswith("---", i):
                out += "—"
                i += 3
                continue
            if s.startswith("--", i):
                out += "–"
                i += 2
                continue
            if s.startswith("``", i):
                out += "“"
                i += 2
                continue
            if s.startswith("''", i):
                out += "”"
                i += 2
                continue
            if c == "~":
                out += "&nbsp;"
            elif c == "<":
                out += "&lt;"
            elif c == ">":
                out += "&gt;"
            elif c in "{}":
                pass
            elif c == "'" :
                out += "’"
            else:
                out += c
            i += 1
        return out


# ---------------------------------------------------------------- blocks

def number_labels(body):
    """First pass: sections, figures, tables and appendix sections get numbers for \\ref."""
    labels, sec, sub, fig, tab, appendix = {}, 0, 0, 0, 0, False
    for m in re.finditer(r"\\(section|subsection|appendix|begin\{figure\}|begin\{table\}|label)\b(\{([^}]*)\})?", body):
        kind = m.group(1)
        if kind == "appendix":
            appendix, sec = True, 0
        elif kind == "section":
            sec, sub, last = sec + 1, 0, ("sec", sec + 1)
        elif kind == "subsection":
            sub += 1
            last = ("sub", sub)
        elif kind == "begin{figure}":
            fig += 1
            last = ("fig", fig)
        elif kind == "begin{table}":
            tab += 1
            last = ("tab", tab)
        elif kind == "label":
            name = m.group(3)
            if last[0] == "sec":
                labels[name] = chr(64 + last[1]) if appendix else str(last[1])
            elif last[0] == "sub":
                labels[name] = f"{sec}.{last[1]}"
            elif last[0] == "fig":
                labels[name] = (f"A{last[1]}" if appendix else str(last[1]))
            else:
                labels[name] = str(last[1])
    # appendix figures are numbered A1.. in order of appearance after \appendix
    return labels


def convert(tex, bib):
    doc = Doc(bib)
    body = tex.split("\\begin{document}", 1)[1].split("\\end{document}")[0]
    # appendix figure numbering: count figures after \appendix separately
    main, app = body.split("\\appendix", 1)
    labels = {}
    n_fig = 0
    for m in re.finditer(r"\\begin\{figure\}.*?\\label\{([^}]*)\}", main, re.S):
        n_fig += 1
        labels[m.group(1)] = str(n_fig)
    a_fig = 0
    for m in re.finditer(r"\\begin\{figure\}.*?\\label\{([^}]*)\}", app, re.S):
        a_fig += 1
        labels[m.group(1)] = f"A{a_fig}"
    for m in re.finditer(r"\\begin\{table\}.*?\\label\{([^}]*)\}", body, re.S):
        labels.setdefault(m.group(1), "1")
    sec_labels = number_labels(body)
    for k, v in sec_labels.items():
        if k.startswith(("sec:", "app:")):
            labels[k] = v
    doc.labels = labels

    title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
    author = re.search(r"\\author\{(.*?)\}\n", tex).group(1).replace("\\\\", "<br>")
    date = re.search(r"\\date\{(.*?)\}\n", tex).group(1)
    html = [f'<header><h1>{doc.inline(title)}</h1><p class="auth">{author}</p><p class="date">{date}</p></header>']

    sec, sub, appendix = 0, 0, False
    blocks = re.split(r"\n\s*\n", body)
    in_env = None
    buf = []
    for blk in blocks:
        blk = blk.strip()
        if not blk or blk.startswith("%"):
            continue
        if in_env:
            buf.append(blk)
            if f"\\end{{{in_env}}}" in blk:
                html.append(render_env(in_env, "\n\n".join(buf), doc))
                in_env, buf = None, []
            continue
        env = re.match(r"\\begin\{(figure|table|abstract)\}", blk)
        if env and f"\\end{{{env.group(1)}}}" not in blk:
            in_env, buf = env.group(1), [blk]
            continue
        if env:
            html.append(render_env(env.group(1), blk, doc))
            continue
        for line_blk in split_heads(blk):
            html.append(render_block(line_blk, doc, state := None) if False else render_block(line_blk, doc))
    # numbering of section heads
    out = "\n".join(h for h in html if h)
    out = number_heads(out)
    notes = "".join(f'<li id="fn{i + 1}">{t} <a href="#fr{i + 1}">↩</a></li>' for i, t in enumerate(doc.footnotes))
    refs = sorted(bib.items(), key=lambda kv: (authors(kv[1])[0].lower(), kv[1].get("year", "")))
    cited = set(re.findall(r"\\cite\w*(?:\[[^\]]*\])*\{([^}]*)\}", tex))
    cited = {k.strip() for group in cited for k in group.split(",")}
    reflist = "".join(f"<li>{full_ref(f)}</li>" for k, f in refs if k in cited)
    out = out.replace("<!--REFS-->", f'<section class="refs"><h2>References</h2><ul>{reflist}</ul></section>')
    out += f'<section class="notes"><h2>Notes</h2><ol>{notes}</ol></section>'
    return out


def split_heads(blk):
    """Split a block so that each \\section / \\subsection / \\paragraph starts its own piece."""
    parts = re.split(r"(?=\\(?:section|subsection|paragraph|appendix|bibliography|maketitle)\b)", blk)
    return [p.strip() for p in parts if p.strip()]


def render_block(blk, doc):
    if blk.startswith("\\maketitle") or blk.startswith("\\bibliographystyle"):
        return ""
    if blk.startswith("\\bibliography"):
        return "<!--REFS-->"
    if blk.startswith("\\appendix"):
        return '<div class="appendix-start"></div>'
    m = re.match(r"\\(section|subsection)\{(.*?)\}\s*(\\label\{[^}]*\})?\s*(.*)", blk, re.S)
    if m:
        tag = "h2" if m.group(1) == "section" else "h3"
        rest = render_block(m.group(4).strip(), doc) if m.group(4).strip() else ""
        return f'<{tag} class="{m.group(1)}">{doc.inline(m.group(2))}</{tag}>' + rest
    m = re.match(r"\\paragraph\{(.*?)\}\s*(.*)", blk, re.S)
    if m:
        return f'<p><strong>{doc.inline(m.group(1))}</strong> {doc.inline(m.group(2))}</p>'
    if blk.startswith("\\begin{itemize}"):
        items = re.split(r"\\item\s*", blk.replace("\\begin{itemize}", "").replace("\\end{itemize}", ""))
        return "<ul>" + "".join(f"<li>{doc.inline(x.strip())}</li>" for x in items if x.strip()) + "</ul>"
    if "\\begin{itemize}" in blk:
        pre, rest = blk.split("\\begin{itemize}", 1)
        return f"<p>{doc.inline(pre.strip())}</p>" + render_block("\\begin{itemize}" + rest, doc)
    return f"<p>{doc.inline(blk)}</p>"


def render_env(env, blk, doc):
    if env == "abstract":
        inner = blk.replace("\\begin{abstract}", "").replace("\\end{abstract}", "").strip()
        return f'<section class="abstract"><h2>Abstract</h2><p>{doc.inline(inner)}</p></section>'
    cap = ""
    m = re.search(r"\\caption\{", blk)
    if m:
        cap, _ = take_group(blk, m.end() - 1)
    lab = re.search(r"\\label\{([^}]*)\}", blk)
    num = doc.labels.get(lab.group(1), "") if lab else ""
    if env == "figure":
        img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", blk).group(1).replace(".pdf", ".png")
        return (f'<figure><img src="{img}" alt=""><figcaption><strong>Figure {num}.</strong> '
                f"{doc.inline(cap)}</figcaption></figure>")
    tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", blk, re.S).group(1)
    rows = [r.strip() for r in re.split(r"\\\\", tab) if r.strip()]
    trs = []
    for r in rows:
        r = re.sub(r"\\(toprule|midrule|bottomrule)", "", r).strip()
        if not r:
            continue
        cells = [doc.inline(c.strip()) for c in re.split(r"(?<!\\)&", r)]
        tag = "th" if not trs else "td"
        trs.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    return (f'<figure class="tab"><figcaption><strong>Table {num}.</strong> {doc.inline(cap)}</figcaption>'
            f'<table>{"".join(trs)}</table></figure>')


def number_heads(html):
    sec, sub, appendix = 0, 0, False
    out = []
    for piece in re.split(r"(<h2 class=\"section\">|<h3 class=\"subsection\">|<div class=\"appendix-start\"></div>)", html):
        if piece == '<div class="appendix-start"></div>':
            appendix, sec = True, 0
            out.append(piece)
            continue
        if piece == '<h2 class="section">':
            sec, sub = sec + 1, 0
            out.append(piece + (f"{chr(64 + sec)}&ensp;" if appendix else f"{sec}&ensp;"))
            continue
        if piece == '<h3 class="subsection">':
            sub += 1
            out.append(piece + (f"{chr(64 + sec)}.{sub}&ensp;" if appendix else f"{sec}.{sub}&ensp;"))
            continue
        out.append(piece)
    return "".join(out)


CSS = """
@page { size: Letter; margin: 0.9in 0.95in; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.45; color: #111; max-width: 6.6in; margin: 0 auto; }
header { text-align: center; margin-bottom: 1.4em; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 .5em; font-weight: 600; }
.auth { margin: .2em 0; } .date { color: #555; margin: 0; font-size: 9.5pt; }
h2 { font-size: 12.5pt; margin: 1.5em 0 .5em; } h3 { font-size: 11pt; margin: 1.2em 0 .4em; }
.abstract { margin: 0 .35in 1.4em; font-size: 10pt; } .abstract h2 { text-align: center; font-size: 10.5pt; margin: 0 0 .3em; }
p { margin: 0 0 .65em; text-align: justify; hyphens: auto; }
ul { margin: 0 0 .7em 1.2em; padding: 0; } li { margin: .15em 0; }
figure { margin: 1.1em 0 1.2em; page-break-inside: avoid; } figure img { width: 100%; }
figcaption { font-size: 9pt; line-height: 1.35; margin-top: .4em; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin-top: .4em; }
th, td { padding: 3px 5px; text-align: center; border-bottom: 1px solid #ddd; } th { border-top: 1.5px solid #222; border-bottom: 1px solid #222; } tr:last-child td { border-bottom: 1.5px solid #222; }
td:first-child, th:first-child { text-align: left; }
.m i { font-style: italic; } .mt { font-style: normal; }
sup.fn a { text-decoration: none; color: #1f5fbf; font-size: 7.5pt; }
.notes, .refs { font-size: 9pt; } .refs ul { list-style: none; margin-left: 0; } .refs li { margin-bottom: .35em; padding-left: 1.2em; text-indent: -1.2em; }
.pending { color: #b4232a; font-weight: 600; }
a { color: #1f5fbf; }
"""


def main():
    tex = open(os.path.join(HERE, "main.tex")).read()
    bib = parse_bib(os.path.join(HERE, "refs.bib"))
    body = convert(tex, bib)
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>Preview</title><style>{CSS}</style></head><body>{body}</body></html>'
    out = os.path.join(HERE, "preview.html")
    open(out, "w").write(html)
    pdf = os.path.join(HERE, "preview.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file://" + out], check=True, capture_output=True)
    print("wrote", out, "and", pdf)


if __name__ == "__main__":
    main()
