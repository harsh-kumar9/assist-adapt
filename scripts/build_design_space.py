"""Build docs/design-space.html from the template, inlining figures as data URIs.

  python scripts/build_design_space.py
"""
import base64
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "docs", "design-space.template.html")).read()


def inline(m):
    data = open(os.path.join(ROOT, m.group(1)), "rb").read()
    return "data:image/png;base64," + base64.b64encode(data).decode()


out = re.sub(r"\{\{IMG:([^}]+)\}\}", inline, src)
left = re.findall(r"\{\{[A-Z_]+\}\}", out)
assert not left, f"unfilled placeholders: {left}"
open(os.path.join(ROOT, "docs", "design-space.html"), "w").write(out)
print(f"wrote docs/design-space.html ({len(out) / 1e6:.2f} MB)")
