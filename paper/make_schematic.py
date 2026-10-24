"""Figure 1: the probe. Paper style: no headline text, labels only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7, 1.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=6.6, linespacing=1.3)


box(1, 9, 27, 12, "Community searches problem $P$\nwith an assistant\n(rounds 1 to $T$)", "#e9f1fb", BLUE)
ax.annotate("", xy=(33, 15), xytext=(29, 15), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
box(34, 9, 15, 12, "Copy the\ncommunity\nat round $T$", "#f4f4f1", LINE)
cells = [("same problem $P$\nassistant on", 24), ("same problem $P$\nassistant removed", 17.5),
         ("related problem $P'$\nassistant on", 11), ("related problem $P'$\nassistant removed", 4.5)]
for i, (t, y) in enumerate(cells):
    ax.annotate("", xy=(56, y + 2.2), xytext=(50, 15), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6))
    last = i == 3
    box(57, y, 22, 4.4, t, "#fdeee6" if last else "#f4f4f1", ORANGE if last else LINE)
    ax.annotate("", xy=(84, y + 2.2), xytext=(79.6, y + 2.2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6))
    ax.text(85, y + 2.2, "30 more rounds:\nmean fitness", va="center", fontsize=6, color=INK2)
ax.text(14.5, 5.5, "measured: fitness while assisted", ha="center", fontsize=6, color=INK2)
ax.text(68, 0.6, "$P'$: 20% of payoff tables redrawn", ha="center", fontsize=6, color=MUTED)
for ext in ("pdf", "png"):
    fig.savefig(f"paper/figures/fig1_probe.{ext}", dpi=300, bbox_inches="tight")
print("ok")
