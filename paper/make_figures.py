"""Build the paper's figures in paper/figures from the results in results/.

  python paper/make_figures.py          (run from the repo root)

Static figures are copied from figures/. The language model figures are rebuilt here so that they show
pick-one-of-two with the fixed chooser (G-5) and use the paper's names for each assistant.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

COPY = {"fig2_help_now_after": "p1_during_vs_after", "fig3_mechanism": "p1b_mechanism",
        "fig4_learned": "p2_frontier", "fig5_sequence": "p3_oracle_continual",
        "figA1_replication": "p0_replication", "figA2_robustness": "robustness",
        "figA3_policy_actions": "p2_actions"}
for new, old in COPY.items():
    for ext in ("pdf", "png"):
        shutil.copy(f"figures/{old}.{ext}", f"{OUT}/{new}.{ext}")

from src.llm import analyze_eval, analyze_p3  # noqa: E402

NAMES = {"none": "No AI", "sbp": "SBP (oracle nudge)", "pcc": "PCC (oracle full answer)",
         "gated": "Gated oracle (Simulation 2 rule)", "base": "LLM, untrained", "now": "LLM, scalar gain",
         "thumbs": "LLM, thumbs up/down", "thumbskl": "LLM, thumbs + KL",
         "pairwisefix": "LLM, pick one of two", "pairwisekl": "LLM, pick one of two + KL"}

# Figure 6: frozen checkpoints after the first problem, with context
analyze_eval.ORDER = ["sbp", "pcc", "gated", "base", "now", "thumbs", "thumbskl", "pairwisefix", "pairwisekl"]
analyze_eval.NAMES = dict(analyze_eval.NAMES, **NAMES)
analyze_eval.COLORS = dict(analyze_eval.COLORS, pairwisefix=analyze_p3.ARM_COLORS["pairwisefix"])
with tempfile.TemporaryDirectory() as tmp:
    for f in glob.glob("results/p3eval/b/*_p0_adapter_p0.json"):
        shutil.copy(f, tmp)
    tag = os.path.basename(tmp)
    analyze_eval.main(tmp)
    for ext in ("pdf", "png"):
        shutil.move(f"figures/p3eval_{tag}.{ext}", f"{OUT}/fig6_llm_frozen.{ext}")
    os.remove(f"results/summary/p3eval_{tag}.csv")

# Figure 7: gap to SBP over training checkpoints
subprocess.run([sys.executable, "scripts/analysis/p3b_gap_figure.py"], check=True, capture_output=True)
for ext in ("pdf", "png"):
    shutil.copy(f"figures/p3b_gap_over_training.{ext}", f"{OUT}/fig7_gap_over_training.{ext}")

# Figure A4: training dynamics with context, rho = 1 (old pick-one-of-two chooser left out)
analyze_p3.ARM_NAMES = dict(analyze_p3.ARM_NAMES, **NAMES)
with tempfile.TemporaryDirectory() as tmp:
    for d in glob.glob("results/p3b/*_s*"):
        if os.path.basename(d).rsplit("_s", 1)[0] in ("now", "thumbs", "thumbskl", "pairwisefix", "pairwisekl"):
            os.symlink(os.path.abspath(d), os.path.join(tmp, os.path.basename(d)))
    analyze_p3.main(tmp, "results/p3_oracle", "paper_figA4_tmp")
    for ext in ("pdf", "png"):
        shutil.move(f"figures/paper_figA4_tmp.{ext}", f"{OUT}/figA4_llm_training.{ext}")
print("figures written to", OUT)
