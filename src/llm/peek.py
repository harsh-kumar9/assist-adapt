"""Quick per-problem summary of P3 runs: python -m src.llm.peek results/p3 now_s0 kl_s0 ..."""
import json
import statistics as st
import sys


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(st.mean(v), 2) if v else None


def summ(root, run):
    rows = [json.loads(l) for l in open(f"{root}/{run}/rounds.jsonl")]
    for k in range(4):
        rk = [r for r in rows if r["problem"] == k]
        if not rk:
            continue
        early = [r for r in rk if r["round"] < 10]
        late = [r for r in rk if r["round"] >= 30]
        parts = [f"{run:14s} p{k} n{len(rk):2d} fit {mean(rk, 'fitness'):.3f}"]
        for name, rs in (("early", early), ("late", late)):
            parts.append(f"{name}: improve {mean(rs, 'improve_shown')} top {mean(rs, 'top_share')} "
                         f"grp {mean(rs, 'group_distinct')} back {mean(rs, 'step_back')} edit {mean(rs, 'shown_edit')}")
        print(" | ".join(parts))


if __name__ == "__main__":
    for run in sys.argv[2:]:
        try:
            summ(sys.argv[1], run)
        except FileNotFoundError:
            print(run, "missing")
