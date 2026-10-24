"""What did each P4b reward teach the model? Ask every trained model (epoch-20 adapter) and the untrained
one the same questions and compare their answers.

The questions: for each seed, every team in the never-updated model's culture at the end of epoch 20, with
the same prompt (top 3 approaches as context) on that epoch's problem; 8 answers each at the deployment
temperature. Per answer: copies the top approach, copies any approach in the context, helps, distance from
the asker's own approach, new to the culture, distance to the nearest approach in the culture.

  python -m src.llm.policy_probe --root results/p4b --out results/p4b_probe/probe.jsonl
"""
import argparse
import json
import os
import types

import numpy as np

from src.llm.online import LLM, parse, popcount, render, top_context
from src.nk import Landscape

N, K, M = 20, 9, 10
MC = (1 << M) - 1

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results/p4b")
ap.add_argument("--arms", default="thumbs,thumbskl,pcreative,hcreative,thumbsT2,pcreativeT2,hcreativeT2")
ap.add_argument("--seeds", default="0,1,2,3,4,5")
ap.add_argument("--adapter", default="adapter_e20")
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--chunk", type=int, default=50)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

llm = LLM(types.SimpleNamespace(model=args.model, device="cuda", lora_r=16, lr=5e-5, temperature=1.0,
                                max_new_tokens=16, micro_batch=48, kl=0.0))
init = {k: v.clone() for k, v in get_peft_model_state_dict(llm.model).items()}  # untrained LoRA (B = 0)
done = set()
if os.path.exists(args.out):
    done = {(r["arm"], r["seed"]) for r in map(json.loads, open(args.out))}


def state(seed):
    """The never-updated model's culture at the end of the run, and the last epoch's problem."""
    cfg = json.load(open(os.path.join(args.root, f"frozen_s{seed}", "config.json")))
    st = json.load(open(os.path.join(args.root, f"frozen_s{seed}", "state.json")))
    rho = cfg["rho"]
    land = Landscape(N, K, M, rho, np.random.default_rng([seed, int(round(rho * 1000)), 0]))
    srng = np.random.default_rng([seed, int(round(rho * 1000)), 3])
    for _ in range(st["next_epoch"] - 1):
        land = land.shifted(cfg["shift"], srng)
    return np.array(st["codes"], dtype=np.int64), land.F


for seed in [int(s) for s in args.seeds.split(",")]:
    codes, F = state(seed)
    fit = F[codes]
    ctx = top_context(codes, fit, 3)
    ctx_mods = {c & MC for c, _ in ctx}
    culture = np.unique(codes & MC)
    users = [render(c, context=ctx) for c in codes]
    for arm in ["frozen"] + args.arms.split(","):
        if (arm, seed) in done:
            continue
        if arm == "frozen":
            set_peft_model_state_dict(llm.model, init)
        else:
            path = os.path.join(args.root, f"{arm}_s{seed}", args.adapter, "adapter_model.safetensors")
            set_peft_model_state_dict(llm.model, load_file(path))
        texts = []
        for c0 in range(0, len(users), args.chunk):
            texts += llm.sample(users[c0:c0 + args.chunk], args.G, seed=seed * 7919 + c0)[2]
        rec = dict(copy_top=[], in_context=[], helpful=[], d_asker=[], new=[], d_culture=[], distinct=[],
                   valid=[], help_new=[], help_d_culture=[], help_d_asker=[])
        for i in range(len(codes)):
            ans = [parse(texts[i * args.G + j]) for j in range(args.G)]
            ok = [s for s in ans if s is not None]
            rec["valid"].append(len(ok) / args.G)
            rec["distinct"].append(len(set(ok)))
            for s in ok:
                h = bool(F[(codes[i] & ~MC) | s] > fit[i])
                dc = int(min(popcount(s ^ int(c)) for c in culture))
                da = popcount(s ^ (int(codes[i]) & MC))
                rec["copy_top"].append(s == (ctx[0][0] & MC))
                rec["in_context"].append(s in ctx_mods)
                rec["helpful"].append(h)
                rec["d_asker"].append(da)
                rec["new"].append(dc > 0)
                rec["d_culture"].append(dc)
                if h:
                    rec["help_new"].append(dc > 0)
                    rec["help_d_culture"].append(dc)
                    rec["help_d_asker"].append(da)
        row = dict(arm=arm, seed=seed, **{k: float(np.mean(v)) if v else float("nan") for k, v in rec.items()})
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
