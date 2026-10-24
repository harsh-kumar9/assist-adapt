"""How varied, valid and helpful are the untrained model's answers at different sampling temperatures?
Community states come from the no-AI culture with turnover (P4a setting), after 30 rounds of epochs 1-3.

  python -m src.llm.temp_probe --temps 1.0,1.3,1.6,2.0
"""
import argparse
import types

import numpy as np

from src.culture import round_with
from src.llm.online import LLM, parse, render, top_context
from src.nk import Landscape
from src.sim import Community

N, K, M, H = 20, 9, 10, 10
MC = (1 << M) - 1

ap = argparse.ArgumentParser()
ap.add_argument("--temps", default="1.0,1.3,1.6,2.0")
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--asks", type=int, default=24)
args = ap.parse_args()
llm = LLM(types.SimpleNamespace(model=args.model, device="cuda", lora_r=16, lr=5e-5, temperature=1.0,
                                max_new_tokens=16, micro_batch=48, kl=0.0))
for T in [float(t) for t in args.temps.split(",")]:
    stats = dict(distinct=[], valid=[], helpful=[], copy_top=[], new_to_culture=[])
    for seed in [int(s) for s in args.seeds.split(",")]:
        land = Landscape(N, K, M, 1.0, np.random.default_rng([seed, 1000, 0]))
        comm = Community(100, N, H, 0.05, np.random.default_rng([seed, 2]))
        rng = np.random.default_rng([seed, 1])
        codes = comm.codes0.copy()
        for _ in range(30):
            codes = round_with(codes, land.F, comm, "none", 0.5, 0.1, rng)
        fit = land.F[codes]
        ctx = top_context(codes, fit, 3)
        culture = set((codes & MC).tolist())
        askers = rng.choice(100, args.asks, replace=False)
        _, _, texts = llm.sample([render(codes[i], context=ctx) for i in askers], 8, seed=seed, temperature=T)
        for a_i, i in enumerate(askers):
            ans = [parse(texts[a_i * 8 + j]) for j in range(8)]
            ok = [s for s in ans if s is not None]
            stats["distinct"].append(len(set(ok)))
            stats["valid"].append(len(ok) / 8)
            stats["helpful"].append(np.mean([land.F[(codes[i] & ~MC) | s] > fit[i] for s in ok]) if ok else 0)
            stats["copy_top"].append(np.mean([s == (ctx[0][0] & MC) for s in ok]) if ok else 0)
            stats["new_to_culture"].append(np.mean([s not in culture for s in ok]) if ok else 0)
    print(f"temperature {T}: " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in stats.items()), flush=True)
