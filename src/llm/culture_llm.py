"""P4b: an LLM assistant deployed in a culture with turnover, retrained once per epoch.

The world: the paper's model plus turnover (each round each team leaves with probability mu and is
replaced by a newcomer with a random approach) and epochs (every `rounds` rounds 20% of the payoffs
are redrawn). Within an epoch the LLM is deployed frozen: each asking team gets one answer, with the
community's best current approaches in the prompt, and adopts it if it helps. At the end of the epoch
the LLM is updated once, with GRPO, on a sample of that epoch's requests: 8 answers per request,
scored by one of these rewards (each is zero unless the answer helps the asker, as a thumbs up would be):

  frozen     no update (the deployed model never changes)
  thumbs     1 if the answer improved the asker
  pcreative  thumbs x (1 + lam * distance from the asker's own approach / M): new to the person
  hcreative  thumbs x (1 + lam * distance from the nearest approach anyone holds / M): new to the culture

`--kl` adds the usual penalty toward the untrained model. Between epochs the problem changes and the
updated model is redeployed. This is the loop of a deployed model retrained on its users' feedback.

  python -m src.llm.culture_llm --reward hcreative --seed 0 --out results/p4b/hcreative_s0
"""
import argparse
import json
import os
import time
import types

import numpy as np

from src.llm.online import LLM, P_SOCIAL, parse, popcount, render, top_context
from src.nk import Landscape
from src.sim import Community, mean_pairwise_hamming, step as sim_step

N, K, M, H = 20, 9, 10, 10
MC = np.int64((1 << M) - 1)


def reward(kind, sugg, asker, fit_asker, F, culture_mods, lam):
    """Returns (reward, gain). sugg is a module int or None."""
    if sugg is None:
        return 0.0, 0.0
    new = (asker & ~MC) | sugg
    gain = max(0.0, float(F[new] - fit_asker)) * 10
    helped = 1.0 if gain > 0 else 0.0
    if kind == "thumbs":
        return helped, gain
    if kind == "pcreative":
        return helped * (1 + lam * popcount(sugg ^ (asker & MC)) / M), gain
    if kind == "hcreative":
        d = min(popcount(sugg ^ c) for c in culture_mods)
        return helped * (1 + lam * d / M), gain
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="thumbs", choices=["frozen", "thumbs", "pcreative", "hcreative"])
    ap.add_argument("--lam", type=float, default=1.0, help="novelty bonus weight")
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.01, help="turnover per team per round")
    ap.add_argument("--use", type=float, default=0.5)
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--shift", type=float, default=0.2)
    ap.add_argument("--post", type=int, default=30)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--update-prompts", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=64, help="prompts per GRPO step")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=48)
    ap.add_argument("--save-epochs", default="1,5,10,20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    lands = [Landscape(N, K, M, args.rho, np.random.default_rng([args.seed, int(round(args.rho * 1000)), 0]))]
    srng = np.random.default_rng([args.seed, int(round(args.rho * 1000)), 3])
    for _ in range(args.epochs):
        lands.append(lands[-1].shifted(args.shift, srng))
    comm = Community(100, N, H, 0.05, np.random.default_rng([args.seed, 2]))
    comm.spec = comm.spec.copy()
    rng = np.random.default_rng([args.seed, 1])
    trng = np.random.default_rng([args.seed, 11])
    urng = np.random.default_rng([args.seed, 12])  # which logged requests go into each update
    codes = comm.codes0.copy()
    llm = LLM(types.SimpleNamespace(**{k: getattr(args, k) for k in (
        "model", "device", "lora_r", "lr", "temperature", "max_new_tokens", "micro_batch", "kl")}))

    state_path = os.path.join(args.out, "state.json")
    start = 0
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        start, codes = st["next_epoch"], np.array(st["codes"], dtype=np.int64)
        comm.spec = np.array(st["spec"], dtype=np.int64)
        rng.bit_generator.state, trng.bit_generator.state, urng.bit_generator.state = st["rng"], st["trng"], st["urng"]
        llm.load(os.path.join(args.out, "adapter"))
        print(f"resumed at epoch {start}", flush=True)

    save_epochs = {int(x) for x in args.save_epochs.split(",") if x}
    t0 = time.time()
    for e in range(start, args.epochs):
        F = lands[e].F
        log = []  # requests served this epoch: (asker code, asker fitness, context, culture modules)
        recs = []
        for t in range(args.rounds):
            n = len(codes)
            u_social, u_query, _ = rng.random(n), rng.random(n), rng.random(n)
            col = rng.integers(0, comm.H, n)
            fit = F[codes]
            social = u_social < P_SOCIAL
            query = ~social & (u_query < args.use)
            prop = codes ^ (np.int64(1) << comm.spec[np.arange(n), col])
            fit_pad = np.append(fit, -np.inf)
            best_nb = comm.nbr[np.arange(n), np.argmax(fit_pad[comm.nbr], 1)]
            prop = np.where(social, codes[np.minimum(best_nb, n - 1)], prop)
            idx = np.flatnonzero(query)
            ctx = top_context(codes, fit, args.context) if args.context else None
            culture = np.unique(codes & MC)
            shown, helped = [], []
            if len(idx):
                _, _, texts = llm.sample([render(codes[i], context=ctx) for i in idx], 1,
                                         seed=args.seed * 100003 + e * 1000 + t)
                for j, i in enumerate(idx):
                    s = parse(texts[j])
                    log.append((int(codes[i]), float(fit[i]), ctx, culture))
                    if s is None:
                        continue
                    shown.append(s)
                    cand = (codes[i] & ~MC) | s
                    helped.append(F[cand] > fit[i])
                    prop[i] = cand
            codes = np.where(F[prop] > fit, prop, codes)
            leave = np.flatnonzero(trng.random(n) < args.mu)
            if len(leave):
                codes[leave] = trng.integers(0, 2 ** N, len(leave), dtype=np.int64)
                for i in leave:
                    comm.spec[i] = trng.choice(N, H, replace=False)
            rec = dict(epoch=e, round=t, n_query=int(len(idx)), fitness=float(F[codes].mean()),
                       diversity=mean_pairwise_hamming(codes, N), modules=int(len(np.unique(codes & MC))))
            if shown:
                vals, cnt = np.unique(np.array(shown), return_counts=True)
                rec.update(valid=len(shown) / len(idx), improve=float(np.mean(helped)),
                           top_share=float(cnt.max() / cnt.sum()), shown_unique=float(len(vals) / len(shown)))
            recs.append(rec)
        # next problem with the assistant removed (cheap CPU probe, as in P1)
        prng = np.random.default_rng([args.seed, 7, e, 1, 0])
        pc, vals = codes.copy(), []
        for _ in range(args.post):
            pc = sim_step(pc, lands[e + 1].F, comm, "none", 0.0, P_SOCIAL, M, prng)
            vals.append(float(lands[e + 1].F[pc].mean()))
        # update once per epoch
        upd = dict(epoch=e, next_off=float(np.mean(vals)), served=len(log))
        if args.reward != "frozen" and log:
            pick = urng.choice(len(log), min(args.update_prompts, len(log)), replace=False)
            R_all, G_all, D_all = [], [], []
            for c0 in range(0, len(pick), args.chunk):
                batch = [log[k] for k in pick[c0:c0 + args.chunk]]
                out, P, texts = llm.sample([render(a, context=c) for a, _, c, _ in batch], args.G,
                                           seed=args.seed * 100003 + e * 1000 + 999 + c0)
                R = np.zeros((len(batch), args.G))
                for bi, (a, fa, _, cult) in enumerate(batch):
                    for j in range(args.G):
                        R[bi, j], g = reward(args.reward, parse(texts[bi * args.G + j]), a, fa, F, cult, args.lam)
                        G_all.append(g)
                    D_all.append(len({texts[bi * args.G + j] for j in range(args.G)}))
                adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)
                stats = llm.update(out, P, adv.reshape(-1))
                R_all.append(R.mean())
            upd.update(reward=float(np.mean(R_all)), gain=float(np.mean(G_all)), group_distinct=float(np.mean(D_all)),
                       kl=stats["kl"])
        with open(os.path.join(args.out, "rounds.jsonl"), "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out, "epochs.jsonl"), "a") as f:
            f.write(json.dumps(upd) + "\n")
        llm.save(os.path.join(args.out, "adapter"))
        if e + 1 in save_epochs:
            llm.model.save_pretrained(os.path.join(args.out, f"adapter_e{e + 1}"))
        json.dump(dict(next_epoch=e + 1, codes=codes.tolist(), spec=comm.spec.tolist(), rng=rng.bit_generator.state,
                       trng=trng.bit_generator.state, urng=urng.bit_generator.state), open(state_path + ".tmp", "w"))
        os.replace(state_path + ".tmp", state_path)
        print(json.dumps(upd), f"{time.time() - t0:.0f}s", flush=True)
    open(os.path.join(args.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
