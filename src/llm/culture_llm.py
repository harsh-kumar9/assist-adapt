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
  local      thumbs, but only if the answer changes at most `reach` bits of the asker's approach
             (improve my approach, do not replace it; P4a: reach, not novelty, sets the culture's diversity)
  spread     thumbs / (1 + other askers this epoch who adopted the same module from the assistant)

GRPO standardises rewards within each group of G answers, so a reward that only rescales thumbs changes
nothing in a group with one helpful answer, and with ~3% of answers helping most groups have none. The
next three also score answers that do not help (an invalid answer scores below any valid one):

  dup        thumbs - beta * (copies of this answer in the group - 1) / (G - 1): credit shared among
             identical answers (implicit fitness sharing; Smith, Forrest & Perelson 1993; Li et al. 2025)
  crowd      thumbs - lam * share of this epoch's askers whose module is within `radius` flips of the answer
             (fitness sharing with a radius; Goldberg & Richardson 1987)
  rate       10 * (F(answer) - F(asker)), signed: how much better or worse the team found it

Tweak or leap (`--edits k --leap q`; Miu et al. 2018, the best design in the P4a screen): each request is an
edit of at most k choices, except that with probability q it asks for a new approach that no listed team
uses. Edits are scored by `--reward`; leaps earn 1 only if they help, no team holds them, and they change
more than k choices.

Memory (`--memory k`, with `--edits`; step 2 of getting past random): the prompt also lists up to k modules that
the assistant's answers moved some team to earlier this epoch (the provider's thumbs log), within `edits` flips
of the asker's module, most recently helpful first, as the flips that reach them. It starts empty each epoch.
The sim's mem_now_recent (src/culture.py) answers with the first one every time. `--memory-share q` shows the
memory on a share q of requests only (drawn per request; the rest get the plain edit prompt), as the sim's
mem_now_recent_p<pct>; `--memory-reward` scores prompts that show a listing with a different reward than
`--reward` (e.g. thumbs there, dup on the rest), so the provider, not the learner, sets the mix.

`--kl` adds the usual penalty toward the untrained model. `--edits k` changes the interface: instead of a
full approach the model names at most k choices to flip in the asker's approach (reach k by design). Between epochs the problem changes and the
updated model is redeployed. This is the loop of a deployed model retrained on its users' feedback.

  python -m src.llm.culture_llm --reward hcreative --seed 0 --out results/p4b/hcreative_s0
"""
import argparse
import json
import os
import time
import types

import numpy as np

from src.llm.online import LLM, P_SOCIAL, parse, parse_edits, popcount, render, top_context
from src.nk import Landscape
from src.sim import Community, mean_pairwise_hamming, step as sim_step

N, K, M, H = 20, 9, 10, 10
MC = np.int64((1 << M) - 1)


def answer(text, asker, edits):
    """The module the reply proposes: ten digits (full answer), or the asker's module with at most
    `edits` choices flipped (edit interface). None if the reply has neither."""
    if not edits:
        return parse(text)
    mask = parse_edits(text, edits)
    return None if mask is None else (int(asker) & int(MC)) ^ mask


INVALID = {"dup": -1.0, "rate": -10.0}  # crowd: -1 - lam. dup subtracts its group penalty on top


def reward(kind, sugg, asker, fit_asker, F, culture_mods, lam, reach=3, crowd=None, own=None, share=None):
    """Returns (reward, gain). sugg is a module int or None. spread: crowd counts the modules that askers
    adopted from the assistant this epoch, own is the module this asker adopted (None if none). crowd:
    share[x] is the share of this epoch's askers within `radius` flips of module x."""
    if sugg is None:
        return (-1.0 - lam if kind == "crowd" else INVALID.get(kind, 0.0)), 0.0
    new = (asker & ~MC) | sugg
    gain = max(0.0, float(F[new] - fit_asker)) * 10
    helped = 1.0 if gain > 0 else 0.0
    if kind in ("thumbs", "dup"):
        return helped, gain
    if kind == "rate":
        return float(F[new] - fit_asker) * 10, gain
    if kind == "crowd":
        return helped - lam * float(share[sugg]), gain
    if kind == "spread":
        return helped / (1 + crowd.get(sugg, 0) - (sugg == own)), gain
    if kind == "local":
        return helped * (popcount(sugg ^ (asker & MC)) <= reach), gain
    if kind == "leap":
        return helped * (sugg not in culture_mods) * (popcount(sugg ^ (asker & MC)) > reach), gain
    if kind == "pcreative":
        return helped * (1 + lam * popcount(sugg ^ (asker & MC)) / M), gain
    if kind == "hcreative":
        d = min(popcount(sugg ^ c) for c in culture_mods)
        return helped * (1 + lam * d / M), gain
    raise ValueError(kind)


def listed(mem, code, reach, k):
    """Up to k remembered modules within 1..reach flips of this team's module, most recently helpful first."""
    if not k or not mem:
        return []
    m = int(code) & int(MC)
    near = [(r, x) for x, r in mem.items() if 0 < popcount(x ^ m) <= reach]
    return [x for _, x in sorted(near, key=lambda p: -p[0])[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="thumbs",
                    choices=["frozen", "thumbs", "spread", "dup", "crowd", "rate", "pcreative", "hcreative", "local"])
    ap.add_argument("--lam", type=float, default=1.0, help="novelty bonus weight; crowding penalty weight (crowd)")
    ap.add_argument("--beta", type=float, default=0.5, help="duplicate penalty weight (dup)")
    ap.add_argument("--radius", type=int, default=2, help="crowding radius in flips (crowd)")
    ap.add_argument("--reach", type=int, default=3, help="most bits a rewarded answer may change (local)")
    ap.add_argument("--edits", type=int, default=0,
                    help="edit interface: the model names at most this many choices to flip (0 = full answers)")
    ap.add_argument("--leap", type=float, default=0.0,
                    help="tweak or leap: share of requests that ask for a new approach instead of an edit")
    ap.add_argument("--memory", type=int, default=0,
                    help="list up to this many modules that helped some team this epoch, within --edits flips")
    ap.add_argument("--memory-share", type=float, default=1.0,
                    help="share of requests that are shown the memory (drawn per request)")
    ap.add_argument("--memory-reward", default=None, choices=["thumbs", "dup"],
                    help="reward for prompts that show a listing (default: --reward)")
    ap.add_argument("--menu", type=int, default=1,
                    help="answers per request; the team tries each and keeps the best that helps")
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
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature while deployed")
    ap.add_argument("--update-temperature", type=float, default=None,
                    help="sampling temperature for the per-epoch update (default: same as deployed)")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=48)
    ap.add_argument("--gen-batch", type=int, default=0,
                    help="most sequences per generate call (0 = all at once); the update batch stays --chunk")
    ap.add_argument("--save-epochs", default="1,5,10,20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert not args.memory or args.edits, "--memory needs --edits"
    assert args.memory or (args.memory_share == 1 and not args.memory_reward), "--memory-share/-reward need --memory"

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
    lrng = np.random.default_rng([args.seed, 13])  # which requests ask for a leap
    mrng = np.random.default_rng([args.seed, 14])  # which requests are shown the memory (--memory-share)
    codes = comm.codes0.copy()
    llm = LLM(types.SimpleNamespace(**{k: getattr(args, k) for k in (
        "model", "device", "lora_r", "lr", "temperature", "max_new_tokens", "micro_batch", "kl", "gen_batch")}))

    state_path = os.path.join(args.out, "state.json")
    start = 0
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        start, codes = st["next_epoch"], np.array(st["codes"], dtype=np.int64)
        comm.spec = np.array(st["spec"], dtype=np.int64)
        rng.bit_generator.state, trng.bit_generator.state, urng.bit_generator.state = st["rng"], st["trng"], st["urng"]
        if "lrng" in st:
            lrng.bit_generator.state = st["lrng"]
        if "mrng" in st:
            mrng.bit_generator.state = st["mrng"]
        llm.load(os.path.join(args.out, "adapter"))
        print(f"resumed at epoch {start}", flush=True)

    save_epochs = {int(x) for x in args.save_epochs.split(",") if x}
    t0 = time.time()
    for e in range(start, args.epochs):
        F = lands[e].F
        log = []  # requests served this epoch: (asker code, asker fitness, context, culture modules, leap, adopted module,
        #           listed modules)
        crowd = {}  # module -> askers who adopted it from the assistant this epoch
        mem = {}  # module -> last round an answer moved a team to it this epoch
        reqs = []  # every request served: round, team slot, asker code, answer shown (-1 if none), helped
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
            shown, sizes, helped, leaps, menu_distinct, copies = [], [], [], [], [], []
            if len(idx):
                leap = lrng.random(len(idx)) < args.leap if args.leap else np.zeros(len(idx), bool)
                lists = [listed(mem, codes[i], args.edits, args.memory) for i in idx]
                if args.memory_share < 1:
                    lists = [ls if d else [] for ls, d in zip(lists, mrng.random(len(idx)) < args.memory_share)]
                _, _, texts = llm.sample([render(codes[i], context=ctx, edits=args.edits, leap=bool(lp), memory=ls)
                                          for i, lp, ls in zip(idx, leap, lists)], args.menu,
                                         seed=args.seed * 100003 + e * 1000 + t)
                held = set(culture.tolist())
                for j, i in enumerate(idx):
                    opts = [answer(texts[j * args.menu + c], codes[i], 0 if leap[j] else args.edits) for c in range(args.menu)]
                    opts = [o for o in opts if o is not None]
                    menu_distinct.append(len(set(opts)))
                    s = max(opts, key=lambda o: F[(codes[i] & ~MC) | o]) if opts else None  # the team tries each
                    cand = None if s is None else (codes[i] & ~MC) | s
                    good = s is not None and F[cand] > fit[i]
                    log.append((int(codes[i]), float(fit[i]), ctx, culture, bool(leap[j]), int(s) if good else None, lists[j]))
                    reqs.append((t, int(i), int(codes[i]), -1 if s is None else int(s), int(good), len(lists[j]),
                                 int(s in lists[j])))
                    if s is None:
                        continue
                    if lists[j]:
                        copies.append((s in lists[j], good))
                    if good:
                        crowd[int(s)] = crowd.get(int(s), 0) + 1
                        mem[int(s)] = t  # listed from the next round on
                    shown.append(s)
                    sizes.append(popcount(s ^ (int(codes[i]) & int(MC))))
                    helped.append(good)
                    if leap[j]:
                        leaps.append((good, s not in held))
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
                           top_share=float(cnt.max() / cnt.sum()), shown_unique=float(len(vals) / len(shown)),
                           edit_size=float(np.mean(sizes)))
                if args.menu > 1:
                    rec.update(menu_distinct=float(np.mean(menu_distinct)))
                if args.memory:
                    rec.update(listed=float(np.mean([len(ls) > 0 for ls in lists])))
                if copies:
                    cp = np.array(copies, bool)
                    rec.update(copied=float(cp[:, 0].mean()),
                               copy_help=float(cp[cp[:, 0], 1].mean()) if cp[:, 0].any() else None,
                               other_help=float(cp[~cp[:, 0], 1].mean()) if (~cp[:, 0]).any() else None)
                if leaps:
                    lp = np.array(leaps, float)
                    rec.update(n_leap=len(leaps), leap_improve=float(lp[:, 0].mean()), leap_new=float(lp[:, 1].mean()),
                               leap_good=float((lp[:, 0] * lp[:, 1]).mean()))
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
            R_all, G_all, D_all, H_all, C_all = [], [], [], [], []  # H_all: (helpful answers, distinct helpful) per group
            # C_all: share of the G samples that copy a listed module, per prompt with a listing
            L_all = []  # whether the prompt showed a listing, per group
            share = None
            if args.reward == "crowd":
                hist = np.bincount([a & int(MC) for a, *_ in log], minlength=1 << M)
                ball = [m for m in range(1 << M) if popcount(m) <= args.radius]
                share = sum(hist[np.arange(1 << M) ^ m] for m in ball) / len(log)
            for c0 in range(0, len(pick), args.chunk):
                batch = [log[k] for k in pick[c0:c0 + args.chunk]]
                out, P, texts = llm.sample([render(a, context=c, edits=args.edits, leap=lp, memory=ls)
                                            for a, _, c, _, lp, _, ls in batch],
                                           args.G,
                                           seed=args.seed * 100003 + e * 1000 + 999 + c0,
                                           temperature=args.update_temperature)
                R = np.zeros((len(batch), args.G))
                for bi, (a, fa, _, cult, lp, own, ls) in enumerate(batch):
                    suggs = [answer(texts[bi * args.G + j], a, 0 if lp else args.edits) for j in range(args.G)]
                    if ls:
                        C_all.append(np.mean([x in ls for x in suggs]))
                    kind = "leap" if lp else args.memory_reward if ls and args.memory_reward else args.reward
                    good = set()
                    for j, sugg in enumerate(suggs):
                        R[bi, j], g = reward(kind, sugg, a, fa, F, cult, args.lam,
                                             args.edits if lp else args.reach, crowd, own, share)
                        G_all.append(g)
                        if g > 0:
                            good.add(sugg)
                    if kind == "dup":
                        n = {s: suggs.count(s) for s in set(suggs) if s is not None}
                        R[bi] -= args.beta * np.array([1.0 if s is None else (n[s] - 1) / (args.G - 1) for s in suggs])
                    D_all.append(len({texts[bi * args.G + j] for j in range(args.G)}))
                    L_all.append(bool(ls))
                    H_all.append((sum(s in good for s in suggs), len(good)))
                adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)
                stats = llm.update(out, P, adv.reshape(-1))
                R_all.append(R.mean())
            h = np.array(H_all, float)
            upd.update(reward=float(np.mean(R_all)), gain=float(np.mean(G_all)), group_distinct=float(np.mean(D_all)),
                       group_any=float((h[:, 0] > 0).mean()), group_multi=float((h[:, 1] > 1).mean()),
                       kl=stats["kl"])
            if C_all:
                upd.update(update_copy=float(np.mean(C_all)))
            if args.memory_share < 1:  # distinct texts of 8 on prompts with and without a listing
                d, l = np.array(D_all, float), np.array(L_all, bool)
                upd.update(group_distinct_listed=float(d[l].mean()) if l.any() else None,
                           group_distinct_unlisted=float(d[~l].mean()) if (~l).any() else None)
        with open(os.path.join(args.out, "rounds.jsonl"), "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out, "epochs.jsonl"), "a") as f:
            f.write(json.dumps(upd) + "\n")
        names = ("t", "slot", "code", "answer", "helped", "n_listed", "copied")
        cols = dict(zip(names if args.memory else names[:5], map(list, zip(*reqs)))) if reqs else {}
        with open(os.path.join(args.out, "requests.jsonl"), "a") as f:
            f.write(json.dumps(dict(epoch=e, **cols)) + "\n")
        llm.save(os.path.join(args.out, "adapter"))
        if e + 1 in save_epochs:
            llm.model.save_pretrained(os.path.join(args.out, f"adapter_e{e + 1}"))
        json.dump(dict(next_epoch=e + 1, codes=codes.tolist(), spec=comm.spec.tolist(), rng=rng.bit_generator.state,
                       trng=trng.bit_generator.state, urng=urng.bit_generator.state, lrng=lrng.bit_generator.state,
                       mrng=mrng.bit_generator.state),
                  open(state_path + ".tmp", "w"))
        os.replace(state_path + ".tmp", state_path)
        print(json.dumps(upd), f"{time.time() - t0:.0f}s", flush=True)
    open(os.path.join(args.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
