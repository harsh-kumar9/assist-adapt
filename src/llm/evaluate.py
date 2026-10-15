"""G-3 (a): run trained assistants frozen for many fresh communities.

For one training seed and modularity, load every trained adapter for that seed (R_now, KL, step
back), the untrained base model, and the oracle assistants. Each one is the assistant, with no
further learning, for C fresh communities (new networks, starting points and coin flips) on the
problem the adapters last trained on. After T assisted rounds we run the P1 probes: 30 more
rounds with the assistant on the same problem, and 30 rounds on the next problem with the
assistant removed. All assistants see the same communities and the same draws, so every
comparison is paired by community.

All communities move in lockstep; each round, every asker's prompt across all communities goes
to vLLM in one call.

  python -m src.llm.evaluate --rho 1.0 --seed 0 --runs-dir results/p3 --out results/p3eval
"""
import argparse
import json
import os
import time

import numpy as np

from src.llm.online import P_SOCIAL, SYSTEM, parse, popcount, render, top_context
from src.nk import Landscape
from src.sim import Community, mean_pairwise_hamming, propose

N, K, M, H = 20, 9, 10, 10
MC = np.int64((1 << M) - 1)


def landscapes(rho, seed, problems, shift):
    lands = [Landscape(N, K, M, rho, np.random.default_rng([seed, int(round(rho * 1000)), 0]))]
    srng = np.random.default_rng([seed, int(round(rho * 1000)), 3])
    for _ in range(problems):
        lands.append(lands[-1].shifted(shift, srng))
    return lands


class Oracle:
    def __init__(self, kind):
        self.name, self.kind = kind, kind

    def advise(self, batch):
        """batch: list of (codes, fit, F, idx, u_ai, prop). Returns proposals for idx, plus stats."""
        outs = []
        for codes, fit, F, idx, u, prop in batch:
            if self.kind in ("sbp", "pcc"):
                outs.append(propose(self.kind, codes, fit, F, M, idx, u))
            else:  # gated: whichever helps more, else the asker explores
                pcc = propose("pcc", codes, fit, F, M, idx, u)
                sbp = propose("sbp", codes, fit, F, M, idx, u)
                gp, gs = F[pcc] - fit[idx], F[sbp] - fit[idx]
                outs.append(np.where(np.maximum(gp, gs) > 0, np.where(gp >= gs, pcc, sbp), prop[idx]))
        return outs, {}


class LLMAssistant:
    def __init__(self, engine, tok, name, lora_path=None, lora_id=0, step_back=False, temperature=1.0, context=0):
        self.engine, self.tok, self.name = engine, tok, name
        self.lora_path, self.lora_id, self.step_back, self.temperature = lora_path, lora_id, step_back, temperature
        self.context = context

    def advise(self, batch):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        prompts, owners = [], []
        for bi, (codes, fit, F, idx, u, prop) in enumerate(batch):
            ctx = top_context(codes, fit, self.context) if self.context else None
            for i in idx:
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": render(codes[i], step_back=self.step_back, context=ctx)}]
                prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                owners.append(bi)
        if not prompts:
            return [np.zeros(0, np.int64) for _ in batch], {}
        sp = SamplingParams(temperature=self.temperature, top_p=1.0, max_tokens=16)  # pure sampling, as in training
        lora = LoRARequest(self.name, self.lora_id, self.lora_path) if self.lora_path else None
        res = self.engine.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        mods = [parse(r.outputs[0].text) for r in res]
        outs, stats = [], dict(valid=0, back=0, n=0, shown=[])
        pos = 0
        for codes, fit, F, idx, u, prop in batch:
            o = prop[idx].copy()  # invalid answer or step back: the asker explores
            for j, i in enumerate(idx):
                s = mods[pos + j]
                stats["n"] += 1
                if s is None:
                    continue
                stats["valid"] += 1
                if self.step_back and s == (int(codes[i]) & int(MC)):
                    stats["back"] += 1
                    continue
                o[j] = (int(codes[i]) & ~int(MC)) | s
                stats["shown"].append(s)
            pos += len(idx)
            outs.append(o)
        return outs, stats


def round_all(state, F, assistant, use):
    """One round for every community. state: list of dicts with codes, comm, rng."""
    batch, pre = [], []
    for st in state:
        codes, comm, rng = st["codes"], st["comm"], st["rng"]
        n = len(codes)
        u_social, u_query, u_ai = rng.random(n), rng.random(n), rng.random(n)
        col = rng.integers(0, comm.H, n)
        fit = F[codes]
        social = u_social < P_SOCIAL
        query = ~social & (u_query < use) if assistant is not None else np.zeros(n, bool)
        prop = codes ^ (np.int64(1) << comm.spec[np.arange(n), col])
        fit_pad = np.append(fit, -np.inf)
        best_nb = comm.nbr[np.arange(n), np.argmax(fit_pad[comm.nbr], 1)]
        prop = np.where(social, codes[np.minimum(best_nb, n - 1)], prop)
        idx = np.flatnonzero(query)
        batch.append((codes, fit, F, idx, u_ai[idx], prop))
        pre.append((fit, prop, idx))
    stats = {}
    if assistant is not None:
        outs, stats = assistant.advise(batch)
        for (fit, prop, idx), o in zip(pre, outs):
            prop[idx] = o
    for st, (fit, prop, idx) in zip(state, pre):
        st["codes"] = np.where(F[prop] > fit, prop, st["codes"])
    return stats


def evaluate(assistant, comms, lands, k, T, post, use, seed, tag):
    """Assisted for T rounds on lands[k], then probes. Returns per-community metrics."""
    F, Fn = lands[k].F, lands[k + 1].F
    C = len(comms)
    state = [dict(codes=c.codes0.copy(), comm=c, rng=np.random.default_rng([seed, 51, ci]))
             for ci, c in enumerate(comms)]
    during = np.zeros(C)
    shown, n_q, n_valid, n_back = [], 0, 0, 0
    for t in range(T):
        stt = round_all(state, F, assistant, use)
        during += np.array([F[s["codes"]].mean() for s in state])
        if stt:
            shown += stt["shown"]
            n_q += stt["n"]
            n_valid += stt["valid"]
            n_back += stt["back"]
    out = dict(assistant=tag, during=(during / T).tolist(),
               div_T=[mean_pairwise_hamming(s["codes"], N) for s in state])
    snap = [s["codes"].copy() for s in state]
    for key, FF, on in (("same_off", F, False), ("next_off", Fn, False), ("same_on", F, True)):
        if on and assistant is None:
            out[key] = out["same_off"]
            continue
        st2 = [dict(codes=snap[ci].copy(), comm=c, rng=np.random.default_rng([seed, 52, ci, int(on), int(FF is Fn)]))
               for ci, c in enumerate(comms)]
        acc = np.zeros(C)
        for t in range(post):
            round_all(st2, FF, assistant if on else None, use)
            acc += np.array([FF[s["codes"]].mean() for s in st2])
        out[key] = (acc / post).tolist()
    if shown:
        vals, cnt = np.unique(np.array(shown), return_counts=True)
        out.update(top_share=float(cnt.max() / cnt.sum()), distinct_answers=int(len(vals)))
    if n_q:
        out.update(valid=n_valid / n_q, step_back=n_back / n_q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs-dir", required=True, help="where <arm>_s<seed>/adapter live")
    ap.add_argument("--arms", default="now,kl,stepback")
    ap.add_argument("--problem", default="3", help="landscape index the adapters last trained on; "
                    "a comma list evaluates several checkpoints in one job")
    ap.add_argument("--adapter", default="adapter", help="adapter dir name inside each run; 'auto' = adapter_p<k>")
    ap.add_argument("--communities", type=int, default=200)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--post", type=int, default=30)
    ap.add_argument("--use", type=float, default=0.5)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--no-llm", action="store_true", help="oracles only (CPU)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg0 = json.load(open(os.path.join(args.runs_dir, f"{args.arms.split(',')[0]}_s{args.seed}", "config.json")))
    lands = landscapes(args.rho, args.seed, cfg0["problems"], cfg0["shift"])
    comms = [Community(100, N, H, 0.05, np.random.default_rng([args.seed, 50, c])) for c in range(args.communities)]
    os.makedirs(args.out, exist_ok=True)
    engine = tok = None
    if not args.no_llm:
        from transformers import AutoTokenizer
        from vllm import LLM
        tok = AutoTokenizer.from_pretrained(args.model)
        engine = LLM(model=args.model, enable_lora=True, max_lora_rank=16, max_loras=1, dtype="bfloat16",
                     gpu_memory_utilization=0.85, max_model_len=512, enable_prefix_caching=True, seed=args.seed)
    t0 = time.time()
    for k in [int(x) for x in str(args.problem).split(",")]:
        adapter = f"adapter_p{k}" if args.adapter == "auto" else args.adapter
        path = os.path.join(args.out, f"rho{args.rho:g}_s{args.seed}_p{k}_{adapter}.json")
        done = json.load(open(path)) if os.path.exists(path) else {}
        jobs = [("none", None), ("sbp", Oracle("sbp")), ("pcc", Oracle("pcc")), ("gated", Oracle("gated"))]
        if engine is not None:
            jobs.append(("base", LLMAssistant(engine, tok, "base", context=cfg0.get("context", 0))))
            for i, arm in enumerate(args.arms.split(",")):
                run = os.path.join(args.runs_dir, f"{arm}_s{args.seed}")
                rc = json.load(open(os.path.join(run, "config.json")))
                jobs.append((arm, LLMAssistant(engine, tok, arm, lora_path=os.path.abspath(os.path.join(run, adapter)),
                                               lora_id=100 * (k + 1) + i + 1, step_back=rc.get("step_back", False),
                                               context=rc.get("context", 0))))
        for tag, assistant in jobs:
            if tag in done:
                continue
            done[tag] = evaluate(assistant, comms, lands, k, args.T, args.post, args.use, args.seed, tag)
            json.dump(done, open(path + ".tmp", "w"))
            os.replace(path + ".tmp", path)
            r = done[tag]
            print(f"p{k} {tag:9s} during {np.mean(r['during']):.3f} same_on {np.mean(r['same_on']):.3f} "
                  f"next_off {np.mean(r['next_off']):.3f} div_T {np.mean(r['div_T']):.2f} "
                  f"top_share {r.get('top_share')} distinct {r.get('distinct_answers')} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
