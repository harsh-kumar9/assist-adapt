"""P3: an LLM assistant that keeps learning inside the community, across a sequence of problems.

The community is the paper's model (100 agents, NK landscape). When an agent asks, the LLM sees
the agent's 20 choices and writes new values for choices 1-10. There is no oracle: what the LLM
knows about the landscape comes only from its own training. After every round we update the LLM
with GRPO: for each asker we draw G suggestions, score each with the simulator, show the first
one to the agent, and push up the better-than-average ones.

Problems come in sequence. Problem k+1 redraws 20% of problem k's payoff tables. At the end of
each problem we copy the community and run 30 rounds on the next problem with the assistant
gone (the P1 probe), then carry on with the assistant into the next problem.

The same driver runs the oracle assistants (none, sbp, pcc) on CPU with identical draws, so
LLM runs and oracle runs on a seed share landscapes, network, start and coin flips.

  python -m src.llm.online --assistant llm --reward now --seed 0 --out results/p3/now_s0
  python -m src.llm.online --assistant pcc --seed 0 --out results/p3/pcc_s0
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np

from src.nk import Landscape
from src.sim import Community, mean_pairwise_hamming, propose, step as sim_step

P_SOCIAL = 0.1
SYSTEM = "You help research teams improve their approach."
USER = ("A team's approach is 20 yes/no choices, written as digits (1 = yes).\n"
        "Choices 1-10 (you can change these): {mc}\n"
        "Choices 11-20 (fixed): {rest}\n"
        "Recommend values for choices 1-10. Reply with exactly ten digits and nothing else.")


# P4b option 2: the assistant suggests edits (which choices to flip), at most k of them
EDIT_ASK = ("Suggest up to {k} of choices 1-10 to flip. Reply with their numbers separated by commas "
            "(for example: {example}) and nothing else.")
FULL_ASK = "Recommend values for choices 1-10. Reply with exactly ten digits and nothing else."


STEP_BACK = "\nIf no change would help, repeat the current values and the team will explore on its own."


CONTEXT_HEAD = "Approaches the best-scoring teams use right now (choices 1-10 | choices 11-20, score):\n"


def digits(code, lo, hi):
    return "".join(str((int(code) >> j) & 1) for j in range(lo, hi))


def top_context(codes, fit, k, N=20, M=10):
    """The k best distinct configurations in the community, with their fitness (G-3 b)."""
    out, seen = [], set()
    for i in np.argsort(-fit, kind="stable"):
        if int(codes[i]) in seen:
            continue
        seen.add(int(codes[i]))
        out.append((int(codes[i]), float(fit[i])))
        if len(out) == k:
            break
    return out


def render(code, N=20, M=10, step_back=False, context=None, edits=0):
    bits = [(int(code) >> j) & 1 for j in range(N)]
    text = USER.format(mc="".join(map(str, bits[:M])), rest="".join(map(str, bits[M:])))
    if edits:
        text = text.replace(FULL_ASK, EDIT_ASK.format(k=edits, example=", ".join(["2", "7", "9"][:edits])))
    if context:
        lines = "".join(f"{i + 1}. {digits(c, 0, M)} | {digits(c, M, N)}, score {f:.2f}\n"
                        for i, (c, f) in enumerate(context))
        text = CONTEXT_HEAD + lines + "\n" + text
    return text + STEP_BACK if step_back else text


def parse(text, M=10):
    """First run of M binary digits (spaces and commas ignored) -> module int, or None."""
    m = re.search(r"[01]{%d}" % M, re.sub(r"[\s,]", "", text))
    if not m:
        return None
    return sum(int(ch) << j for j, ch in enumerate(m.group(0)))


def parse_edits(text, k, M=10):
    """Numbers 1..M in the reply, first k distinct ones -> bit mask of choices to flip, or None."""
    pos = []
    for x in re.findall(r"\d+", text):
        x = int(x)
        if 1 <= x <= M and x not in pos:
            pos.append(x)
    if not pos:
        return None
    return sum(1 << (p - 1) for p in pos[:k])


def chat_text(tok, user):
    """Chat-formatted prompt. Templates that already start with BOS (Gemma) have it removed, because
    the tokenizer adds it again when the text is encoded; Qwen templates have no BOS, so no change."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if tok.bos_token and text.startswith(tok.bos_token):
        text = text[len(tok.bos_token):]
    return text


def lora_targets(model_type):
    """LoRA on the language model's attention and MLP projections. Gemma 3 4B ships as a vision-language
    model, so its vision tower (which also has q/k/v projections) is excluded."""
    names = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    if model_type == "gemma3":
        return rf".*language_model.*\.({names})"
    return names.split("|")


def popcount(x):
    return bin(int(x)).count("1")


# ---------------------------------------------------------------- the LLM

class LLM:
    def __init__(self, args):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.args = torch, args
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                         target_modules=lora_targets(base.config.model_type))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
        if hasattr(base.config, "text_config"):
            base.config.text_config.use_cache = False
        self.model = get_peft_model(base, cfg)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                     lr=args.lr, weight_decay=0.0)

    def chat(self, user):
        return chat_text(self.tok, user)

    def sample(self, users, G, seed, temperature=None):
        torch = self.torch
        temperature = self.args.temperature if temperature is None else temperature
        torch.manual_seed(seed)
        enc = self.tok([self.chat(u) for u in users], return_tensors="pt", padding=True).to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(**enc, use_cache=True, do_sample=True, temperature=temperature, top_p=1.0,
                                      top_k=0, max_new_tokens=self.args.max_new_tokens,
                                      num_return_sequences=G, pad_token_id=self.tok.pad_token_id)
        P = enc["input_ids"].shape[1]
        texts = self.tok.batch_decode(out[:, P:], skip_special_tokens=True)
        return out, P, texts

    def _logp(self, seqs, P, attn):
        torch = self.torch
        pos = (attn.long().cumsum(1) - 1).clamp(min=0)
        L = seqs.shape[1]
        # logits only for the positions that predict answer tokens
        logits = self.model(input_ids=seqs, attention_mask=attn, position_ids=pos,
                            logits_to_keep=L - P + 1).logits[:, :-1]
        return torch.log_softmax(logits.float(), -1).gather(-1, seqs[:, P:, None]).squeeze(-1)

    def update(self, out, P, adv):
        """One on-policy policy-gradient step with group-relative advantages (+ optional KL)."""
        torch = self.torch
        pad, eos = self.tok.pad_token_id, self.tok.eos_token_id
        attn = (out != pad).long()
        comp = out[:, P:]
        cmask = (comp != pad).float()
        # keep the first EOS in the loss even when pad == eos
        first_pad = (comp == pad).float().cumsum(1) == 1
        cmask = torch.where((comp == eos) & first_pad, torch.ones_like(cmask), cmask)
        attn[:, P:] = cmask.long()
        adv = torch.tensor(adv, dtype=torch.float32, device=self.args.device)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        n_tok = cmask.sum().clamp(min=1)
        stats = dict(kl=0.0)
        for s in range(0, len(out), self.args.micro_batch):
            sl = slice(s, s + self.args.micro_batch)
            if not torch.any(adv[sl] != 0) and self.args.kl == 0:
                continue
            lp = self._logp(out[sl], P, attn[sl])
            loss_tok = -adv[sl, None] * lp
            if self.args.kl > 0:
                with torch.no_grad(), self.model.disable_adapter():
                    ref = self._logp(out[sl], P, attn[sl])
                d = ref - lp
                kl = torch.exp(d) - d - 1  # k3 estimator
                loss_tok = loss_tok + self.args.kl * kl
                stats["kl"] += float((kl.detach() * cmask[sl]).sum() / n_tok)
            ((loss_tok * cmask[sl]).sum() / n_tok).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return stats

    def save(self, path):
        self.model.save_pretrained(path)
        self.torch.save(self.opt.state_dict(), os.path.join(path, "opt.pt"))

    def load(self, path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.model, load_file(os.path.join(path, "adapter_model.safetensors")))
        self.opt.load_state_dict(self.torch.load(os.path.join(path, "opt.pt")))


# ---------------------------------------------------------------- rewards

def score(sugg, cur, fit_cur, F, M, kind, edit_cost):
    """Reward for one suggestion (module int or None) to an agent at `cur`."""
    if sugg is None:
        return -0.1, 0.0, 0
    mc = (1 << M) - 1
    new = (cur & ~mc) | sugg
    gain = max(0.0, float(F[new] - fit_cur)) * 10
    edit = popcount((sugg ^ cur) & mc)
    if kind == "now":
        return gain, gain, edit
    if kind == "edit":  # nudge prior: a full rewrite of the module earns (1 - edit_cost) of its gain
        return gain * (1 - edit_cost * edit / M), gain, edit
    raise ValueError(kind)


def feedback(gains, kind, rng, valid=None):
    """User feedback on G answers per asker, as deployed assistants collect it (G-3 b).

    thumbs:   the asker tries the answer and gives a thumbs up if it improved their score.
    pairwise: the asker sees two answers side by side and picks the one that is better for them;
              when both are equally (un)helpful they pick at random. Answers are paired (0,1),
              (2,3), ...; the asker is shown the first pair and adopts what they picked.
              With `valid` (G-5), a tie between a well-formed and a malformed answer goes to the
              well-formed one, as a real user would choose.
    Returns rewards (n, G) and the index of the answer the asker adopts.
    """
    n, G = gains.shape
    if kind == "thumbs":
        return (gains > 0).astype(float), np.zeros(n, np.int64)
    R = np.zeros((n, G))
    for a in range(0, G - 1, 2):
        x, y = gains[:, a], gains[:, a + 1]
        tie_pick = rng.random(n) < 0.5
        if valid is not None:
            va, vb = valid[:, a], valid[:, a + 1]
            tie_pick = np.where(va != vb, va, tie_pick)
        first = np.where(x == y, tie_pick, x > y)
        R[:, a], R[:, a + 1] = first, ~first
    return R, np.where(R[:, 0] > 0, 0, 1)


# ---------------------------------------------------------------- driver

def entropy(xs):
    _, c = np.unique(xs, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant", default="llm", help="llm | none | sbp | pcc | gated")
    ap.add_argument("--reward", default="now", choices=["now", "edit", "thumbs", "pairwise"],
                    help="now: scalar gain; thumbs: 1 if it helped; pairwise: asker picks the better of two")
    ap.add_argument("--edit-cost", type=float, default=0.8)
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--prefer-valid", action="store_true",
                    help="pairwise: ties between a well-formed and a malformed answer go to the well-formed one (G-5)")
    ap.add_argument("--context", type=int, default=0,
                    help="show the k best distinct configurations in the community (G-3 b)")
    ap.add_argument("--step-back", action="store_true",
                    help="unchanged answer = asker explores on their own; scored by that exploration (D-013)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--use", type=float, default=0.5)
    ap.add_argument("--problems", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=60, help="rounds per problem")
    ap.add_argument("--shift", type=float, default=0.2)
    ap.add_argument("--post", type=int, default=30)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    N, K, M, H = 20, 9, 10, 10
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    lands = [Landscape(N, K, M, args.rho, np.random.default_rng([args.seed, int(round(args.rho * 1000)), 0]))]
    srng = np.random.default_rng([args.seed, int(round(args.rho * 1000)), 3])
    for _ in range(args.problems):
        lands.append(lands[-1].shifted(args.shift, srng))
    comm = Community(100, N, H, 0.05, np.random.default_rng([args.seed, 2]))
    mc = np.int64((1 << M) - 1)

    llm = LLM(args) if args.assistant == "llm" else None
    state_path = os.path.join(args.out, "state.json")
    log_path = os.path.join(args.out, "rounds.jsonl")
    probe_path = os.path.join(args.out, "probes.jsonl")
    rng = np.random.default_rng([args.seed, 1])
    codes = comm.codes0.copy()
    start = 0
    if os.path.exists(state_path):  # resume at a problem boundary
        st = json.load(open(state_path))
        start, codes = st["next_problem"], np.array(st["codes"], dtype=np.int64)
        rng.bit_generator.state = st["rng"]
        if llm:
            llm.load(os.path.join(args.out, "adapter"))
        print(f"resumed at problem {start}", flush=True)

    t0 = time.time()
    for k in range(start, args.problems):
        F = lands[k].F
        logf = open(log_path, "a")
        for t in range(args.rounds):
            n = len(codes)
            u_social, u_query, u_ai = rng.random(n), rng.random(n), rng.random(n)
            col = rng.integers(0, comm.H, n)
            fit = F[codes]
            social = u_social < P_SOCIAL
            query = ~social & (u_query < args.use) if args.assistant != "none" else np.zeros(n, bool)
            prop = codes ^ (np.int64(1) << comm.spec[np.arange(n), col])
            fit_pad = np.append(fit, -np.inf)
            best_nb = comm.nbr[np.arange(n), np.argmax(fit_pad[comm.nbr], 1)]
            prop = np.where(social, codes[np.minimum(best_nb, n - 1)], prop)
            idx = np.flatnonzero(query)
            rec = dict(problem=k, round=t, n_query=int(len(idx)))
            if len(idx) and args.assistant == "gated":  # help with whichever gains more, else step back
                pcc = propose("pcc", codes, fit, F, M, idx, u_ai[idx])
                sbp = propose("sbp", codes, fit, F, M, idx, u_ai[idx])
                gp, gs = F[pcc] - fit[idx], F[sbp] - fit[idx]
                best = np.where(gp >= gs, pcc, sbp)
                helps = np.maximum(gp, gs) > 0
                prop[idx] = np.where(helps, best, prop[idx])
                rec.update(step_back=float(1 - helps.mean()), full=float((helps & (gp >= gs)).mean()))
            elif len(idx) and args.assistant in ("sbp", "pcc"):
                prop[idx] = propose(args.assistant, codes, fit, F, M, idx, u_ai[idx])
                shown = prop[idx] & mc
                rec.update(shown_edit=float(np.mean([popcount((s ^ c) & mc) for s, c in zip(shown, codes[idx])])))
            elif len(idx) and llm:
                ctx = top_context(codes, fit, args.context) if args.context else None
                out, P, texts = llm.sample([render(codes[i], step_back=args.step_back, context=ctx) for i in idx], args.G,
                                           seed=args.seed * 100003 + k * 1000 + t)
                explore_gain = np.maximum(0, F[prop[idx]] - fit[idx]) * 10  # what the asker gets on their own
                R, gains, edits, mods = [], [], [], []
                for gi, i in enumerate(idx):
                    row = []
                    for j in range(args.G):
                        s = parse(texts[gi * args.G + j])
                        r, g, e = score(s, int(codes[i]), float(fit[i]), F, M,
                                        args.reward if args.reward in ("now", "edit") else "now", args.edit_cost)
                        if args.step_back and s is not None and s == (int(codes[i]) & int(mc)):
                            r, g = float(explore_gain[gi]), float(explore_gain[gi])  # stepping back = exploring
                        row.append(r)
                        mods.append(-1 if s is None else s)
                        gains.append(g)
                        edits.append(e)
                    R.append(row)
                R = np.array(R)
                mods = np.array(mods).reshape(len(idx), args.G)
                pick = np.zeros(len(idx), np.int64)  # which sample the asker sees and may adopt
                if args.reward in ("thumbs", "pairwise"):
                    R, pick = feedback(np.array(gains).reshape(len(idx), args.G), args.reward,
                                       np.random.default_rng([args.seed, 9, k, t]),
                                       valid=(mods >= 0) if args.prefer_valid else None)
                adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)
                shown = mods[np.arange(len(idx)), pick]
                ok = shown >= 0
                back = ok & (shown == (codes[idx] & mc)) if args.step_back else np.zeros(len(idx), bool)
                give = ok & ~back
                prop[idx[give]] = (codes[idx[give]] & ~mc) | shown[give]
                stats = llm.update(out, P, adv.reshape(-1))
                rec.update(step_back=float(back.mean()))
                g_arr, e_arr = np.array(gains).reshape(len(idx), args.G), np.array(edits).reshape(len(idx), args.G)
                g_sh, e_sh = g_arr[np.arange(len(idx)), pick], e_arr[np.arange(len(idx)), pick]
                rec.update(valid=float(ok.mean()), reward=float(R.mean()), gain_shown=float(g_sh.mean()),
                           improve_shown=float((g_sh > 0).mean()), shown_edit=float(e_sh[ok].mean()) if ok.any() else None,
                           shown_unique=float(len(np.unique(shown[ok])) / max(1, ok.sum())),
                           shown_entropy=entropy(shown[ok]) if ok.any() else None,
                           top_share=float(np.unique(shown[ok], return_counts=True)[1].max() / ok.sum()) if ok.any() else None,
                           group_distinct=float(np.mean([len(set(r)) for r in mods])), kl=stats["kl"],
                           example=texts[0][:40])
                # oracle references for the same askers
                pcc = propose("pcc", codes, fit, F, M, idx, u_ai[idx])
                sbp = propose("sbp", codes, fit, F, M, idx, u_ai[idx])
                rec.update(pcc_gain=float(np.maximum(0, F[pcc] - fit[idx]).mean() * 10),
                           sbp_gain=float(np.maximum(0, F[sbp] - fit[idx]).mean() * 10))
            codes = np.where(F[prop] > fit, prop, codes)
            rec.update(fitness=float(F[codes].mean()), diversity=mean_pairwise_hamming(codes, N),
                       modules=int(len(np.unique(codes & mc))), seconds=round(time.time() - t0, 1))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if t % 10 == 0:
                print(json.dumps({k_: rec[k_] for k_ in rec if k_ != "example"}), flush=True)
        logf.close()
        # probes: copy the community and run without the assistant on this problem and the next
        with open(probe_path, "a") as pf:
            for li, pl in ((0, lands[k]), (1, lands[k + 1])):
                prng = np.random.default_rng([args.seed, 7, k, li, 0])
                pc, vals = codes.copy(), []
                for _ in range(args.post):
                    pc = sim_step(pc, pl.F, comm, "none", 0.0, 0.1, M, prng)
                    vals.append(float(pl.F[pc].mean()))
                pf.write(json.dumps(dict(problem=k, landscape="same" if li == 0 else "next", ai="off",
                                         mean=float(np.mean(vals)), final=vals[-1])) + "\n")
        if llm:
            llm.save(os.path.join(args.out, "adapter"))
            llm.model.save_pretrained(os.path.join(args.out, f"adapter_p{k}"))  # kept per problem (G-3)
        json.dump(dict(next_problem=k + 1, codes=codes.tolist(), rng=rng.bit_generator.state),
                  open(state_path + ".tmp", "w"))
        os.replace(state_path + ".tmp", state_path)
        print(f"problem {k} done, {time.time() - t0:.0f}s", flush=True)
    open(os.path.join(args.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
