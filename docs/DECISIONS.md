# Decisions

Status: `paper` = copied from Fazelpour, O'Brien & Rubin (arXiv:2608.19390), no judgment involved.
`proposed` = my call for the pilot, waiting for Harsh. `ratified` = Harsh agreed.

## D-001 Scope: one model, one new probe (ratified 2026-10-13)

We start from the paper's NK model unchanged and add one thing: after T rounds with the AI, the
problem changes and the AI goes away. We drop skill decay, preference change, assistant
retraining, archives and cultural transmission, LLM-simulated humans, learned reward models and
LLM judges.

Why: each of these adds a free assumption. If the first effect shows up in the published model
plus one probe, nobody can say we built it in. Each dropped piece can come back later as its own
experiment if the first result holds.

## D-002 Model parameters (paper)

N = 20 decisions, M = 10 in the computational module, K = 9, H = 10 decisions per agent's
specialisation set, 100 agents, p_social = 0.1, 100 rounds, connected Erdos-Renyi network.
Modularity rho in {0.6, ..., 1.0}, AI use rate in {0.125, ..., 0.875}. AI designs: NP (copy the
computational module of the best agent), TDR (same, from a random top-10% agent), SBP (best
single flip in the computational module for this agent). Agents adopt anything only if it is
better. One action per round. Measures: mean fitness, mean pairwise Hamming distance and its AUC.

## D-003 Things the paper leaves open (ratified 2026-10-13)

- Network density: p_edge = 0.05 (mean degree about 5). The paper does not say. Check 0.1 later.
- Fitness scale: (f / max f)^8, the Lazer & Friedman (2007) transform. The paper says
  "transformed and normalized so that max f(d) = 1". Every rule in the model compares fitness
  values, so the transform does not change any trajectory, only the numbers we report.
- Synchronous updates: everyone acts on the state at the start of the round.
- Ties: argmax takes the first index. Payoffs are continuous, so ties only happen between agents
  holding the same configuration.
- Same landscape, network, starting points and per-round coin flips across conditions for a
  given seed, so all comparisons are paired.

## D-004 The shift (ratified 2026-10-13)

The related problem keeps the dependency structure and redraws the payoff tables of 20% of the
decisions (4 of 20), picked at random across both modules, independent of anything in the run.
We also run shift = 0 (same problem) so we can separate "the problem changed" from "the AI left".
Probe times T in {10, 30, 60, 100}; 30 rounds after each probe.

Why 20%: enough that the old optimum is usually no longer optimal, small enough that old
experience still helps. 50% is a later sensitivity check.

## D-005 New assistant rules (ratified 2026-10-13)

Both pick from the same candidates SBP sees (the M single flips of the agent's computational
module) and only among improving flips:

- SBP-rand(q): a random flip among the q best.
- SBP-div(q): among the q best, the flip that moves the agent furthest from the rest of the
  population (largest increase in summed Hamming distance), ties broken by fitness.

q in {3, all}. Same information and the same evaluation budget as SBP. Rank-based, so the scale
transform in D-003 does not matter. Randomising and diversifying are separate rules so we can
tell "less greedy" apart from "diversity-seeking".

## D-006 P1 outcomes (ratified 2026-10-13)

- Fitness while assisted: mean fitness over rounds 1..T. This is what someone evaluating the
  assistant during use would see.
- Fitness after: mean fitness over the 30 rounds after the probe, in four cells:
  same problem or shifted, AI on or off. The primary cell is shifted + AI off.
- Diversity at T.

All as percent difference from the no-AI community on the same seed, with a 95% CI from the
paired differences. Absolute fitness, not improvement from the start of the probe, so an
assistant cannot look good by leaving people with worse starting points.

## D-007 Compute (Harsh's instruction, 2026-10-13)

CPU work runs directly on ada (`scripts/ada_sweep.sh`, nohup, a lock file against double
launches, finished shards skipped). mira and vega are for GPU jobs only, through SLURM. P0-P2 are
CPU-only. P3 is the first stage that needs GPUs.

## D-008 P2 learner (superseded by D-011)

A small selector over the same candidate flips: softmax over a linear score of a few features
(fitness gain, gain rank, how common the agent's current value is on that decision, round).
Fit with an evolution strategy on the episode reward, trained on one set of landscape seeds and
tested on held-out seeds. Three rewards: fitness while assisted only; that plus fitness after
with the AI still on; that plus fitness after a shift with the AI off. A neural policy with PPO
only if the linear one is clearly too weak.

Why ES and a linear policy: a handful of parameters, no credit-assignment machinery, and the
learned weights are readable. If this finds nothing the simple rules miss, a bigger learner is
unlikely to be worth it.

## D-009 LLM and human stages deferred (ruled 2026-10-13: LLM path, no human study)

Decide after P2. Options: (a) GRPO an LLM assistant (Qwen3.5-9B on mira/vega) on a text version
of the task with the P2 rewards, (b) a browser version of the task with people, following the
design of Kumar et al. 2024 (assisted rounds, then unassisted rounds on a shifted problem).

## D-010 P1b: copy-based rules (ratified 2026-10-13)

P1 at 1000 seeds put the effect in the copy-based assistants, not the single-flip ones. SBP
barely reduces diversity, so choosing among its flips had nothing to fix. NP and TDR at rho = 1
cut diversity by about half within 10 rounds, and NP ranks above SBP while assisted and below it
after a shift with the AI off (14 flips at Bonferroni p < 4.4e-5, all NP vs the SBP family).

So before P2 we ran the simple rules where the effect is, on the same seeds, rho in
{0.8, 0.9, 1.0}:

- TDR-25%: TDR with the pool widened to the top 25%.
- TDR-inverse: TDR, but a top-10% agent is drawn with probability proportional to
  1 / (number of agents already using its computational module).
- PCC: the paper's personalised chunk-copy (appendix A.1), every agent's computational module
  tried in the querier's context.

Same candidates or a superset of TDR's, same information as the paper's designs.

## G-1 Gate P1 -> P2 (ratified 2026-10-13)

Gate (1) holds: how an assistant looks while in use does not predict what it leaves behind,
and the ranking flips. Gate (2) for the single-flip rules: nothing to fix. Gate (2) for the
copy rules: see P1b. P2 as written in D-008 (a selector over SBP's flips) aims at the wrong
place and should be retargeted at the copy-based assistant.

P1b outcome (1000 seeds): TDR-25% and TDR-inverse add 1-2 points after the shift at the same
help during use, and cut TDR's "worse after shift" settings from 3 to 2. They do not remove the
shift penalty at rho = 1. PCC is best on both axes everywhere, but at rho = 1 it keeps the least
of what it gives (use 0.5, T = 30: +43.6% while assisted, +5.3% after the shift, a 12.6-point
shift penalty versus 3.8 for SBP). Every design that copies a whole computational module
roughly halves diversity at rho = 1. Across all P1b settings the shift penalty correlates with
diversity lost at the shift, r = 0.63. The split that matters is copying a whole solution
versus a one-step nudge, not how the copy source is sampled.

## D-011 P2 retargeted: full answer, nudge, or step back (ratified 2026-10-13, replaces D-008)

Each time an agent asks, the assistant picks one of three actions:

- full answer: PCC, the peer computational module that is best in this agent's context;
- nudge: SBP, the best single flip;
- step back: no suggestion, the agent explores on its own this round.

Policy: softmax over a linear score of six features (round, population diversity, the agent's
fitness rank, gain from the full answer, gain from the nudge, how many agents already hold the
full-answer module). About 15 parameters, fit with an evolution strategy. Rewards R_now, R_long,
R_adapt as in D-008. Baselines the learner has to beat: always PCC, always SBP, fixed mixes
(PCC with probability 0.25 / 0.5 / 0.75, else SBP), and a schedule (PCC for the first k rounds,
then SBP). Train on seeds 0-799, test on 800-999; also test on a 50% shift and an unseen use
rate. rho in {0.8, 0.9, 1.0}. CPU on ada.

Why: P1b shows the trade-off sits between full answers and nudges. The mixes and the schedule
are the simple rules; if one of them already matches the learned policy, that rule is the
result. The action set also maps onto LLM assistance (write it for me, suggest an edit, let
me try), which is what a P3 GRPO run would train.

## Rulings, 2026-10-13

Harsh: "run all of these, they sound good. human study probably not possible, we just focus on
LLMs." Ratified: D-001, D-003 (plus the robustness sweep), D-004, D-006, D-010, D-011, G-1.
D-009: the human study is off; P3 is the LLM stage.

## D-012 P3: an LLM assistant that keeps learning inside the community (proposed)

Harsh, 2026-10-13: "how do you make the LLMs break curse of homogenization over the course of
continual learning". Human study is off; the next stage is LLMs.

Setup (`src/llm/online.py`). The community is the paper's model at rho = 1 (where P1b found the
largest shift penalty). When an agent asks, Qwen3-4B-Instruct-2507 (LoRA r = 16) sees the agent's
20 choices as digits and writes new values for choices 1-10. There is no oracle: what the model
knows about the landscape comes only from its own updates. After each round, one GRPO step: G = 8
suggestions per asker, each scored by the simulator, the first one shown to the agent. Problems
come in sequence (4 problems x 60 rounds); each new problem redraws 20% of the last one's payoff
tables. At the end of each problem we copy the community and run 30 rounds without the assistant
on the next problem (the P1 probe). The same driver runs none, SBP and PCC on CPU with the same
draws, so oracle and LLM runs on a seed are paired.

The curse, as we would measure it: the model's suggestions collapse to one answer for everyone
(share of askers getting the most common suggestion, entropy across askers), the community's
diversity follows, the model's own samples collapse (distinct suggestions within a group of 8,
so GRPO has no signal left), and after each shift both recover more slowly than the problem
before.

Levers, cheapest first, each a separate arm on the same seeds:

1. Reward sees only this answer (R_now): the baseline, expected to collapse.
2. Standard RL fix: KL to the base model. Keeps the model's samples varied; the question is
   whether that keeps the community varied.
3. Nudge prior: a suggestion's gain is discounted by how much of the module it rewrites.
   P1b's finding as a per-query reward. Used only if P2 shows R_adapt prefers nudges.
4. Reward sees the community later (R_adapt per query via counterfactual branches), if 3 is not
   enough.
5. Replay of earlier askers, and several assistants instead of one, later.

Gate 0 (smoke): the model emits valid ten-digit answers and a GRPO step runs. Gate 1 (R_now, 2
seeds): reward rises within the first problem and the collapse measures move. If the model
cannot learn the landscape at all in 60 rounds, we give it more rounds per problem before
adding arms.

## D-003 robustness outcome (2026-10-14)

P1b rerun on the same 1000 seeds with a denser network (p_edge 0.1), a bigger shift (50% of
tables) and a smoother landscape (K = 5). The pattern holds in all three. At rho = 1, use 0.5,
T = 30, every whole-module copier loses 2 to 6 times more to the shift than SBP. With a 50%
shift, PCC goes from +43.6% while assisted to -1.3% after and SBP is the only assistant still
ahead of no AI (+1.8%). Bonferroni ranking flips: 6 (main), 8, 14, 16. Shift penalty vs
diversity lost: r = 0.63, 0.52, 0.48, 0.49. The open network density does not change the result.

## P3 oracle baseline (2026-10-14, corrected)

Oracle assistants through the P3 driver, 300 seeds, rho = 1, use 0.5, four problems of 60
rounds, each redrawing 20% of the last one's tables. Every community converges within the first
problem (PCC by round 10). Fitness while assisted, % vs no AI with 95% CI:

| problem | PCC | SBP |
|---|---|---|
| 1 | +27.8 [+24.9, +30.7] | +13.2 [+10.9, +15.5] |
| 2 | +4.0 [-1.0, +8.9] | +5.6 [+1.2, +9.9] |
| 3 | -3.6 [-8.4, +1.3] | +3.4 [-1.2, +7.9] |
| 4 | +2.7 [-2.7, +8.1] | +4.1 [-1.1, +9.3] |

Paired SBP - PCC: -14.6 points in problem 1 (p < 1e-18), then +1.6 (p = .52), +6.9 (p = .01),
+1.4 (p = .62). Once the community has converged, both assistants lose most of what they offered;
PCC loses all of its large early lead, because copying a peer's module offers nothing when every
peer holds the same module. My first reading ("SBP keeps helping") was too strong. The
next-problem AI-off probe has CIs of about +-5% and shows no clear differences.

For the LLM this frames the question: once the community is uniform, the only help left is help
that brings in something new. Whether an LLM trained online keeps generating new options or
collapses onto one answer is what the R_now, KL and step-back arms measure.

## D-013 P2 outcome and what it changes for P3 (2026-10-14)

P2, ES seed 0, tested on seeds 800-999 (held-out shift 50% and use 0.75 included). The three
rewards learn the same policy: help a lot in the first ~5 rounds (nudges first, then full
answers), then step back about 85% of the time. At rho = 1, use 0.5, shift 20%: learned R_now
+86.5% while assisted / +26.6% after the shift with the AI gone; R_adapt +85.7 / +26.9;
R_long +86.0 / +26.9. The fixed rules that always answer top out at +68 / +24 (25% full, 75%
nudges). Gated rules that use the same gain information (help only if it improves the asker,
otherwise step back) nearly match the learned policies: whichever helps more +84.8 / +24.5;
25% full first +82.5 / +26.2. Gating with full answers only: +55 / +9.

Reading: in this model the reward horizon does not matter. What matters is (1) being able to
step back when help would not improve the asker, so people explore instead of spending a round
on advice they reject, and (2) mixing full answers with nudges. A myopic reward already learns
both once stepping back is an option. Learning adds a few points over a one-line rule.

Change to P3 (proposed): the nudge-prior arm in D-012 assumed R_adapt would prefer nudges. It
does not. The arm that P2 supports is abstention: if the LLM returns the asker's choices
unchanged, the asker explores on their own that round, and that answer is scored by the
exploration's gain. Arms become R_now, R_now + KL (both running), and R_now + step back. The
counterfactual R_adapt arm stays on hold unless the LLM arms show something P2 could not.

## P3 gate 1, first look (2026-10-14, R_now seed 0, problem 1 of 4)

The model learns at first: in rounds 0-5 its answers improve the asker 31-43% of the time, about
3.4 of 8 samples differ, and 46-77% of askers get distinct answers. Mean fitness over the first
problem is 0.44 (no AI 0.34, oracle nudge 0.40, oracle full answer 0.61, same seed). Then it
collapses. By round 10 only 2% of its answers help, while the oracle full answer could still
give gains of 1.1-1.5 (x10 fitness units). By round 30, 95% of askers get the identical answer
(1101001010) and its 8 samples are nearly the same (1.3 distinct), so GRPO has no signal. After
the shift it keeps giving that answer to everyone (share 1.0) and does not recover. This is
the curse as D-012 defined it, in the first run. Waiting for the KL and step-back arms and seed 1.

## G-2 More seeds and rho = 0.8 (Harsh, 2026-10-14: "run more seeds once these finish, and add rho 0.8")

Each arm's follow-ups go in once both of its first two seeds have finished and been checked.
rho = 1: seeds 2-5 for R_now, KL and step back (results/p3). rho = 0.8: seeds 0-5 for all
three (results/p3_r08), where the best module depends on the asker's other choices. Oracle
baselines at rho = 0.8 (none, SBP, PCC, gated; 300 seeds) run on ada CPU (results/p3_oracle_r08).
R_now and step back submitted 2026-10-14 03:4x UTC; KL follows when kl_s0 and kl_s1 finish.

First two seeds, all three arms (before the extra seeds): every arm collapses onto one answer
for most askers by the second half of problem 1 (share 0.82-0.98), and the 8 samples per asker
stop differing (1.0-1.8 distinct). R_now seed 1 collapsed onto echoing the asker's own choices.
Step back learned to abstain and then abstained almost always (seed 1: ~100% from late problem
1), which equals no assistant. KL keeps samples varied a little longer but still collapses.
Caveat for reading recovery after shifts: after a shift the oracles often find nothing to gain
either (the converged community is often still at a local optimum in choices 1-10), so "does
not recover" is mostly "nothing to recover". The LLM also never sees other agents, so the fair
oracle comparison is SBP; early in problem 1 the LLM reaches ~0.6 of SBP's gain.

KL arm, first two seeds complete (2026-10-14): KL to the base model brings sample variety back.
kl_s1 never collapses (late in problems 1-4: share of askers on the most common answer 0.25,
0.15, 0.34, 0.70; distinct answers in a group of 8: 5.3, 6.4, 4.3, 2.6). kl_s0 collapses in
problems 1-2 (share 0.82-0.98, 1.2 distinct) and recovers in problem 3 (share 0.21, 5.5
distinct). Reading: when rewards go to zero, the KL term pulls the model back toward the base
model's spread, which keeps GRPO's signal alive. Its answers still rarely improve anyone late in
a problem, because the community has converged. All 30 follow-ups (3 arms x {rho 1 seeds 2-5,
rho 0.8 seeds 0-5}) submitted; KL ones at 04:0x UTC.

## G-2 outcome: P3 with 6 seeds at rho = 1 and rho = 0.8 (2026-10-14)

36 runs, all complete (3 arms x 6 seeds x 2 modularities). Numbers in
results/summary/p3_stats.txt and p3_plasticity.txt (scripts/analysis/).

1. The model collapses. Late in each problem, share of askers given the most common answer /
   distinct answers in a group of 8 (mean over seeds, problems 1-4):
   rho = 1: R_now 0.98, 0.98, 0.89, 0.94 / 1.1-1.3; step back 0.84, 0.99, 1.00, 0.86 / 1.0-1.6;
   KL 0.77, 0.77, 0.55, 0.74 / 2.2-3.6.
   rho = 0.8: R_now 0.83-0.89 / 1.3-1.6; step back 0.91-0.98 / 1.0-1.3; KL 0.49-0.67 / 2.7-4.1.
2. KL to the base model is the only lever tested that keeps the model's answers varied:
   KL - R_now in distinct answers of 8, averaged over problems: +1.47 [-0.20, +3.13], p = .07
   (rho = 1); +1.88 [+0.79, +2.96], p = .007 (rho = 0.8). Step back does not help (it collapses
   onto always stepping back or onto one answer).
3. Community effects are not resolved. Over the first problem the LLM arms are within a few
   percent of no AI (rho = 1: -3 to +1%; rho = 0.8: +3 to +13%), CIs +-10 to 20 points; the oracle
   nudge is +19 / +28% and the gated rule +61 / +56%. Later problems and the next-problem probe
   have CIs of +-25 to 80 points with 6 seeds.
4. The LLM is a weak assistant here. Early in problem 1 its shown answers capture ~40% (rho = 1)
   and ~10% (rho = 0.8) of the gain the oracle nudge would give; after shifts 3-33%, with no
   reliable difference between arms. At rho = 0.8 it does not learn to condition on the asker.

Reading: the model-side curse is real and robust (one answer for everyone, no variety left for
GRPO); a KL anchor to the base model keeps variety; but in this setup the LLM never helps much,
so its collapse does not measurably move the community, and 6 seeds cannot resolve community
effects anyway.

## G-3 What next for P3 (proposed)

(a) Evaluate, do not just train. Save the adapter at the end of every problem (now only the last
    one is kept) and run each checkpoint frozen as the assistant for many fresh communities on
    the same landscape (200 per checkpoint, inference only). This turns 6 noisy trajectories
    into tight community-level estimates: does a collapsed model help or hurt the people it
    serves, versus a KL model, versus no AI.
(b) Make the LLM a realistic, stronger assistant: show it a few of the community's best current
    choices in the prompt (what an LLM trained on community data knows). This is the copy-based
    regime where the simulations found the largest homogenisation cost, and the closest analogue
    to an LLM learning from its users' outputs.
(c) Dose-response on the one lever that worked: KL 0.05 vs 0.2, and an entropy bonus.
Recommendation: (a) first with the existing design (cheap, answers the open question), then (b).

## G-3 rulings and D-014 (b): user feedback as the training signal (2026-10-14)

Harsh: "do a then b", then "start b once a finishes. we can connect it to RLHF and preference
learning paradigm. e.g., model developers ask for feedback whether a session is good or bad, or
show two outputs and ask to select".

(a) runs as 12 evaluation jobs (src/llm/evaluate.py): each seed's final R_now, KL and step-back
adapters, the base model, no AI and the oracles serve the same 200 fresh communities on the
problem the adapters last trained on, then the P1 probes. Training now also keeps an adapter
per problem (adapter_p<k>) for (b).

(b) design (proposed details): the assistant sees the 3 best distinct approaches currently used
in the community, with their scores (`--context 3`), and learns only from user feedback of the
kind deployed assistants collect:
- thumbs (`--reward thumbs`): the asker tries the answer and gives a thumbs up if it improved
  their score. All helpful answers count the same.
- pick one of two (`--reward pairwise`): the asker sees two answers and picks the one better for
  them; when both are equally (un)helpful they pick at random. 8 samples = 4 such pairs.
- scalar gain (`--reward now`), with context, as the bridge to (a).
Why this connects: user feedback is individual and immediate, which is R_now. No single user's
rating sees what the community can do later. Hypothesis: thumbs (satisficing) collapses less
than pairwise (always pushes to the single best answer). Runs: 3 arms x 6 seeds x rho {1, 0.8},
no KL first; then KL on whichever feedback collapses most; then the (a) evaluator on the
per-problem adapters.

D-014 addendum (2026-10-14). Harsh asked whether to add KL arms for thumbs and pairwise. Recommended
yes and added: KL-anchored training is the realistic RLHF condition (no-KL is the ablation), and KL
was the only lever that kept the model varied in (a). thumbskl and pairwisekl, beta 0.05, same 6 seeds
x rho {1, 0.8}: 24 more runs, ~1 GPU-hour each. Not added for the scalar arm, which (a) already
covers with KL (without context). Total (b): 5 arms x 6 seeds x 2 rho = 60 runs.

## G-3 (a) outcome: frozen checkpoints on fresh communities (2026-10-14)

12 jobs, 200 fresh communities per training seed, 6 seeds x rho {1, 0.8}; % vs no AI on the same
communities, mean over seeds with t(5) CIs (results/summary/p3eval_a.csv, figures/p3eval_a).

rho = 1, while assisted / next problem with the AI removed:
untrained LLM -4.0 [-5.7, -2.4] / -1.4; R_now (collapsed) -2.1 [-15.2, +11.1] / -4.9;
KL +6.9 [-17.4, +31.1] / +1.9; step back -7.1 [-17.1, +2.9] / -6.0 [-10.8, -1.1];
oracles: SBP +19.3 / +3.4, PCC +43.3 / +5.8, gated +78.2 / +26.0.
rho = 0.8: untrained -4.3 [-7.2, -1.4]; R_now -5.7; KL +1.4; step back +5.8; oracles +32 / +49 / +70.

The collapsed R_now model gave 14 distinct answers to ~54k askers per seed (rho = 1); KL 370;
untrained ~700. Diversity after 30 rounds: none 4.70, R_now 4.13, KL 3.66, step back 4.57, PCC 2.05.
KL - R_now: +8.9 (rho 1) and +7.2 (rho 0.8) points while assisted, n.s.

Reading: the collapsed model neither helps nor measurably hurts the people it serves, and it
homogenises them only a little, because its answers rarely improve anyone and so are rarely
adopted. Homogenisation needs answers that get adopted. The untrained model is slightly harmful:
its answers cost people rounds. This is why (b) matters: with the community's best approaches in
the prompt, answers can help and get adopted. (b) launched 2026-10-14 19:2x UTC, 60 runs.

(b) first run check (2026-10-14 19:38 UTC, p3b/now_s0, problem 1): runs on GPU with context, and the
per-problem adapter is saved. With the community's best approaches in the prompt the model's answers
improve 56-68% of askers in rounds 1-3 (without context ~30%); by round 3, 92% of askers get the same
answer and by round 5, 100%; distinct community modules 90 -> 22 (round 3) -> 6 (round 5) -> 1. It
learns "copy the best for everyone" within a few rounds: NP behaviour, adopted. 59 runs queued.

## (b) training results (2026-10-15)

60 runs, all complete; numbers in results/summary/p3b_stats.txt (scripts/analysis/p3b_stats.py),
figures p3b_llm_rho1 / p3b_llm_rho08. With the community's 3 best approaches in the prompt:

1. Every feedback type collapses, faster than in (a). Late in each problem, share of askers on the
   top answer 0.70-1.00 and 1.0-2.2 distinct answers of 8 (rho = 1). Community modules at round 10:
   4.8-7.7 for the LLM arms vs 47 with no AI, 32 with SBP, 1.2 with PCC (rho = 1); ~30 vs 50 / 43 /
   27 at rho = 0.8. The model's answers are adopted and the community follows them.
2. The feedback form matters for how far the model collapses. Pick one of two collapses completely
   (top share 0.97-1.00, 1.0 distinct of 8, excluding degenerate seeds). Thumbs keeps the most
   variety: thumbs - pairwise +0.66 distinct of 8 [+0.02, +1.29], p = .046 at rho = 1 (n = 5,
   degenerate seed excluded); +0.42 n.s. at rho = 0.8. Pairwise - scalar: -0.24 (p = .08) and
   -0.64 (p = .05).
3. KL rescues pick-one-of-two at rho = 0.8 (+1.09 distinct, p = .025), n.s. at rho = 1; it does
   nothing for thumbs. Without KL, one pairwise seed per rho degenerated into free-text reasoning
   ("We are given 20 yes/no choices ...") with 0% valid answers by problem 4. Caveat: my simulated
   user picks at random between a malformed and a useless valid answer, so malformed output is not
   penalised; a real user would prefer the well-formed one. With KL this never happened.
4. Community fitness: as noisy as (a) with 6 seeds (problem 1 at rho = 1: LLM arms +8 to +18%,
   n.s.; PCC +35%, gated +61%). The frozen-checkpoint evaluation (24 jobs, checkpoints after
   problems 1 and 4, 100 fresh communities each) is running.

## (b) frozen evaluation (2026-10-15)

24 jobs: checkpoints after problem 1 and after problem 4, every (b) arm plus the untrained model
(both with the community context) and the oracles on 100 fresh communities per seed, 6 seeds x rho
{1, 0.8}. Per arm: results/summary/p3eval_b_p0.csv, p3eval_b_p3.csv; trained arms pooled:
p3b_eval_pooled.txt (scripts/analysis/p3b_eval_pooled.py). % vs no AI, t(5) over seeds.

rho = 1, after problem 1 (pooled over the 5 trained arms):
- while assisted +18.9 [+2.2, +35.6] (p = .03); next problem with the AI removed -7.0 (p = .11)
- trained LLM - oracle nudge (SBP): +1.5 while assisted (n.s.), -14.9 [-25.3, -4.5] (p = .014)
  after the shift. Same help now, worse later: the P1 dissociation, now for learned LLM assistants.
- diversity after 30 rounds: no AI 4.67, SBP 4.44, trained LLM 2.52, untrained LLM 2.96, PCC 2.04
- the untrained model with context already does most of this: +13.2 now, -4.3 after. Showing the
  best approaches is enough to make an LLM a copy-the-best assistant; training on user feedback
  adds a little help now and none of the later cost goes away.
rho = 0.8, after problem 1: +1.6 while assisted (n.s.), -5.3 [-8.2, -2.4] after (p = .005); the
  untrained model -6.5 after (p < .001). Copying the best approaches without fitting the asker hurts
  at lower modularity, as NP does in the paper.
After problem 4: rho = 1 +19.5 now, +1.8 after (n.s.), LLM - SBP after -2.9 (n.s.); rho = 0.8
  -1.6 / -2.7 (n.s.). The later-cost gap is not there at the last checkpoint.
Feedback form and KL: no reliable differences in community outcomes between arms. They differ in
how many distinct answers the model gives (eval-time distinct answers at rho = 1, after problem 4:
pairwise 92, R_now 329, thumbs 340, thumbs + KL 349).

## G-4 and G-5 (Harsh, 2026-10-15: "do G-4 and G-5")

G-5: `--prefer-valid` for pick one of two. On a tie in gain, a well-formed answer beats a malformed
one (100% vs ~45% before); other ties stay coin flips; a real gain still wins. New arm name
pairwisefix. Not rerun with KL: the KL version never degenerated (valid ~100%), so the fix barely
changes it.
G-4: (i) frozen evaluation of the existing (b) checkpoints after problems 2 and 3 (adapter_p1,
adapter_p2), same design as p0/p3: 24 jobs. (ii) more seeds at rho = 1: thumbs seeds 6-11 and
pairwisefix seeds 0-11; pairwisefix also at rho = 0.8 seeds 0-5. 24 training runs, all with per-
problem adapters. Once (i) and the training are done, evaluate the new runs at all four checkpoints
into the same result files (the evaluator skips assistants already present, communities are seeded,
so the new arms are paired with the old ones).

### G-4 (i) and G-5 results (2026-10-15)

Gap trajectory, existing (b) arms pooled, 6 seeds, trained LLM - oracle nudge (SBP), % of no-AI:
| checkpoint after problem | rho 1 while assisted | rho 1 next problem, AI off | rho 0.8 next problem, AI off |
|---|---|---|---|
| 1 | +1.5 | -14.9 [-25.3, -4.5] p = .014 | -12.6 p < .001 |
| 2 | -1.2 | -9.3 [-15.8, -2.7] p = .015 | -10.1 p = .022 |
| 3 | +0.5 | -5.4 n.s. | -11.8 p = .001 |
| 4 | +0.1 | -2.9 n.s. | -10.1 p < .001 |
At rho = 1 the later cost shrinks steadily over continual training while help in the moment stays
equal to the nudge; at rho = 0.8 it stays at 10-13 points (and the LLM is ~30 points below the
nudge while assisted there).

G-5: with the fixed chooser no pick-one-of-two run degenerates (min valid >= 0.98 over 18 runs,
vs 0.00 for one seed per rho before). It still collapses fully: late share on the top answer
0.98-1.00, 1.0 distinct of 8. Thumbs keeps more variety than fixed pick-one-of-two with 12 seeds
at rho = 1: +0.73 distinct of 8 [+0.44, +1.01], p < .001; rho = 0.8 +0.25 n.s. (6 seeds).
So the degeneration was the chooser artifact; the collapse under pick-one-of-two is real.

### G-4 (ii) result: 12 seeds at rho = 1 (2026-10-16)

Frozen evaluation of every (b) checkpoint, 100 fresh communities per seed. Difference from the
oracle nudge on the next problem with the AI removed, points of no-AI fitness, t CIs over seeds
(results/summary/p3b_eval_arms.txt; figure p3b_gap_over_training):
| checkpoint after | thumbs (12) | pick one of two, fixed chooser (12) | untrained with context (12) |
|---|---|---|---|
| problem 1 | -11.3 p = .008 | -11.5 p < .001 | -9.7 p = .001 |
| problem 2 | -8.3 p = .053 | -9.9 p = .001 | -10.5 p = .001 |
| problem 3 | -2.1 n.s. | -7.9 p = .043 | -0.7 n.s. |
| problem 4 | -5.3 p = .031 | -7.6 p = .004 | -6.1 p = .006 |
While assisted, every arm is level with the nudge (within +-6, mostly n.s.; thumbs +6.3 at
problem 4, p = .02). rho = 0.8 (6 seeds): 24-37 points below the nudge while assisted and 6-14
below it on the next problem at every checkpoint.
Thumbs - pick one of two, paired over 12 seeds at rho = 1: while assisted +6.9 pooled (p = .07;
+12.4 at problem 4, p = .03); next problem +2.5 pooled (n.s.; +5.8 at problem 3, p = .04).

Reading: with 12 seeds the result holds at every checkpoint for pick one of two and at three of
four for thumbs. An LLM assistant that sees what works for others gives the same immediate help as
the oracle nudge and leaves people 5-12 points worse placed for the next problem. The untrained
model with the same prompt shows most of the cost, so most of it comes from what the assistant is
shown, and feedback training does not remove it. Thumbs is a little better than pick one of two
(more varied answers, somewhat more help later), not significantly when pooled.

## G-6: write-up, and a second model family (Harsh, 2026-10-24)

Harsh: "go ahead do the writing but instead of reproducing in a bigger model of same family, we
should try different family. I think gemma3-4b is also in cache, we can reproduce. the writeup should
be easy to understand following the writing style of the original reference paper".

Gemma reproduction: google/gemma-3-4b-it (cached), same design as the 12-seed Qwen result: rho = 1,
context 3, thumbs and pick one of two with the fixed chooser, 12 seeds each (results/p3g), then the
frozen evaluation of all four checkpoints on 100 fresh communities per seed with untrained Gemma and
the oracles. Code changes, no-ops for Qwen: Gemma 3 4B loads as a vision-language model, so LoRA
targets only the language model's projections (regex), the duplicated BOS from Gemma's chat template is
removed before tokenizing, and vLLM runs with images disabled. GPU smoke (job 420471): training and
the frozen evaluation both run; Gemma's first answers improve 47-79% of askers and by round 3, 81% of
askers get the same answer.
Write-up: LaTeX in paper/ (Overleaf-ready; no TeX compiler on the Mac or ada), with a readable PDF
preview rendered from the same source.

## D-015 Culture over time: the minimum addition, and rewards for per-epoch learning (2026-10-24)

Harsh: "keep what we have but also explore culture over time. I want to test different reward models
for RL. e.g., incentivizing p-creativity, h-creativity, what else? question is what is the minimum
addition to the original paper ... to study what happens if LLMs are assisting over time, and can we
learn something at each epoch to break the curse of homogenization over time. this can have
implications for continual learning, and for ... AI for science or companies deploying their own LLMs".
Also asked "whats the RQ ... why assisted vs unassisted"; answered that the current framing (help now vs
adapt later) came from the pruning step and treats homogenization as mechanism, not outcome, and that the
original model cannot show homogenization over time because every community converges within ~60 rounds.

Minimum addition (proposed): two changes, everything else as in Fazelpour et al.
1. Turnover: each round each team leaves with probability mu and is replaced by a newcomer with a random
   approach and specialization. Gives diversity a steady state for an assistant to push down.
2. Epochs with a learning assistant: the problem changes every epoch (20% of payoffs) and the assistant
   is retrained once per epoch on that epoch's requests, then redeployed frozen. The loop of a deployed
   model retrained on its users' feedback.
Rewards for the per-epoch update (all zero unless the answer helps the asker):
- thumbs (helpfulness; with and without KL), the current practice;
- P-creativity: bonus for distance from the asker's own approach (new to the person). Copying a peer is
  P-creative; prediction: still homogenises (individual novelty up, collective diversity down);
- H-creativity: bonus for distance from the nearest approach anyone in the culture holds (new to the
  culture). Copying earns no bonus; prediction: breaks the ratchet;
- later: diversity gain after uptake, rarity, delayed reward (after the next change), community welfare,
  SFT on adopted answers (the model-collapse loop).
Pilots: P4a (CPU, oracles, turnover 0-0.02, rho 1 and 0.8, 200 seeds, 20 epochs; job 420513): is there a
steady state, does copy-style help lower it? P4b (GPU, Qwen with context, frozen / thumbs / thumbs+KL /
pcreative / hcreative, 20 epochs, 6 seeds, ~25 GPU-hours; src/llm/culture_llm.py): does the culture's
diversity ratchet down across epochs, and which reward stops it? Novelty is measured against the living
culture, not all of history: with 1,024 possible modules a history archive fills within a few epochs.

Gemma replication on hold (Harsh, 2026-10-24: "the gemma reproducibility runs we can hold off for now,
lets focus on this"). The 8 runs already running (thumbs and pairwisefix, seeds 0-3) finish; the 16
pending ones are held with `scontrol hold` (release with `scontrol release <ids>`). Focus is D-015.

### P4a result and P4b launch (2026-10-24)

P4a (200 seeds, 20 epochs of 60 rounds, results/summary/p4a_culture.csv, figures/p4a_culture): turnover
gives a steady state. No-AI diversity (mean pairwise Hamming): 0.12 (mu 0), 1.61 (0.005), 2.56 (0.01),
3.85 (0.02). At rho = 1, copy-style help lowers it by ~37% (PCC 1.62 and gated 1.65 vs 2.56 at mu
0.01) and nudges do not (SBP 2.72); at rho = 0.8, PCC does not lower it. No ratchet with oracles:
diversity's trend across epochs relative to no AI is flat for every assistant (all p > 0.1). While the
oracle is present the lower diversity costs nothing: assisted cultures do 10-55% better, also in the
first 10 rounds after each change. So an assistant that always knows the truth can homogenise a culture
for free; the question for P4b is what happens when the assistant's knowledge comes from the culture.
P4b launched: rho = 1, mu = 0.01, context 3, 20 epochs of 60 rounds, 6 seeds x {frozen, thumbs,
thumbs + KL 0.05, pcreative, hcreative} (lam 1), 256 requests x 8 answers per epoch update. 30 runs,
results/p4b. Smoke (job 420516) passed.

P4b first look (seed 0, epochs 1-5, 2026-10-24): with the community's best approaches in the prompt,
even the never-updated model gives the same answer to 81-99% of askers within an epoch (it copies the
top-scoring approach), and its 8 samples per request for the update are nearly identical (1.0-1.2
distinct). GRPO learns only from differences within a group, so every reward then gives zero advantage
and nothing is learned: the reward variants cannot differ if the model never proposes anything else.
The running batch stays as the "standard practice" condition (update samples at temperature 1). Added
`--update-temperature` (deployment unchanged) and a diagnostic (src/llm/temp_probe.py, job 420561) of
answer variety, validity, helpfulness and new-to-culture share at temperatures 1.0-2.0, to choose an
exploration setting for a second batch (proposed: thumbs, P-creativity, H-creativity at that temperature).

### P4b standard practice, first result (2026-10-24, 4 of 6 seeds finished)

Temperature check (job 420561; 24 askers x 3 seeds, no-AI cultures after 30 rounds, 8 answers each):
distinct answers per request 1.97 / 2.25 / 2.69 / 3.18 at temperature 1.0 / 1.3 / 1.6 / 2.0; all valid
at every temperature; helpful 0.17 -> 0.15; new to the culture 0.25 -> 0.30. In the LLM's own (already
narrower) cultures the update groups had 1.0-1.5 distinct answers at temperature 1.

Standard practice (update at temperature 1), seeds 0-3, epochs 3-20 (results/summary/p4b_culture_llm.csv,
figures/p4b_culture_llm). Culture diversity, mean pairwise Hamming: no AI 2.64, LLM never updated 1.93,
thumbs 1.59, thumbs + KL 1.62, P-creativity 1.75, H-creativity 1.57 (PCC oracle in P4a: 1.62). Paired
with the never-updated model: thumbs -0.34 (p = 0.009), thumbs + KL -0.30 (p = 0.02), H-creativity
-0.36 (p = 0.008), P-creativity -0.18 (p = 0.25). Retraining does learn something: the share of answers
that help rises from 0.028 to 0.040-0.046. No ratchet: the gap to the never-updated model is flat across
epochs for every reward (slopes within +-0.01 per epoch, all p > 0.3). Fitness while assisted and on the
next problem with the AI removed: no difference from no AI or from the never-updated model (wide CIs).
Reading: the deployed model homogenises the culture within the first epoch (it hands out the top
approach), and retraining on thumbs makes it homogenise as much as an oracle that hands out full answers.
The novelty bonuses change nothing because the model almost never proposes two different helpful answers
to the same request, so there is nothing for the bonus to choose between.

Exploration batch launched (jobs 420636-420653): thumbs, P-creativity and H-creativity with update
samples at temperature 2.0 (deployment stays at 1.0), seeds 0-5, results/p4b/{thumbs,pcreative,
hcreative}T2_s*. The question: once the model proposes alternatives, does a reward for novelty to the
culture keep the culture more diverse than thumbs does, at what cost in help? Note: samples come from
the temperature-2 distribution while the gradient is taken on the temperature-1 policy (no importance
correction); this is a common exploration shortcut and is noted as such.

### P4b full result, why novelty rewards fail, and what sets diversity (2026-10-24)

All 42 runs done (6 seeds). Culture diversity, epochs 3-20 (no AI 2.69 on these seeds; never-updated LLM
1.91). Paired with the never-updated model: thumbs -0.30 (p = 0.002), thumbs + KL -0.28 (p = 0.002),
P-creativity -0.15 (p = 0.14), H-creativity -0.29 (p = 0.004); with exploration (update samples at
temperature 2): thumbs -0.26 (p = 0.008), P-creativity -0.29 (p = 0.002), H-creativity -0.23 (p = 0.05).
Against thumbs at the same temperature no reward differs (all p > 0.1; P-creativity at temperature 1 is
+0.14, p = 0.10, and goes away with exploration). Exploration did raise variety in the update groups (1.2
-> 1.7-1.9 distinct answers of 8). No ratchet in any arm. Next problem with the AI removed: no arm
differs from the never-updated model.

Policy probe (job 420982, src/llm/policy_probe.py, results/p4b_probe): every epoch-20 model and the
untrained one answer the same 100 teams x 8 samples in the never-updated model's final culture.
Training works on help: helpful answers 1.7% untrained -> 21-22% for every reward. But among the trained
models' helpful answers, none is new to the culture (0.000 in all 42 models; untrained 17%), and they
rewrite ~4.7 of the asker's 10 bits. The trained model helps by handing out an approach someone already
holds. So the H-creativity bonus is never paid and H-creativity trains exactly like thumbs.

Ideal learners (P4a, 200 seeds, rho 1, mu 0.01; results/p4a and results/p4a_reach, figures/p4a_reach):
what if the model learned each reward perfectly, searching all 1,024 modules? Diversity: any helpful
module (ideal thumbs) 1.95, farthest from the asker (ideal P-creativity) 1.88, farthest from the culture
(ideal H-creativity) 1.94; PCC (best approach already in the culture, which is what the trained LLM does)
1.62; no AI 2.56. The preference among helpful answers changes the route, not the end: adoption needs an
improvement, so repeated strong help takes each team to its best module whatever the reward prefers.
What sets diversity is reach, how many bits the assistant may change: best module within 1 bit (SBP)
2.72, within 2: 2.74, 3: 2.53, 5: 1.90, 10: 1.58. Help while assisted (% vs no AI): +9.5, +35, +52, +66,
+69. Next problem with the AI removed (new P4a probe, same as P4b's): +7, +21, +30, +34, +34. So reach 3
keeps diversity at the no-AI level with three quarters of the maximum help and most of the carry-over to
the next problem. PCC is the worst trade: as homogenising as unlimited reach, a third of the help (+23%,
next problem +13%). In this setting homogenisation costs nothing on the next problem: every assistant
leaves the culture better off than no AI when it is removed, because the problem changes only 20%.

Reading: rewarding novelty cannot break homogenisation here, for two reasons: the model never finds
helpful novel answers, and even a model that did would end where any strong helper ends. The lever is
how much of a person's approach the assistant replaces. Launched (jobs 421178-421183): a `local` reward,
thumbs only if the answer changes at most 3 bits of the asker's approach ("improve my approach, don't
replace it"), update temperature 2, seeds 0-5, results/p4b/localT2_s*; compare with thumbs at
temperature 2. Open: whether diversity matters for anything in this setting (it does not for the next
problem at shift 0.2); candidates are a larger shift and rho 0.8, where P3 found the costs.

### Local reward result, and a correction to the probe numbers (2026-10-24)

Local reward (thumbs only for answers within 3 bits of the asker's approach, update temperature 2,
6 seeds): culture diversity 1.63 vs thumbs at temperature 2 1.65 (paired -0.03, p = 0.45); fitness while
assisted slightly lower (-0.03, p = 0.04). Its reward falls to ~0 in late epochs for most seeds: the
model almost never proposes a helpful answer within 3 bits, so there is little to learn from. Probe:
the local models still answer with the top approach (copy share 0.83 vs 0.41 for thumbs at temperature 2,
p = 0.08), and their helpful answers still rewrite ~4.6 bits. The reward did not teach small edits.

Correction to the probe summary above: the mean helpful share (1.7% untrained -> 21-22% trained) is
driven by seed 0, where the trained models' one answer helps 99% of teams. Per seed the median is 2.1%
untrained and 6-8% trained. Every trained model gives nearly the same answer to every team (1.0-1.3
distinct of 8 samples; the top approach or one other fixed approach). The finding that none of the
trained models' helpful answers is new to the culture stands.

Reading: per-epoch reward learning from sparse feedback collapses the model to one answer per culture,
whatever the reward. Within the reward family tested (thumbs, + KL, P-creativity, H-creativity, local;
with and without exploration) nothing changes the culture's diversity. The oracle sweep says reach is
the lever, and the LLM does not learn reach from reward. Next options (proposed, not run): (1) check that
homogenisation costs something with turnover (larger shift, rho 0.8; CPU); (2) give reach to the model
through the interface instead of the reward: ask for at most k bit changes to the asker's approach
(an edit, as in "suggest changes to my draft") rather than a full approach, and let thumbs choose among
edits; (3) more update steps per epoch.

## D-016 Design space of rewards and interactions (Harsh, 2026-10-24: "do 1 then 2. we should more systematically explore the design space of reward models and interactions. ground in literature ... show me the equations of all reward models we have")

Equations of every reward and oracle so far: docs/REWARDS.md.

### Option 1: does homogenisation cost anything with turnover? (P4a oracles, 200 seeds; results/p4a_cost, figures/p4a_cost)

Grid: shift 0.2 / 0.5 / 1.0 x rho 1 / 0.8 x turnover 0 / 0.01; assistants SBP, best within reach 2/3/5/10,
PCC, ideal thumbs, ideal H-creativity. New measure: the next problem with the AI removed, round 30.
- Shift 0.2 (P4b so far): the next problem is close to this one, so whatever raised fitness carries over;
  no cost of homogenisation (reach 10 +27.8% at round 30, reach 3 +27.1%).
- Shift 1.0: nothing carries over; every assistant within +-2% of no AI at round 30.
- Shift 0.5, rho 1, turnover 0.01: homogenisation costs. Round 30 of the next problem, AI removed, vs no
  AI: SBP +3.7%, reach 2 +7.0%, reach 3 +7.3%, reach 5 +3.9%, reach 10 +1.9%, PCC -1.0% (diversity
  3.04, 3.04, 2.80, 2.11, 1.77, 1.81; no AI 2.82). Paired: reach 3 - reach 10 +5.4% [4.3, 6.5],
  reach 3 - PCC +8.3% [6.9, 9.8]. Reach 10 helps more while assisted (+73% vs +55%) and less later. PCC
  (the oracle closest to the trained LLM) gives +20% while assisted and nothing after.
- Without turnover the cultures converge anyway (diversity 0.1-1.0) and differences are small.
So the cost of narrowing shows at intermediate change, which is where the rest of D-016 runs: rho 1,
turnover 0.01, shift 0.5 (shift 0.2 kept as the second setting).

### Option 2 launched: reach through the interface (jobs 421293-421316, results/p4b_s05)

`--edits k`: the model names at most k choices to flip instead of writing a full approach (reach k by
design). Smoke (job 421260): all answers valid, 4.8 distinct of 8 samples per request (full answers:
1.2), 58% helpful at the start of a run. Arms at shift 0.5, 6 seeds: frozen and thumbs (temperature 2)
with full answers; frozen and thumbs (temperature 2) with edits of at most 3.

### Design-space oracle grid (jobs 421284-421291, results/p4a_grid)

Ideal assistants by answer source (any module / only modules held in the culture, i.e. copying) x
preference among helpful answers within reach (largest gain / any / rarest = fitness sharing) x reach
1-10, plus 4 and 10 separate assistants (islands) for the copying source. Shift 0.2 and 0.5. Purpose:
screen designs cheaply before giving any to the LLM. Literature grounding for the axes: verified
bibliography from a literature pass (2026-10-24), to go in the design-space page.

### Cost grid at rho 0.8, and the first design-space screen (2026-10-24)

rho 0.8 (the best module depends on each team's other choices): with turnover, searching help keeps the
culture more diverse than no AI at every reach (shift 0.5: reach 1 3.16 ... reach 10 3.33, no AI 2.79) and
later fitness rises with reach (round 30 of the next problem, AI off: +3.8% at reach 1, +9.6% at reach 10).
Copying (PCC) still narrows (2.76) and leaves +3.5%. So: copying is the bad choice at both rho; reach
matters when everyone's best answer is the same (rho 1). Earlier wording "reach, not preference, sets
diversity" holds at rho 1 only.

First screen (rho 1, turnover 0.01, shift 0.5 and 0.2; 200 seeds; results/p4a_grid, figures/p4a_grid_*,
results/summary/p4a_grid_*.csv). Source x preference x reach, plus islands. At shift 0.5:
- Copying from the culture is the worst trade at every reach (+3 to +26% while assisted, at most +2.6% on
  the next problem); searching any module at reach 3: +55% / +7.3%.
- Later fitness peaks at reach 2-3 for every preference.
- Rarest helpful answer (fitness sharing) vs largest gain: +2.9 points on the next problem at reach 5 and 10
  (p < 1e-11), diversity +0.2, about 3 points less help while assisted; no difference at reach 2-3. Random
  helpful is worse than largest gain at small reach (-1.6, p = 0.01).
- Correction to the earlier reading: at full reach the ideal novelty learners (1.94) are more diverse than
  the greedy helper (1.58) and about as diverse as random-helpful (1.95); preference moves diversity by
  ~0.3, reach by ~1.2.
- 4 or 10 separate copying assistants: no benefit (islands are scattered over one random network, so teams
  copy across them; needs network-aligned groups).
At shift 0.2 the ordering of diversity is the same, but later fitness is carry-over and rarity does not help.

Design-space page (equations of all rewards, the five design choices with verified literature, results):
docs/design-space.html, published at https://claude.ai/artifact/NLTTiQwjtxaMKMXewr88vF.

Second screen launched (jobs 421351-421357): least help that works (capped at 2/3/5), local competition
(reach 3/5/10), tweak or leap (reach 2-3, leap share 10-25%), recombination toward a rare donor (at most
3/5/10 bits), use rate 0.25 and 0.75; and the first screen at rho 0.8. Adaptive reach (1/5 rule) not
screened: with an oracle that searches its whole range, "no success" means no improvement within reach,
so the ES rule (shrink the step on failure) points the wrong way; it fits an LLM that proposes, not an
exhaustive oracle.

### Second screen result (2026-10-24; rho 1, turnover 0.01, shift 0.5, 200 seeds)

Paired with the plain reach-3 greedy assistant (+55% while assisted, +7.3% on the next problem, AI off):
- Tweak or leap (reach 3, 10% leaps to a helpful module nobody holds): +6.2 [5.5, 6.9] help now, +0.6 n.s.
  later; at reach 2 vs plain reach 2: +14.7 now, +1.3 later (p = 0.02). Best design so far.
- Recombination toward a rare donor, up to 5 bits: +3.3 now, +0.3 n.s. later; up to 10 bits vs unbounded
  greedy: -11.9 now, +5.5 [4.6, 6.4] later.
- Local competition at reach 5 vs greedy reach 5: -0.6 now, +1.9 [1.3, 2.6] later.
- Least help that works (capped 3): -7.1 now, -0.9 n.s. later. Not useful.
- Use rate 0.25 vs 0.75: little change at reach 3; strong and copying assistants narrow more with more use.
rho 0.8 first screen: searching keeps diversity above no AI everywhere; preferring random helpful is weakest;
copying stays at or below no AI and gives at most +3.8% later.
Reading: mostly small changes plus a few targeted jumps (Miu et al. 2018's tweaks and leaps) dominate. For
the LLM this suggests edits of at most 3 choices plus occasional full answers rewarded only when helpful and
new to the culture. Page updated (version 2).

### LLM with edits, and chance baselines (2026-10-25; rho 1, turnover 0.01, shift 0.5)

LLM arms (jobs 421293-421316, results/p4b_s05, 6 seeds) paired with no AI on the same seeds, epochs 3-20.
Chance baselines added to culture.py (`rnd<k>`: flip a random 1..k choices of the asker's module; `rndfull`:
random module; blind to the landscape, always answer, kept only if they help; job 421782, 200 seeds,
results/p4a_cost/shift0.5). Analysis scripts/analysis/p4b_interface.py, figures/p4b_interface,
results/summary/p4b_interface.csv.

| arm | diversity vs no AI | help while assisted | next problem, AI off (mean 30 rounds) | answers that help | distinct per round |
|---|---|---|---|---|---|
| LLM full, never updated | -0.62 | -0.9% | -9.0% | 4.2% | 9% |
| LLM full, thumbs T2 | -0.92 | -1.1% | -8.9% (p = 0.02) | 6.0% | 5% |
| LLM edit <=3, never updated | -0.02 | -3.3% | -7.9% | 3.0% | 32% |
| LLM edit <=3, thumbs T2 | +0.05 | -2.0% | -2.7% | 2.6% | 38% |
| random edit, 1 choice (200 seeds) | +0.06 | -0.5% | 0.0% | 3.6% | |
| random edit, 1-3 choices | +0.14 | +15.2% | +4.9% | 4.1% | 87% |
| random full answer | +0.50 | +19.9% | +6.0% | 4.0% | 98% |
| PCC | -1.00 | +20.2% | +2.2% | | |
| reach 3 | -0.01 | +55.3% | +10.9% | | |
| tweak or leap 3/10% | -0.14 | +61.7% | +12.3% | | |

Help and distinct shares for the random arms from an instrumented rerun of seeds 0-5.

- Edits remove the narrowing: edit minus full, diversity +0.60 (never updated) and +0.98 (thumbs), both
  p = 0.001. Help while assisted n.s.; next problem +1.1 (n.s.) and +6.2 (p = 0.11).
- The LLM is worse than guessing at this shift. Same help rate as random proposals, but random gives
  +15 to +20% while assisted and the LLM about 0: LLM minus random of the same form, -16.2 (p = 0.02) and
  -16.5 (p = 0.004) for full answers, -15.9 (p = 0.01) and -14.7 (p = 0.03) for edits. Next problem: full
  answers -9.6 (p = 0.05) and -9.5 (p = 0.01) vs random; edits -5.3 and 0.0, n.s. Where the gap comes from
  (which teams get helped, gain sizes) is not logged.
- Thumbs does not raise the help rate over 20 epochs (3-6% throughout, every arm). At shift 0.2 the same
  LLM gave +8 to +15% while assisted (results/summary/p4b_culture_llm*.csv), mostly by passing on the top
  approaches; at 0.5 that stops working and it has nothing else.
- 6-seed LLM next-problem contrasts are noisy (seed spread about +-15%); with 200 seeds PCC is +2.2% on the
  next problem, the 6-seed subset said -4.4%.
- Reading: the interface fixes homogenisation, but the culture question is moot while the assistant helps
  less than a random mutation. The full-answer LLM narrows the culture without helping: almost everyone gets
  the same answer and the few it helps all move there.

Tweak or leap for the LLM launched as planned (jobs 421766-421777: tlE3 never updated, tlE3T2 thumbs at
update temperature 2; `--edits 3 --leap 0.1 --shift 0.5`, seeds 0-5; mira/vega/grace).
Proposed, not run: (a) answer at a higher temperature, to test whether variety alone closes the gap to
random; (b) show the asker's neighbours instead of the culture's top approaches; (c) several edits per
request, the team tries each. Design-space page version 3.

### Answering temperature (Harsh, 2026-10-25: "go with (a), run it on grace too")

Does variety alone make the LLM as useful as random proposals? Never-updated model, shift 0.5, seeds 0-5,
full answers and edits (<=3) at deployment temperature 1.5, 2.0 and 2.5 (1.0 = the interface runs); 36
jobs on mira/vega/grace, results/p4b_s05/frozen{,E3}D{15,20,25}_s*. Retrained arms wait for this
result. Sampling uses the full vocabulary (no top-p/top-k); the earlier probe (job 420561) had every
answer valid up to 2.0, so 2.5 is the top. An invalid answer leaves the team to its own search that
round, so validity is reported with each arm. Analysis scripts/analysis/p4b_temp.py (vs no AI and vs
random of the same form, paired by seed).

### Tweak or leap and answering temperature, results (2026-10-25; shift 0.5, 6 seeds, epochs 3-20)

Tweak or leap (jobs 421766-421777): edits <=3 plus a full answer on 10% of requests. Never updated:
diversity +0.08 vs no AI, help while assisted -0.5%, next problem -0.1%; thumbs T2: +0.18, -1.1%, -7.4%.
Vs random edits: -13.1 (p = 0.008) and -13.8 (p = 0.02) while assisted. Only ~3% of leaps help and are new
to the culture. At temperature 1 it is plain edits again.

Answering temperature (never updated; results/summary/p4b_temp.csv, figures/p4b_temp):

| T | edit: help / next / distinct | full: help / next / distinct |
|---|---|---|
| 1.0 | -3.2% / -7.9% / 0.32 | -0.9% / -9.0% / 0.09 |
| 1.5 | +2.9% / -4.2% / 0.39 | +2.5% / -6.8% / 0.10 |
| 2.0 | +6.3% / +0.7% / 0.46 | +2.3% / -6.5% / 0.13 |
| 2.5 | +8.2% / +1.3% / 0.55 | +6.5% / -3.8% / 0.15 |
| random | +15.2% / +4.9% / 0.87 | +19.9% / +6.0% / 0.98 |

Valid >= 0.98 everywhere. Edits, T 2.5 minus 1.0 paired: +11.5 while assisted (p = 0.03), +9.2 next problem
(p = 0.04); slopes +7.6 and +6.5 per unit T (p = 0.048, 0.046). Gap to random edits at 2.5: -4.4 and +3.9,
n.s. Diversity stays at the no-AI level. Full answers: distinct rises only to 0.15, diversity -0.5 vs no AI,
+7.5 while assisted at 2.5 (p = 0.11). The help share barely moves (edits 0.030 -> 0.037): more directions,
not more help per answer.
Reading: with edits and enough answer variety the LLM keeps the culture as diverse as no AI and helps about
as much as random edits. Still far from the oracle edit (+55%).
Proposed next (not run): retraining on the varied stream, answering at 2.5: thumbs with edits, tweak or leap
never updated and thumbs; 18 jobs. Design-space page version 4.

### Practical variety: menu and spread reward (Harsh, 2026-10-26: "temp 2.5 is not really practical";
### 2026-10-28: "a and b sounds good")

Temperature 2-2.5 is a diagnosis (the LLM falls short of random because it gives nearly the same answer to
everyone), not a design. Two practical ways to get the variety at deployment temperature 1, edits (<=3),
shift 0.5, seeds 0-5 (results/p4b_s05):
- (a) Menu: `--menu 3`, three answers per request at temperature 1; the team tries each and keeps the best
  that helps. Never updated (frozenE3M3, jobs 426342-426352 even). Fair reference: random menu `rnd3x3`
  (three random 1-3 flip edits, best kept; job 426341, 200 seeds).
- (b) Spread reward: R = h / (1 + m), m = other askers this epoch who adopted the same module from the
  assistant (delta = 0, counted over the epoch's served requests). Update samples at temperature 2,
  deployment 1 (spreadE3T2, jobs 426343-426353 odd); compare with thumbs (thumbsE3T2), same settings.
Fake-LLM CPU test of both paths passed; rnd3 reproduces the stored P4a baseline exactly.

### Gemma replication dropped (Harsh, 2026-10-28: "we also have some blocked jobs related to this project which we can drop")

Cancelled the 16 held P3 Gemma-3-4B jobs (420489-420504: thumbs and pairwise --prefer-valid, seeds 4-11,
results/p3g). Seeds 0-3 of both arms finished and stay in results/p3g; the seed 4-11 dirs hold only the
frozen code copy, so the runs can be resubmitted with scripts/p3_submit.sh if the replication comes back.

### Grace only (Harsh, 2026-10-28: "can we only use grace for this work? leave mira and vega for my other projects")

Every scripts/*.sbatch now has `--nodelist=grace` (cpu.sbatch included; P3 scripts untested on the A6000).
The 8 menu/spread runs already on mira and vega (8-16 of 20 epochs done) are left to finish (~20 min);
moving them would have queued them behind the 4 grace jobs and added ~4 h.

### Menu and spread reward, results (2026-10-28; shift 0.5, 6 seeds, epochs 3-20)

Jobs 426342-426353 (mira/vega/grace), rnd3x3 job 426341 (200 seeds). results/summary/p4b_interface.csv,
figures/p4b_interface. Paired by seed; diversity in Hamming units, fitness and next problem in % of no AI.

| arm | diversity | help while assisted | next problem | answers that help | distinct |
|---|---|---|---|---|---|
| edit, never updated | -0.02 | -3.3% | -7.9% | 3.0% | 0.32 |
| edit, thumbs | +0.05 | -2.0% | -2.7% | 2.6% | 0.38 |
| (a) edit, best of 3, never updated | +0.36 (p = 0.09) | +2.5% | -3.0% | 3.7% | 0.30 |
| (b) edit, spread reward | -0.07 | +4.6% | -1.1% | 3.1% | 0.44 |
| random edit (200 seeds) | +0.14 | +15.2% | +4.9% | 4.1% | 0.87 |
| random edit, best of 3 (200 seeds) | +0.13 | +24.7% | +6.4% | | |

(a) Menu: vs single edits +0.38 diversity (p = 0.08), +5.8 while assisted (p = 0.18), +4.9 next problem
(p = 0.24). Vs random best of 3: -15.4 while assisted (p = 0.03), +0.9 next problem (n.s.). The three
temperature-1 answers are often the same edit: 1.87 distinct of 3 per request.
(b) Spread: vs thumbs +6.6 while assisted (p = 0.08), -0.12 diversity, +1.6 next problem (both n.s.). First
trained arm above no AI while assisted (+4.6%, n.s.). Gap to random edits -8.1 (p = 0.25), down from -14.7 for
thumbs; distinct 0.38 -> 0.44.
Reading: neither keeps up with random proposals of the same form while assisted. Both keep the culture as
diverse as no AI (the edit form does that, not the reward). Spread is the best trained arm so far, but the
gain over thumbs is not significant at 6 seeds. On these 6 seeds random edits are weaker than on 200
(+12.6% / -2.7% vs +15.2% / +4.9%), so next-problem numbers near zero are within seed noise.

### Rewards that score unhelpful answers (Harsh, 2026-10-28: "can we brainstorm the reward design more and then run more trained versions?"; plan ratified: "yes go ahead")

Why the rewards so far changed nothing: GRPO z-scores rewards within each group of G = 8 answers to one request.
With ~3% of answers helping, about 4 in 5 groups have no helpful answer (no signal under thumbs), and in a group
with one helpful answer any reward that only rescales it (spread, P/H-creativity, gain) gives exactly the thumbs
update. They differ only in groups with two different helpful answers (~2%, estimated). So spread was nearly
thumbs; its +6.6 is most likely noise, and the "more spread seeds" proposal is dropped. Design rule: a reward
must separate answers thumbs scores the same, mostly the unhelpful ones. Brainstorm (GA niching, ecology,
economics, education, LLM RL) in the 2026-10-28 conversation and on the design-space page.

Round 1, four arms, seeds 0-5, shift 0.5, update samples at T 2, deployment T 1; invalid answers score below
any valid one (else junk dodges the penalties):
- dupE3T2 (edits <=3): thumbs - 0.5 * (copies of the answer in the group - 1) / 7. Implicit fitness sharing
  (Smith, Forrest & Perelson 1993; Rosin & Belew 1997); DARLING (Li et al. 2025). Can training buy the
  variety T 2.5 gave? Jobs 426458-426468 even.
- crowdT2 (full answers): thumbs - 1.0 * share of this epoch's askers within 2 flips of the answer. Fitness
  sharing with a radius (Goldberg & Richardson 1987); congestion pricing (Vickrey 1969); Kitcher 1990. Does
  paying for crowding stop the narrowing where it happens (thumbs full: -0.92)? Jobs 426459-426469 odd.
- rateE3T2 (edits): signed rating 10 * (F(x') - F(m)). Is sparse feedback the problem? Watch edit size (risk:
  1-flip edits lose least). After reading round 1.
- dupT2 (full answers): duplicate penalty. After reading round 1.
New logging (all runs from now): edit_size per round; group_any (groups with a helpful answer) and
group_multi (groups with >= 2 distinct helpful answers) per epoch; requests.jsonl with every request's round,
team slot, asker code, answer and helped, to count repeat answers to the same team. Fake-LLM CPU test: groups
with any signal 25% thumbs, 81% dup, ~100% crowd and rate.

### Correction: groups with two different helpful answers (2026-10-28)

The ~2% in the entry above was an estimate. First epoch of the new runs (untrained model, update samples at
T 2): groups with any helpful answer 0.105 / 0.254 (dupE3T2 s0 / s1) and 0.113 / 0.098 (crowdT2 s0 / s1);
groups with two different helpful answers 0.059 / 0.098 and 0.059 / 0.023. Help clusters in a few easy
askers, so 2-10% of groups, not ~2%. The rule stands (75-90% of groups give thumbs no signal; rescaling
rewards act only in the multi-helpful groups), but "spread's +6.6 is most likely noise" was too strong: it is
open. Not adding spread seeds unless asked. Page updated (version 7).

### Duplicate and crowding penalty, results (2026-10-28)

Round 1 done (jobs 426458-426469, grace, all 20 epochs). Shift 0.5, 6 seeds, epochs 3-20, paired by seed,
% vs no AI (scripts/analysis/p4b_interface.py, p4b_same_askers.py).
- dupE3T2: diversity +0.09 vs no AI (thumbs edits +0.06); fitness while assisted +10.9% (thumbs -2.0%);
  next problem -3.2% (thumbs -2.7%). vs thumbs edits: fit +12.9 (p=0.004), div +0.03, next -0.5. vs rnd3:
  fit -1.8 (p=0.59), div +0.06, next -0.5. Distinct answers per update group 5.6 -> 8.0 of 8 by epoch 4 in
  every seed; 87% of deployed answers unique (thumbs 38%); 1.6% repeats to the same unchanged team; edit size
  2.78 (rnd3 2.0). On its own requests it helps 4.11% of the time, rnd3 on the same askers 4.11%.
  Reading: the penalty turns the LLM into a random edit generator. The 15-point gap to random edits under
  thumbs was repetition; training removes it but does not get past random (the model never sees the
  landscape). The edit form already kept diversity; this adds fitness.
- crowdT2: diversity -0.12 vs no AI (thumbs full -0.92, never updated -0.62); vs thumbs full div +0.80
  (p=0.051), fit -1.3, next +3.7; vs random full answers fit -17.8 (p=0.003). Help rate 3.0% (thumbs full
  6.0%); on its own requests 3.0% vs 5.6% for a random module. Seeds split: s1, s3 collapsed to one answer
  (shown 86-100% of the time, 60-68% of askers within 2 flips of it; diversity 2.2-2.3, like thumbs); s2, s4
  and partly s0 settled on a module far from everyone that almost never helps (help 0.4%; diversity 2.9-3.5);
  s5 learned varied answers (5% top share, diversity 3.0).
  Reading: on average it stops the narrowing, but mostly by making the assistant less useful, not more
  varied. Two failure modes: a fully collapsed group gets zero advantage under any reward, and the
  full-answer model starts near collapse (1.6-2.5 distinct answers of 8 at T 2); and charging for nearness to
  askers pays the model to be far and useless.
Round 2 submitted after reading this, as planned (grace, jobs 427124-427135): rateE3T2 s0-5 (even) and dupT2
s0-5 (odd). dupT2 is now the key one: does the duplicate penalty work on full answers, which start near
collapse?

### Getting past random: memory screen (Harsh, 2026-10-28: "meanwhile, brainstorm how to get past random"; plan ratified: "go ahead")

Diagnosis (scripts/analysis/p4b_memory_sources.py, dupE3T2 logs and landscapes, epochs 3-19). At rho 1, K 9,
M 10 every module choice depends on all the others, so the module part is a random lookup table: fitness
correlation between a module and its 1-flip neighbour 0.00 (2 flips 0.00; Kauffman's K = N - 1 case). A
module's value does not depend on the rest of the approach (corr 0.999), so what helps one team helps any worse
team. After a 50% shift a module's value correlates 0.44 with its old value. So structure-based operators
(crossover, differential evolution, PBIL/UMDA, step-size rules, bit priors) cannot beat random here, and neither
can a reward change alone; only knowing which modules are good can. Help rate on dup's own requests (open loop):
random edit 4.1%; a module that helped someone last epoch, within 3 flips, 7.8%; perfect memory of last epoch's
problem 14.0%; the most recent module that helped someone this epoch, within 3 flips, 27.0% (98% of requests
have one); another team's helpful edit applied to this team 9.9%; some edit helps 56%. The strong source is what
helped others this epoch, i.e. copying through the assistant, the channel that narrows cultures (PCC -1.00).
Target that exists: perfect-knowledge best edit <=3, +55% with no diversity loss.

Step 1, ratified: screen memory assistants without the LLM in the P4a sim (rho 1, turnover 0.01, shift 0.5,
200 seeds, CPU on grace). They see only what a provider has: the asker's approach and the log of which shown
answers helped. Answer a remembered helpful module within 3 flips of the asker's, else a random edit of 1-3
flips (rnd3). Arms: this-epoch memory (most recent); rarest in the culture (fitness sharing); quota (a module
leaves memory after 3 teams adopted it from the assistant); 4 separate memories (team i in group i mod 4);
memory used with probability 0.25 / 0.5, else random; last epoch's memory only (the analog of per-epoch
retraining). Compare with rnd3, best edit <=3 (rad3), PCC, no AI. Step 2 after reading it: the winning design
as memory in the LLM's prompt with the dup reward, next to fine-tuning on thumbs-up data as the realistic
baseline.

### Memory screen, results (2026-10-28; jobs 427296, 427303, grace CPU; results/p4a_mem/shift0.5)

rho 1, turnover 0.01, shift 0.5, 200 seeds, epochs 3-20, paired with no AI (scripts/analysis/p4a_mem.py,
figures/p4a_mem, results/summary/p4a_mem.csv). Memory assistants in src/culture.py (Memory class). After the
first job I added six arms (other mixing rates; mixing with quota or separate memories; last epoch mixed) and a
split of help rates by source, then reran all memory arms (identical fitness; checked seed 0). Diagnosis
numbers: the repo script gives 8.1% for "last epoch's log" (7.8% in the entry above; random draws).

| design | diversity vs no AI | fitness while assisted | next problem | answers that help (memory / random) |
|---|---|---|---|---|
| memory always | -0.16 | +10.8% | +3.0% | 6.3% (7.0 / 17.9) |
| memory on 75% of answers, else random edit | -0.20 | +22.5% | +3.5% | 8.9% (11.4 / 3.7) |
| 50% | -0.09 | +24.0% | +4.5% | 8.4% (14.5 / 3.1) |
| 25% | -0.02 | +22.5% | +4.3% | 7.0% (19.0 / 3.2) |
| 10% | +0.06 | +19.4% | +4.4% | 5.4% (23.4 / 3.5) |
| 50%, 3 adoptions each | -0.01 | +19.3% | +4.4% | |
| 50%, 4 separate memories | -0.06 | +22.3% | +5.4% | |
| 25%, 3 adoptions each | +0.07 | +19.3% | +5.1% | |
| always, rarest in culture | -0.15 | +8.3% | +1.7% | |
| always, 3 adoptions each | -0.07 | +16.1% | +3.9% | |
| always, 4 separate memories | -0.15 | +17.1% | +2.8% | |
| last epoch, always | -0.04 | +0.9% | -0.5% | 3.4% |
| last epoch, 25% | +0.12 | +18.2% | +5.2% | 4.3% |
| random edit | +0.14 | +15.2% | +4.9% | 3.8% |
| best edit <=3 (perfect knowledge) | -0.01 | +55.3% | +10.9% | 7.9% |
| best answer in culture (PCC) | -1.00 | +20.2% | +2.2% | |

vs random edits (paired, 95% CI): memory always fit -4.4 [-5.6, -3.1], div -0.30, next -2.0 [-3.4, -0.6];
50% fit +8.8 [+7.6, +10.0], div -0.24, next -0.5 n.s.; 25% fit +7.4 [+6.1, +8.6], div -0.17, next -0.6 n.s.;
10% fit +4.3, div -0.08; last epoch always fit -14.3, next -5.5; last epoch 25% fit +3.0 [+1.9, +4.2], div -0.02.
Quota and separate memories on top of mixing: fit -1.6 to -4.7, div +0.04 to +0.09, i.e. the same trade as
using memory less (25% with quota = 10% plain: +19.3 / +0.07 vs +19.4 / +0.06).
Reading:
- Memory alone is a copying channel: it beats random on help rate (6.3% vs 3.8%) but loses on fitness and
  narrows the culture, like PCC on a small scale. Mixed with random edits it beats random by 4 to 9 points while
  assisted, at about the no-AI level of diversity at 25% or less, with no detectable later cost or gain.
  Random edits find new good modules, memory spreads them; each needs the other (explore vs exploit, March 1991).
- Still far below perfect knowledge (+55%). Memory gets about a fifth of the way from random to best edit.
- What a once-per-epoch retrained model could know (last epoch) is weak: worse than random if always used,
  +3 if mixed in.
- Per answer, a remembered module helps far more often than a random edit at every mixing rate (e.g. 14.5% vs
  3.1% at 50%). So a learner rewarded per answer (thumbs) should copy more and more, toward "always", the worst
  design here: the individually best answer is not the best for the culture (Rogers 1988; Rendell et al. 2010).
  This is a prediction for the LLM, not yet a result.

### Step 2: memory in the LLM's prompt (2026-10-29; ratified: "Go ahead")

Same setup as thumbsE3T2 and dupE3T2 (edits <=3, top-3 approaches in the prompt, update samples at T 2,
deployed at T 1, shift 0.5, 6 seeds, grace), plus a memory block (`--memory 3`). The prompt lists up to 3
modules that the assistant's answers moved some team to this epoch (the provider's thumbs log: an answer that
helped is one the team adopted), within 1-3 flips of the asker's module, most recently helpful first, written as
the choices this team would flip to reach them. Memory starts empty each epoch (new problem) and holds answers
that helped in earlier rounds of the epoch, as the sim's mem_now_recent. No scores are shown. Update prompts
re-render the listing the asker saw. Runs memthumbsE3T2 (thumbs) and memdupE3T2 (duplicate penalty, beta 0.5).

Readouts per epoch: share of requests with a listing; share of answers that copy a listed edit (epoch 0 is the
untrained model's rate); help rate of copies and of other answers; share of the 8 update samples that copy.
Plus the usual diversity, fitness while assisted, next problem, vs the no-memory arm on the same seeds, rnd3 and
the sim memory arms. Prediction (from the sim, before the runs): under thumbs copying rises across epochs and
fitness falls toward memory-always (+11%) with the culture narrowing; dup may hold a mix. Later, if useful: the
never-updated model with memory, and the provider showing the memory on 50% of requests with dup. Fine-tuning
on thumbs-up data is deferred.

Step 2 build and smoke (2026-10-29). Code: `--memory` in src/llm/culture_llm.py, `memory=` in render
(src/llm/online.py). CPU check with a stub LLM: with memory off, rounds/epochs/requests logs are byte-identical
to the committed code (thumbs and dup); with memory on, listings rebuilt from requests.jsonl match the logged
ones exactly; a stub that always copies the first listing helps 5.3% (copies 7.2%), one that copies half the
time 6.8% (copies 17.3%), a random-edit stub 4.0%, close to the sim's memory-always / p50 / rnd3. A random edit
matches a listed one 1.6% of the time (chance). First GPU smoke ran out of memory in the update's sampling (64
prompts x 8 = 512 sequences with the longer prompt); added `--gen-batch 256`, which generates the same padded
prompts in slices and leaves the update batch at 64 prompts (default 0 keeps the old single call). Smoke, seed 0,
epoch 0, untrained model: 98% of requests had a listing; 58% of answers copied a listed edit (64% in the last 20
rounds; 62% of update samples at T 2); copies helped 7.3%, other answers 4.7%; diversity in the last 20 rounds
1.80 (same model without memory 2.66, no AI 4.45, PCC 1.34); fitness 0.548 (without memory 0.552). Submitted
memthumbsE3T2 and memdupE3T2, seeds 0-5, with --gen-batch 256 (jobs 427538-427549, grace).

### Round 2 results: signed rating, duplicate penalty on full answers (2026-10-29; jobs 427124-427135)

6 seeds, shift 0.5, epochs 3-20, paired by seed (scripts/analysis/p4b_interface.py, p4b_same_askers.py;
figures/p4b_interface). Diversity in Hamming units vs no AI; fitness and next problem in % of no-AI fitness.

| arm | diversity | fitness while assisted | next problem | helps | distinct answers | repeats |
|---|---|---|---|---|---|---|
| edit, signed rating (rateE3T2) | -0.07 | -6.9% | -3.5% | 1.8% | 18% | 96% |
| full answer, duplicate penalty (dupT2) | -0.08 | +20.0% | +2.8% | 5.2% | 85% | 2% |
| full answer, thumbs (thumbsT2) | -0.92 | -1.1% | -8.9% | | | |
| random full answer (rndfull, same seeds) | +0.49 | +15.3% | +0.6% | | | |

Contrasts: rateE3T2 - thumbsE3T2 div -0.12 (p=0.29), fit -4.9 (p=0.29), next -0.9 (p=0.76); rateE3T2 - rnd3 fit
-19.6 (p=0.007). dupT2 - thumbsT2 div +0.84 (p=0.036), fit +21.1 (p=0.028), next +11.7 (p=0.028); dupT2 -
rndfull div -0.57 (p=0.12), fit +4.7 (p=0.27), next +2.2 (p=0.54); dupT2 - crowdT2 fit +22.5 (p=0.004), next
+8.0 (p=0.044). Help rate on each run's own requests: rateE3T2 1.8% vs rnd3 6.0% (below random in all 6 seeds,
-3.3 to -5.2 points); dupT2 5.3% vs rndfull 3.4% (+1.9; seeds 0, 2, 3 equal random to 0.1 point, seeds 1, 4, 5
+2.3 to +6.5).
Reading:
- The signed rating fails. It scores every answer, so every group has a gradient, and the model collapses to
  one answer per approach: 96% of repeat requests get the same answer, which did not help the first time. It
  helps a third as often as random edits on the same askers. Rewarding "less bad" pays for a safe answer, not
  a helpful one. Edit size stayed near 3 (2.66), so the failure is not timidity.
- The duplicate penalty also rescues full answers from collapse: +21 points over thumbs, with diversity and
  next problem recovered. It ends about level with random full answers, as dupE3T2 did with random edits.
  The three seeds that beat random on their own requests answer with a module another asker holds 5-8% of the
  time (1.6-2.5% in the other seeds, about chance, which is roughly 20 held modules / 1024): copying what is
  already in the culture, at a low rate, the same channel the memory screen found.
- Across round 1 and 2: rewards that score unhelpful answers either make the model a random generator (dup,
  both interfaces) or make it worse (crowd helps less, rate collapses). None beats random. Consistent with the
  diagnosis that only knowing good modules can.

### Step 2 results: memory in the LLM's prompt (2026-10-29; jobs 427538-427549)

6 seeds, shift 0.5, epochs 3-20, paired by seed (scripts/analysis/p4b_memory.py, figures/p4b_memory,
results/summary/p4b_memory.csv; also in p4b_interface). Copy shares are among answers to prompts with a listing;
help rates of copies and other answers are pooled over each seed's requests. No time-outs (dup runs 3.6 h).

| arm | diversity | fitness while assisted | next problem | helps | listed | copies | copies help / others | modules in memory | distinct answers | repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| memory, thumbs | +0.01 | -2.4% | -3.9% | 3.9% | 71% | 83% (50-96 by seed) | 5.3% / 1.6% | 14.5 | 16% | 77% |
| no memory, thumbs | +0.05 | -2.0% | -2.7% | 2.6% | | | | | 38% | |
| memory, duplicate penalty | +0.15 | +12.5% | -1.5% | 4.4% | 96% | 4.9% (2-9) | 23.4% / 3.3% | 73.2 | 83% | 3% |
| no memory, duplicate penalty | +0.09 | +10.9% | -3.2% | 4.2% | | | | | 87% | 2% |
| random edit (same seeds) | +0.02 | +12.6% | -2.7% | | | | | | | |

Chance of copying (random edit of 1-3 flips): 1.6-1.7%. Copying by epoch, thumbs: 0.54 (untrained), 0.79, 0.66,
0.69, ... 0.97, 0.91, 0.87, 0.90, 0.95, 0.94, 0.99, 0.86; duplicate penalty: 0.54, 0.41, 0.18, 0.06, then 0.02 to
0.08. Update samples (T 2) follow the deployed share closely. 78% of the thumbs arm's helps come from copies
(dup arm 22%).
Contrasts: memory thumbs - no memory thumbs div -0.04 (p=0.82), fit -0.4 (p=0.93), next -1.2 (p=0.68); memory
thumbs - rnd3 fit -15.0 (p=0.001); memory dup - no memory dup div +0.06 (p=0.69), fit +1.6 (p=0.21), next +1.7
(p=0.41); memory dup - rnd3 fit -0.2 (p=0.95), div +0.12 (p=0.47); memory dup - memory thumbs fit +14.8 (p=0.001).
Reading:
- The trap is real for behaviour: retrained on thumbs, the model copies more and more, from 54% untrained to
  90-99% in the last eight epochs. Per answer a copy helps 5.3% and its own edit 1.6%, so thumbs pays for copying.
- But copying always does not change the culture's outcome relative to the same model without memory: same
  diversity, same fitness, both about 15 points below random edits. Copying finds nothing new: the memory holds
  14.5 helpful modules per epoch, against 73 when the model explores. The sim's memory-always design fell back on
  random edits; the LLM's own edits under thumbs are repetitive and worse than random, so there is little to copy.
  Copying replaced one kind of repetition with another (distinct answers 16% vs 38%).
- The duplicate penalty trains copying out: 8 samples that copy the same listed edit are duplicates of each other,
  so copies are penalised as repeats. Copying falls to 2-8%, near chance. When it does copy, the copy helps 23%
  (the sim at 10% memory: 23.4%), but at 5% that adds +1.6 (n.s.). It stays a random edit generator.
- Neither reward finds the mix the sim says is best (memory on 25-50% of requests: +7 to +9 over random edits).
  Thumbs drives the learner to always copy; the duplicate penalty to never. The mix has to come from outside.

Next, proposed (not ratified): the provider sets the mix. Show the memory on a share q of requests; score prompts
with memory by thumbs (the model learned to copy there) and prompts without by the duplicate penalty (it learned
to explore there); one model, which can tell the two prompt types apart. q = 0.5 and 0.25, 6 seeds, grace, 12
runs. Prediction from the sim (p50, p25): +7 to +9 over random edits with diversity near no AI. Readouts: copy
share on memory prompts, distinct answers on the others, the usual three. Grounding: explore and exploit kept in
separate units (March 1991); social learning pays only when individual learning continues (Rogers 1988; Rendell
et al. 2010).

### Step 3: the provider sets the mix (2026-10-29; ratified: "go ahead, run the trained version first")

As proposed above. Code: `--memory-share q` and `--memory-reward` in src/llm/culture_llm.py. Each request is
shown the memory with probability q (own random stream, so the other draws are unchanged); a request drawn for
memory with no remembered module within 3 flips gets the plain edit prompt, as the sim's p<pct> designs fall back
on a random edit. The reward follows the prompt the model saw: thumbs if it showed a listing, the duplicate
penalty (beta 0.5) otherwise, so the model can tell which reward applies. Memory still records every answer that
helped, shown or not. Update prompts are drawn from the epoch's requests as before, so they carry the same mix.
New update readouts: distinct texts of 8 on prompts with and without a listing.
CPU check with a stub LLM: with the defaults (q 1, no memory reward) rounds/epochs/requests logs are byte-identical
to fda4883 (memory thumbs and memory dup, 6 epochs); with the mix, every update prompt with a listing is scored
by thumbs and every other by dup (0 mismatches of about 15k), listings rebuilt from requests.jsonl match, and the
shown share is 0.500 / 0.250 of requests with a non-empty listing. A stub that copies the first listing when shown
helps 7.5% at q 0.5 and 6.1% at q 0.25, against 5.3% always shown and 4.0% for a random-edit stub (3 seeds), the
same order as the sim's help rates. The stubs' fitness is not comparable: the stub seeded each answer on the prompt
text, so teams holding the same approach got the same "random" edit in a round. With independent draws per
request, the driver with a random-edit stub matches the sim's rnd3 over 100 seeds: fitness while assisted +0.7%
[-1.0, +2.4], diversity +0.03 [-0.06, +0.11], next problem +1.8% [-0.1, +3.7]. So the sim's rnd3 on the same seeds
is a fair reference for the LLM arms (it was used that way since round 1 without this check).
No GPU smoke: the model's path (render, sample, update) is the one memdupE3T2 ran.
Runs memmix50E3T2 and memmix25E3T2, seeds 0-5, edits <=3, update T 2, shift 0.5, --gen-batch 256, 6 h limit
(jobs 427923-427934, grace; 3 waves of 4, about 11 h).
Fixed-rule comparison (the provider copies on the memory share, no training there) deferred until these are read.

### Step 3 results: the provider sets the mix (2026-10-30; jobs 427923-427934)

6 seeds, shift 0.5, epochs 3-20, paired by seed (scripts/analysis/p4b_memory.py, p4b_same_askers.py,
figures/p4b_mix, results/summary/p4b_memory.csv). All 12 runs finished (3.1-3.5 h). Diversity in Hamming units vs
no AI; fitness and next problem in % of no-AI fitness. Copies are among answers to prompts with a listing.

| arm | diversity | fitness while assisted | next problem | helps | listed | copies | copies help / others | modules in memory | distinct of 8, no listing / listing |
|---|---|---|---|---|---|---|---|---|---|
| memory on 50% (memmix50E3T2) | +0.07 | +19.5% | +2.1% | 7.8% | 47% | 91% | 14.1% / 1.6% | 37.4 | 7.88 / 1.83 |
| memory on 25% (memmix25E3T2) | -0.02 | +17.6% | +0.9% | 6.8% | 24% | 86% | 19.2% / 2.7% | 57.8 | 7.93 / 2.78 |
| memory always, thumbs | +0.01 | -2.4% | -3.9% | 3.9% | 71% | 83% | 5.3% / 1.6% | 14.5 | 2.69 (all prompts) |
| memory always, duplicate penalty | +0.15 | +12.5% | -1.5% | 4.4% | 96% | 4.9% | 23.4% / 3.3% | 73.2 | 7.87 (all prompts) |
| random edit (sim rnd3, same seeds) | +0.02 | +12.6% | -2.7% | | | | | | |

Copying by epoch (prompts with a listing): 50%: 0.53 (untrained), 0.59, 0.65, 0.75, ... 0.99 in 6 of the last 9
epochs; 25%: 0.43 to 0.93-0.98. It copies the first (most recently helpful) listing 96% of the time (memory always,
thumbs: 88%). On prompts without a listing it helps 3.1% (50%) and 3.6% (25%), about a random edit's rate.
Help rate on each run's own requests vs a random edit on the same askers: 50% 7.9% vs 2.9% (+4.9 points, all 6
seeds), 25% 6.8% vs 3.4% (+3.4, all 6); memory always dup 4.4% vs 3.9%; memory always thumbs 3.8% vs 5.0%.
Contrasts: memmix50 - rnd3 fit +6.9 (p=0.052; 5 of 6 seeds above; epochs 3-11 +4.3, 12-20 +9.3, p=0.014), next
+4.7 (p=0.042; 6 of 6), div +0.05 (p=0.62); memmix50 - memory always dup fit +7.1 (p=0.024), next +3.6 (p=0.059),
div -0.08 (p=0.63); memmix50 - memory always thumbs fit +21.9 (p<0.001); memmix25 - rnd3 fit +5.0 (p=0.40; 3 of 6
above), next +3.6 (p=0.44), div -0.04 (p=0.82); memmix25 - memory always dup fit +5.2 (p=0.17); memmix50 - memmix25
fit +1.9 (p=0.70), div +0.08 (p=0.54).
Sim, same designs (P4a, 200 seeds), vs random edit: 50% fit +8.8, 25% +7.4, always -4.4; next problem -0.5, -0.6,
-2.0; diversity vs no AI -0.09, -0.02, -0.16 (random edit +0.14).
Reading:
- First LLM design here that beats random edits, by about what the sim predicted (+6.9 vs +8.8 at 50%; +5.0 vs
  +7.4 at 25%, noisier), with the culture's diversity at the no-AI level. The next problem is not hurt: +4.7 vs
  random edits, more than the sim's -0.5; part of that may be the driver's own offset (+1.8 [-0.1, +3.7] in the
  random-edit check), and against memory-always dup in the same driver it is +3.6 (p=0.059).
- One model learns both behaviours: on prompts with a listing, thumbs drives copying from 53% to ~99%, and what it
  copies is the most recent listing, i.e. it learns the sim's rule; on prompts without one, the duplicate penalty
  keeps it giving 7.9 distinct answers of 8 that help about as often as random edits. The provider's share sets
  the mix; the learner does not have to find it, which step 2 showed it cannot.
- Why copying pays here and not under memory-always thumbs: the exploring half keeps filling the memory (37 and 58
  helpful modules per epoch vs 14.5), so copies help 14-19% vs 5%. Social learning pays only while individual
  learning continues (Rogers 1988; Rendell et al. 2010); the provider keeps the two in separate prompts (March 1991).
- Since the model copies the first listing 96% of the time, the fixed-rule comparison would test nearly the same
  policy. I suggest dropping it.
- Limits: 6 seeds (fitness p=0.052 at 50%), one model (Qwen3-4B), one setting (rho 1, shift 0.5, edits <=3).

Next, proposed (not ratified): firm up the headline with seeds 6-11 of both mix arms (12 runs, grace, ~10 h),
reading fitness and next problem vs random edits on 12 seeds. After that, if it holds, one robustness check
(shift 0.2 or rho 0.8) before writing it up.

### Step 3, seeds 6-11 (2026-10-30; ratified: "Yes go ahead")

memmix50E3T2 and memmix25E3T2, seeds 6-11, same code and flags as seeds 0-5 (src unchanged since 31f9b06; jobs
428486-428497, grace, 6 h limit). Read on 12 seeds: fitness while assisted and next problem vs random edits (sim rnd3
on the same seeds), diversity vs no AI, copying and distinct-answer readouts. The no-memory and memory-always arms
stay at seeds 0-5. The other setting (shift 0.2 or rho 0.8) is chosen after reading these, if the gain holds.

### Step 3 on 12 seeds (2026-10-30; jobs 428486-428497)

Both mix arms, seeds 0-11, same reading as the 6-seed entry (results/summary/p4b_memory_12seeds.csv,
figures/p4b_mix_12seeds; `p4b_memory.py results/p4b_s05 memmix50E3T2 memmix25E3T2`). The 12 new runs took
3.1-3.5 h each. Fitness is scaled by the no-AI mean over the 12 seeds, so seeds 0-5 read slightly differently from
the 6-seed entry (+7.1 instead of +6.9 at 50%).

| arm | diversity | fitness while assisted | next problem | helps (random edit, same askers) | listed | copies | copies help / others | modules in memory | distinct of 8, no listing / listing |
|---|---|---|---|---|---|---|---|---|---|
| memory on 50% (memmix50E3T2) | +0.03 | +22.6% | +4.8% | 8.0% (2.9%) | 47% | 92% | 14.4% / 1.8% | 37.3 | 7.89 / 1.86 |
| memory on 25% (memmix25E3T2) | +0.00 | +20.7% | +3.2% | 6.7% (3.3%) | 24% | 88% | 18.8% / 2.5% | 55.0 | 7.92 / 2.41 |
| random edit (sim rnd3, same 12 seeds) | +0.12 | +14.9% | +4.1% | | | | | | |

Contrasts vs random edits, paired by seed, 95% CI:
- 50%: fit +7.7 [+2.6, +12.9] (p=0.007; 9 of 12 seeds above; seeds 0-5 +7.1, 6-11 +8.4; epochs 12-20 +10.1,
  p=0.013, 11 of 12). Next +0.7 [-5.8, +7.1] (p=0.83; seeds 0-5 +4.9, 6-11 -3.6). Div -0.09 [-0.30, +0.11] (p=0.35).
- 25%: fit +5.9 [-1.0, +12.7] (p=0.088; 7 of 12), next -0.9 [-8.0, +6.2] (p=0.78), div -0.11 [-0.32, +0.09] (p=0.25).
- 50% - 25%: fit +1.9 (p=0.52), next +1.6 (p=0.52), div +0.02 (p=0.84).
Help on its own requests vs a random edit on the same askers: 50% +5.1 points, 25% +3.4, above in all 12 seeds.
Sim, same designs on these 12 seeds, vs rnd3: 50% fit +10.0, next -0.0, div -0.09; 25% fit +8.8, next +2.8,
div -0.14 (200 seeds: 50% +8.8, -0.5, -0.23; 25% +7.4, -0.6, -0.17).
Reading:
- The gain while assisted holds at 50%: +7.7 on 12 seeds, the same in both halves, about three quarters of what
  the sim predicts on the same seeds (+10.0). Diversity stays at the no-AI level (+0.03), as the sim predicts.
- The next-problem gain does not hold. Seeds 6-11 give -3.6, and the 12-seed mean is +0.7, which is what the sim
  predicts (-0.0 on these seeds, -0.5 on 200). The +4.7 at 6 seeds was noise and I read too much into it in the
  6-seed entry. What stands: the next problem is not hurt (+4.8 vs no AI, like random edits) and not helped
  beyond random edits.
- 25% is weaker and not separated from random edits (p=0.088), nor from 50% (+1.9, p=0.52).
- On this setting the answer to the question is: with the provider setting the mix, the LLM helps more than
  random edits while assisted, without narrowing the culture and without a cost on the next problem. It does not
  leave the culture better able to adapt than random edits do.

Next, proposed (not ratified): one other setting, rho 0.8 (shift 0.5), 50% arm only.
- Why rho 0.8 rather than shift 0.2: the gain rests on a module that helped one team helping others. At rho 1 a
  module's fitness depends only on its own 10 bits, so that is exactly true; at rho 0.8 each decision depends on 2
  decisions in the other module, so a remembered module's value depends on the rest of the team's approach. That
  is the assumption most likely to break. Shift 0.2 makes the next problem closer to the current one (oracle
  reach-3 next problem +30% vs +11% at shift 0.5) and mostly rescales the next-problem readout.
- Order: (1) sim screen at rho 0.8 and at shift 0.2 (none, rnd3, memory on 50% / 25% / always; 200 seeds; CPU
  job on grace, minutes), since no rnd3 or memory reference exists at either; read whether the sim still predicts
  a gain. (2) If it does at rho 0.8, memmix50E3T2 at rho 0.8, seeds 0-11 (12 runs, grace, ~12 h).

### Sim screen at rho 0.8 and shift 0.2 (2026-10-30; ratified: "yes, run the sim screen first")

src/culture.py unchanged since the shift-0.5 memory screen (md5 same on ada and local). 200 seeds, mu 0.01, use 0.5,
designs none, rnd3, rad3, pcc, mem_now_recent, mem_now_recent_p75/p50/p25/p10. rho 0.8 at shift 0.5 ->
results/p4a_mem/shift0.5/rho0.8_*; rho 1 at shift 0.2 -> results/p4a_mem/shift0.2/rho1_*. CPU on grace, jobs
429273 (p4a_mem_r08) and 429274 (p4a_mem_s02). Read: memory on 50% and 25% minus rnd3, fitness while assisted,
next problem, diversity. The LLM run at rho 0.8 waits on this and on a go.

### Sim screen results at rho 0.8 and shift 0.2 (2026-10-30; jobs 429273, 429274)

200 seeds, epochs 3-20, paired by seed (scripts/analysis/p4a_mem.py <dir> <rho>; results/summary/
p4a_mem_shift0.5_rho0.8.csv, p4a_mem_shift0.2_rho1.csv; figures with the same suffixes). Both jobs completed (15
and 11 min). Memory minus random edits, 95% CI; fitness in % of no-AI fitness, diversity in Hamming units.

| setting | random edit vs no AI | memory 50%: assisted | next | diversity | memory 25%: assisted | next | diversity | memory always: assisted |
|---|---|---|---|---|---|---|---|---|
| rho 1, shift 0.5 (earlier) | +15.2 | +8.8 [+7.6, +10.0] | -0.5 | -0.23 | +7.4 [+6.1, +8.6] | -0.6 | -0.17 | -4.4 |
| rho 0.8, shift 0.5 | +8.7 | +1.9 [+0.7, +3.1] | -1.3 | -0.17 | +2.7 [+1.6, +3.9] | +0.5 | -0.11 | -7.0 |
| rho 1, shift 0.2 | +18.5 | +6.8 [+5.1, +8.5] | +1.2 | -0.19 | +5.2 [+3.4, +6.9] | +0.4 | -0.15 | -1.8 |

Per answer at rho 0.8, 50%: memory helps 9.5%, random 3.6% (rho 1: 14.5% / 3.1%). Best edit (<=3) is still +50.4 over
no AI at rho 0.8, so the headroom is there; memory does not reach it. Why: a module's fitness under one team's other
half correlates 0.14 with its fitness under another's at rho 0.8 (0.999 at rho 1; 6 landscapes x 10 pairs of random
other halves). What helped one team mostly does not carry to the next.
Reading:
- The memory gain depends on modules carrying across teams. At rho 0.8 memory on 50% beats random edits by 1.9, a
  fifth of the rho 1 gain, and still narrows the culture a little (-0.17). Memory always is worse than at rho 1.
- At shift 0.2 the picture is the same as at shift 0.5: +6.8 while assisted, next problem level with random edits.
- For the LLM at rho 0.8: it reached about 3/4 of the sim's gain at rho 1, so the expected gain is about +1.5. On 12
  seeds its interval is about +-5 (seed SD about 8), so a run there cannot tell it from random edits (it would need
  well over 100 seeds). The sim already gives the answer.

Next, proposed (not ratified):
- Do not run the LLM at rho 0.8. Report rho 0.8 as a sim result: memory of what helped is worth little when a
  module's value depends on the rest of the approach.
- For the one LLM check in another setting: memmix50E3T2 at shift 0.2 (rho 1), seeds 0-11, grace, ~12 h; the
  sim predicts +6.8, so about +5 for the LLM, which 12 seeds detect about half the time (24 seeds about 80%). The
  rnd3 and none references at shift 0.2 now exist (results/p4a_mem/shift0.2). Or skip it and write up.

### The LLM at shift 0.2 (2026-10-30; ratified: "run the LLM at 20% shift, 12 seeds")

memmix50E3T2 at shift 0.2 (rho 1), seeds 0-11, same flags as at shift 0.5 except --shift 0.2 (--reward dup
--memory-reward thumbs --memory-share 0.50 --memory 3 --edits 3 --update-temperature 2 --gen-batch 256); src
unchanged since 31f9b06 (md5 same on ada). Out results/p4b_s02, jobs 429335-429346, grace, 6 h limit; another
user holds 2 of grace's 4 GPUs, so about 20 h if that stays. Read on 12 seeds: fitness while assisted, next problem
and diversity vs the sim's rnd3 and none at shift 0.2 on the same seeds (results/p4a_mem/shift0.2); sim prediction
+6.8 while assisted, next +1.2. p4b_memory.py now reads the run's shift and picks the matching sim references
(outputs get a _shift0.2 suffix); p4b_same_askers.py takes ROOT from the environment.
