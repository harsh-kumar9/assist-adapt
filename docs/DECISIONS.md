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

## P3 oracle baseline (2026-10-14)

Oracle assistants through the P3 driver, 300 seeds, rho = 1, use 0.5, four problems of 60
rounds, each redrawing 20% of the last one's tables. Every community converges within the first
problem (PCC by round 10). Fitness while assisted, % vs no AI, problems 1-4: PCC +27.8, +4.0,
-3.6, +2.7; SBP +13.2, +5.6, +3.4, +4.1. PCC works by copying a peer's module, so once it has
made everyone hold the same module it has nothing left to offer, and after the first problem it
is no better than no AI. SBP proposes flips nobody holds and keeps helping. The next-problem
AI-off probe differs by a few percent with CIs of about +-5%, so the continual effect shows up
in fitness while assisted, not in the probe.

For the LLM this sets the question sharply: an assistant whose knowledge comes from the
community behaves like PCC and runs dry; one that keeps generating new options behaves like SBP.
Whether an LLM trained with R_now keeps generating or collapses is what the R_now and KL arms
measure.
