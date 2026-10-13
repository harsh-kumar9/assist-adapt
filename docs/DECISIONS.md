# Decisions

Status: `paper` = copied from Fazelpour, O'Brien & Rubin (arXiv:2608.19390), no judgment involved.
`proposed` = my call for the pilot, waiting for Harsh. `ratified` = Harsh agreed.

## D-001 Scope: one model, one new probe (proposed, 2026-10-13)

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

## D-003 Things the paper leaves open (proposed)

- Network density: p_edge = 0.05 (mean degree about 5). The paper does not say. Check 0.1 later.
- Fitness scale: (f / max f)^8, the Lazer & Friedman (2007) transform. The paper says
  "transformed and normalized so that max f(d) = 1". Every rule in the model compares fitness
  values, so the transform does not change any trajectory, only the numbers we report.
- Synchronous updates: everyone acts on the state at the start of the round.
- Ties: argmax takes the first index. Payoffs are continuous, so ties only happen between agents
  holding the same configuration.
- Same landscape, network, starting points and per-round coin flips across conditions for a
  given seed, so all comparisons are paired.

## D-004 The shift (proposed)

The related problem keeps the dependency structure and redraws the payoff tables of 20% of the
decisions (4 of 20), picked at random across both modules, independent of anything in the run.
We also run shift = 0 (same problem) so we can separate "the problem changed" from "the AI left".
Probe times T in {10, 30, 60, 100}; 30 rounds after each probe.

Why 20%: enough that the old optimum is usually no longer optimal, small enough that old
experience still helps. 50% is a later sensitivity check.

## D-005 New assistant rules (proposed)

Both pick from the same candidates SBP sees (the M single flips of the agent's computational
module) and only among improving flips:

- SBP-rand(q): a random flip among the q best.
- SBP-div(q): among the q best, the flip that moves the agent furthest from the rest of the
  population (largest increase in summed Hamming distance), ties broken by fitness.

q in {3, all}. Same information and the same evaluation budget as SBP. Rank-based, so the scale
transform in D-003 does not matter. Randomising and diversifying are separate rules so we can
tell "less greedy" apart from "diversity-seeking".

## D-006 P1 outcomes (proposed)

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

## D-008 P2 learner (proposed, gated on P1)

A small selector over the same candidate flips: softmax over a linear score of a few features
(fitness gain, gain rank, how common the agent's current value is on that decision, round).
Fit with an evolution strategy on the episode reward, trained on one set of landscape seeds and
tested on held-out seeds. Three rewards: fitness while assisted only; that plus fitness after
with the AI still on; that plus fitness after a shift with the AI off. A neural policy with PPO
only if the linear one is clearly too weak.

Why ES and a linear policy: a handful of parameters, no credit-assignment machinery, and the
learned weights are readable. If this finds nothing the simple rules miss, a bigger learner is
unlikely to be worth it.

## D-009 LLM and human stages deferred (proposed)

Decide after P2. Options: (a) GRPO an LLM assistant (Qwen3.5-9B on mira/vega) on a text version
of the task with the P2 rewards, (b) a browser version of the task with people, following the
design of Kumar et al. 2024 (assisted rounds, then unassisted rounds on a shifted problem).
