# Plan

## The question

Can an AI assistant make a community better at today's problem while leaving it worse at the
next one? If so, can we train the assistant so that does not happen?

"Worse at the next one" here means one concrete thing: after some rounds with the assistant,
the problem changes and the assistant goes away. How well does the community do over the next
30 rounds, compared with a community that never had the assistant?

## Why this setup

We use the NK model of Fazelpour, O'Brien & Rubin (2026, arXiv:2608.19390) without changes.
100 agents search a rugged landscape of 20 binary decisions. Each round an agent copies its
best neighbour, asks the AI, or tries a random change on its own. The paper compares three AI
designs (NP, TDR, SBP) and measures fitness and diversity over 100 rounds.

Their model already has time in it, and almost every community converges to one solution by
round 100, with or without AI. So "run it longer" adds nothing. What it does not have is any
test of what the community can do after the assistant. That is the one thing we add.

Agents never get worse at searching in our model. They do not forget, and their preferences do
not change. The only thing the assistant changes is where agents are on the landscape. If a
gap shows up after the shift, it comes from where the assistant left people, not from a skill
decay we wrote in.

## What we dropped

From the brainstorm, and why:

| Dropped | Why |
|---|---|
| Skill decay, forgetting, preference change | Free assumptions. The result would restate them. |
| Assistant retraining on community outputs, shared archives, cultural transmission | Separate loops. Each needs its own experiment once the first result holds. |
| LLM-simulated humans | LLM populations homogenise on their own, so they would produce the effect we want to measure. |
| Learned reward models, LLM judges | The simulator computes the reward exactly. Nothing to learn yet. |
| Option-value formulas, empowerment estimators, per-user constraints | Too abstract for a first result. "Fitness after a shift with the AI off" is the concrete version. |
| Big factorial over network type, H, K, p_social | One setting from the paper first (K=9, H=10, p_social=0.1). Sweeps after something works. |

## Pilots

Every pilot ends at a gate. We read the output and decide before starting the next one.

### P0. Replicate the paper (done)

The three AI designs over modularity rho 0.6-1.0 and use rate 0.125-0.875, 1000 seeds, 100
rounds. Measures as in the paper: final fitness, early fitness, diversity AUC, each as percent
difference from the no-AI community on the same seed.

Gate: the paper's qualitative findings come back. NP helps only at high modularity and moderate
use and hurts at high use. NP cuts diversity by about 25% on average. TDR beats NP at rho = 1.
SBP helps across modularity levels and keeps diversity at low use. If they do not come back,
we fix the model before anything else.

Status (1000 seeds): passed. NP cuts diversity by 23.8% on average, TDR beats SBP at every use
rate at rho = 1, SBP helps everywhere. One difference: NP's gains start at rho = 0.9, not 0.8.

### P1. Help now, adapt later (done)

Same model, plus probes. At T = 10, 30, 60 and 100 we copy the community and run 30 more rounds
in four cells: same problem or shifted problem (20% of payoff tables redrawn), AI on or off.
Assistants: the paper's three, plus four rules that pick among the same single-flip candidates
as SBP (random or diversity-seeking among the q best improving flips, q = 3 or all).
rho in {0.6, 0.8, 1.0}, use in {0.25, 0.5, 0.75}, 1000 seeds.

The main figure plots fitness while assisted (x) against fitness after a shift with the AI off
(y), one point per assistant. If x and y rank the assistants the same way, the evaluation we
use today is good enough and there is no problem to fix.

Gate to P2: (1) at least one assistant looks helpful on x and harmful or neutral on y, with
CIs that exclude zero; (2) the gap is not closed by the simple rules (random or diversity-seeking
among near-best flips). If (1) fails, we write up a null: in this model, how an assistant
looks during use is a good guide to what it leaves behind. If (1) holds and (2) fails, the
simple rule is the result and P2 is optional.

Status (1000 seeds): (1) holds. At rho = 1, use 0.5, T = 30, NP is +23% while assisted and
-2% after a shift with the AI off; SBP is +18% and +4%. The ranking flips in 14 settings at
Bonferroni p < 4.4e-5, all NP ahead of an SBP-type assistant during use and behind it after.
The shift penalty tracks diversity lost by the shift (r = 0.46 over 162 settings). (2) for the
SBP rules: nothing to fix, SBP barely reduces diversity. Hence P1b.

### P1b. Copy-based rules (done)

P1 put the effect in the assistants that copy whole modules (NP, TDR), so we ran the simple
rules there: TDR with a wider pool (top 25%), TDR with inverse-popularity sampling, and the
paper's personalised chunk-copy (PCC). The two TDR variants help a little (1-2 points after the
shift). PCC is best on both axes but keeps the least of what it gives at rho = 1. The split that
matters is a whole copied solution versus a one-step nudge. See D-010.

### P2. Train the selector (not started, gated on P1; retargeted in D-011)

D-011 replaces the design below: the action is full answer (PCC), nudge (SBP) or step back,
and the learner has to beat fixed mixes of PCC and SBP. The rewards are unchanged.

A selector that picks among SBP's candidate flips using a few features (fitness gain, gain
rank, how common the agent's current choice is on that decision, round). Softmax over a linear
score, fit with an evolution strategy. Three rewards, all computed by the simulator:

1. R_now: mean fitness while assisted. This is the per-interaction reward an assistant gets today.
2. R_long: R_now plus fitness over the next 30 rounds with the AI still on, same problem.
3. R_adapt: R_now plus lambda times fitness after a shift with the AI off.

Train on landscape seeds 0-799, test on 800-999, and on a different shift size (50%) and use
rates not seen in training. Compare at matched R_now.

Gate: R_adapt beats R_long and the best simple rule on held-out landscapes at matched R_now.
If R_long is as good, the problem is the evaluation horizon, not the objective, and that is
the finding. If nothing beats the simple rule, the simple rule is the finding.

### P3. Outside the simulator (deferred, GPU)

Two options, pick one after P2:

- (a) LLM assistant. Render the task as text and GRPO-train Qwen3.5-9B on mira/vega with
  R_now versus R_adapt rewards from the simulator. Tests whether the reward recipe changes an
  LLM policy the same way it changes the small selector.
- (b) People. A browser version of the task: 10 unaided trials, 30 trials with an assigned
  assistant, 20 unaided trials on a shifted landscape. Communities are the unit of analysis.
  This follows the design of Kumar et al. 2024 (arXiv:2410.03703), which compared assisted and
  later unassisted performance on creativity tasks.

My lean is (b). It is the stronger evidence that the effect is about people and not about the
simulator. (a) is cheaper and fits the RL-rewards angle.

## What the results could look like

These are hypothetical shapes, not results.

1. Dissociation. The paper's non-personalised designs and some personalised ones look helpful
   while in use and leave the community worse after a shift. The gap grows with exposure up
   to the point where everyone has converged, then stays flat. This is the "it gets worse
   over time" result, measured instead of assumed.
2. Mechanism. Across all assistants and settings, the shift penalty tracks how much diversity
   the assistant removed by the time of the shift. One line, many points.
3. Fix. A selector trained on R_adapt sits at about the same x as the best assistant and higher
   on y. The learned weights say when it pushes agents apart and when it lets them converge.
4. Nulls that would still be useful. Randomising among good suggestions closes the gap (then
   use that). R_long does as well as R_adapt (then the issue is the evaluation horizon). The
   learned selector does not transfer to new landscapes (then it learned the simulator).

## Why it could matter

Assistants are evaluated and trained on how much they help in the interaction. If two
assistants look equally helpful in the interaction but leave groups differently able to handle
the next problem, the standard evaluation is missing something that grows with adoption. The
sim shows the mechanism in a published model with nothing built in. The trained selector shows
the fix is a choice of reward. The human or LLM stage shows whether either survives outside the
simulator.

"Fast convergence hurts long-run exploration" alone is not new (Lazer & Friedman 2007). The new
part has to be the evaluation (what the community can do after the assistant) and the reward
that optimises it.

## Related work we use

- Fazelpour, O'Brien & Rubin 2026, arXiv:2608.19390. The model.
- O'Brien, Fazelpour, Rubin & Wong 2025, CogSci. Earlier version, scale-free networks, K = 5.
- Lazer & Friedman 2007, ASQ. Network structure and exploration. The baseline effect we have
  to go beyond.
- Kumar et al. 2024, arXiv:2410.03703. Assisted versus later unassisted performance in people.
- Carroll et al. 2022, ICML. Long-horizon objectives can change users. Relevant if R_long
  learns to manipulate the population.
- Chaney, Stewart & Engelhardt 2018, arXiv:1710.11214. Recommender feedback loops. For a later
  archive or retraining loop.
- Mouret & Clune 2015, arXiv:1504.04909. MAP-Elites. Background for the diversity rules.

Other papers from the brainstorm (empowerment-based assistance, the refinement paradox,
T-RECS, RecSim NG) come back only if a later stage needs them. Check every citation before it
goes into a paper.
