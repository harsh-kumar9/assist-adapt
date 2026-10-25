# Reward models in this repo

Every reward we have trained or simulated, written from the code. Last updated 2026-10-24.

## Notation

- Team $i$ holds approach $x_i \in \{0,1\}^{N}$, $N = 20$. The assistant can change only the first
  $M = 10$ bits, the module $m_i$; the rest $r_i$ is fixed for the assistant. $x_i = (m_i, r_i)$.
- Fitness $F(x) \in [0, 1]$: the NK payoff, normalised by its maximum and raised to the 8th power.
- The assistant answers with a module $a \in \{0,1\}^M$, or $\varnothing$ if the text has no ten-digit string.
- Gain: $g(a) = 10 \cdot \max\{0,\ F(a, r_i) - F(x_i)\}$. Helped: $h(a) = \mathbb{1}[g(a) > 0]$.
- $d(\cdot,\cdot)$ is Hamming distance. $C_t = \{m_j(t)\}$ is the set of modules held in the culture at round $t$.
- A team adopts an answer only if it helps: $x_i \leftarrow (a, r_i)$ if $F(a, r_i) > F(x_i)$.

## P2: which kind of help, learned by evolution strategy (src/p2.py)

The policy picks full answer (PCC), nudge (SBP) or step back for each request; $\theta$ is fitted by an
evolution strategy on one episode reward. With $\bar F_t$ the community's mean fitness in round $t$,
$T = 30$ assisted rounds and 30 rounds after:

$$R_{\text{now}} = \tfrac{1}{T}\textstyle\sum_{t \le T} \bar F_t$$

$$R_{\text{long}} = R_{\text{now}} + \tfrac{1}{30}\textstyle\sum_{t=T+1}^{T+30} \bar F_t \quad \text{(same problem, assistant on)}$$

$$R_{\text{adapt}} = R_{\text{now}} + \tfrac{1}{30}\textstyle\sum_{t=T+1}^{T+30} \bar F'_t \quad \text{(problem shifted, assistant off)}$$

Result: all three learn the same policy (D-013).

## P3: the LLM trained online, every round (src/llm/online.py)

Per answer $a$ to asker $i$ (the reward is per sample; $G = 8$ samples per request):

| Name | Reward | Flag |
|---|---|---|
| $R_{\text{now}}$ | $g(a)$; invalid answer $-0.1$ | `--reward now` |
| nudge prior | $g(a)\,\bigl(1 - c\, d(a, m_i)/M\bigr)$, $c = 0.8$ (not run) | `--reward edit` |
| step back | if $a = m_i$: the gain the team gets exploring on its own that round | `--step-back` |
| thumbs | $h(a)$ | `--reward thumbs` |
| pick one of two | samples paired $(2k, 2k{+}1)$: $1$ for the one with larger $g$, $0$ for the other; tie at random (G-5: a valid answer beats an invalid one) | `--reward pairwise [--prefer-valid]` |

## P4b: the LLM retrained once per epoch on its users' requests (src/llm/culture_llm.py)

Within an epoch the model is deployed frozen; at the end, 256 logged requests get $G = 8$ new samples
each, scored by:

| Name | Reward | Reading |
|---|---|---|
| thumbs | $h(a)$ | did it help me |
| P-creativity | $h(a)\,\bigl(1 + \lambda\, d(a, m_i)/M\bigr)$, $\lambda = 1$ | helpful and new to me |
| H-creativity | $h(a)\,\bigl(1 + \lambda\, \min_{c \in C_t} d(a, c)/M\bigr)$, $\lambda = 1$ | helpful and new to the culture |
| local | $h(a)\,\mathbb{1}[d(a, m_i) \le k]$, $k = 3$ | helpful without replacing my approach |

## The update (all LLM arms)

Group-relative advantage over the $G$ samples for request $i$, then one policy-gradient step per chunk:

$$A_{ij} = \frac{R_{ij} - \operatorname{mean}_j R_{ij}}{\operatorname{std}_j R_{ij} + 10^{-4}}$$

$$\mathcal{L}(\theta) = -\frac{1}{n_{\text{tok}}} \sum_{i,j} \sum_{t} \Bigl[ A_{ij} \log \pi_\theta(y_{ijt} \mid \cdot) - \beta\, \bigl(e^{\Delta_{ijt}} - \Delta_{ijt} - 1\bigr) \Bigr], \quad \Delta_{ijt} = \log \pi_{\text{ref}}(y_{ijt} \mid \cdot) - \log \pi_\theta(y_{ijt} \mid \cdot)$$

$\beta = 0$ except the "+ KL" arms ($\beta = 0.05$). The KL term is the k3 estimator against the untrained
model. If all $G$ rewards for a request are equal, $A_{ij} = 0$ and the request teaches nothing.
"Explore" arms draw the update samples at temperature 2 (deployment stays at 1).

## Oracle assistants (src/sim.py, src/culture.py)

Not learned; they show what a perfect learner of a reward would do. For asker $i$, let
$H_i = \{a : F(a, r_i) > F(x_i)\}$ be the helpful modules. PCC and SBP always answer (an answer
that does not help is rejected and the team loses the round); the others step back when nothing in
their range helps (the team explores on its own that round).

| Name | Answer | Ideal of |
|---|---|---|
| PCC | $\arg\max_{a \in C_t} F(a, r_i)$ | best approach already in the culture |
| SBP | $\arg\max_{a:\, d(a, m_i) = 1} F(a, r_i)$ | best single change |
| gated | PCC or SBP, whichever gains more, if either helps; else step back | P2's learned rule |
| rhelp | uniform over $H_i$ | thumbs |
| maxp | $\arg\max_{a \in H_i} d(a, m_i)$ | P-creativity |
| maxh | $\arg\max_{a \in H_i} \min_{c \in C_t} d(a, c)$ | H-creativity |
| minp | $\arg\min_{a \in H_i} d(a, m_i)$ | closest helpful change |
| rad$k$ | $\arg\max_{a \in H_i,\ d(a, m_i) \le k} F(a, r_i)$ | local, with reach $k$ |
| rnd$k$ | $m_i \oplus e$, $e$ a uniformly random set of $j$ choices, $j \sim U\{1..k\}$ | chance for the edit interface |
| rndfull | uniform over $\{0,1\}^{10}$ | chance for the full answer |

Ties in maxp / maxh / minp go to the larger gain. rnd$k$ and rndfull are blind to the landscape and
always answer, as the LLM does; the team keeps the answer only if it helps.
