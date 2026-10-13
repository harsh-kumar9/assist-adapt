# assist-adapt

Can an AI assistant make a community better at today's problem while leaving it worse at the
next one? We test this in the NK model of Fazelpour, O'Brien & Rubin (2026,
arXiv:2608.19390). After T rounds with an assistant, the problem changes and the assistant goes
away. We measure how the community does next.

- `docs/PLAN.md`: the question, what we dropped, the pilots and their gates.
- `docs/DECISIONS.md`: every modelling choice, with its status.

## Layout

```
src/nk.py        landscapes (full 2^20 fitness table per landscape)
src/sim.py       community, assistants, dynamics, probes
src/run.py       sweep runner (one unit = one rho x seed)
src/analyze.py   paired comparisons vs no AI, figures, summary CSVs
scripts/ada_sweep.sh   CPU sweeps on ada (P0, P1)
tests/test_sim.py
figures/         PDF + PNG
results/summary/ CSVs behind every figure
```

## Run

Tests:

```bash
python tests/test_sim.py
```

Sweeps run on ada (CPU only; mira and vega are for GPU jobs):

```bash
rsync -a --exclude results/ --exclude logs/ --exclude .git /Users/harsh/Desktop/assist-adapt/ ada:/ada1/u/harsh/assist-adapt/
```

```bash
ssh ada 'nohup bash /ada1/u/harsh/assist-adapt/scripts/ada_sweep.sh p1 16 > /ada1/u/harsh/assist-adapt/logs/p1.log 2>&1 &'
```

Pull results and make figures:

```bash
rsync -a ada:/ada1/u/harsh/assist-adapt/results/p0 ada:/ada1/u/harsh/assist-adapt/results/p1 /Users/harsh/Desktop/assist-adapt/results/
```

```bash
python -m src.analyze p0 results/p0 && python -m src.analyze p1 results/p1
```
