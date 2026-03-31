# Tadabur ASR — Team Roadmap

This repo is our shared workspace for the Tadabur ASR machine learning project: data → baselines → improved models → evaluation → paper + presentation. This `README.md` is written for us (what we do next, who owns what, and what “done” means).

## Working agreements (how we operate)

- **Meetings**: 1 planning meeting/week + 1 short async check-in midweek.
- **PRs**: small PRs, descriptive titles, one reviewer minimum when possible.
- **Reproducibility rule**: if it can’t be rerun, it doesn’t count as a result.
- **Single source of truth**: metrics/results that go into the paper must be produced by code in this repo.



## Roadmap (sprints)

### Sprint 0 — Repo + data setup

- **Goal**: everyone can run the same baseline end-to-end.
- **Done when**:
  - dataset access instructions are clear (download link, folder path, expected files)
  - a single command/script runs preprocessing + a tiny sanity training run
  - we have a shared metrics file or table format (even if numbers are bad)

### Sprint 1 — Problem statement + baselines

- **Goal**: lock the task definition and get baseline numbers.
- **Done when**:
  - target label/metric is finalized (what we optimize + what we report)
  - at least **2 baselines** run reliably (simple + standard)
  - results are logged in a consistent way (seed, split, metric, config)

### Sprint 2 — EDA + preprocessing + split policy 

- **Goal**: justify data choices and remove obvious pipeline risks.
- **Done when**:
  - preprocessing is documented and implemented (handling missing/invalid entries)
  - EDA produces 3–6 figures/tables we can use in the paper
  - train/val/test (or CV) protocol is chosen and implemented once (no ad‑hoc splits)

### Sprint 3 — Feature engineering + model improvements

- **Goal**: improve over baselines with at least one clear idea.
- **Done when**:
  - 2–4 model variants are implemented behind a common interface/config
  - one ablation table exists (what helped, what didn’t)
  - hyperparameter tuning approach is documented (even if minimal)

### Sprint 4 — Final experiments + error analysis

- **Goal**: produce publication-quality results and insights.
- **Done when**:
  - final models run with fixed seeds and saved configs
  - results tables/figures are generated from scripts (not manual edits)
  - error analysis is written (common failure cases, bias/imbalance notes, limitations)

### Sprint 5 — Paper + presentation packaging 

- **Goal**: a clean repo and a coherent story.
- **Done when**:
  - IEEE paper draft is complete, consistent with repo outputs, and cited properly
  - final repo has “how to run” steps and points to the exact commands used
  - 5-minute slides are ready + one rehearsal completed

