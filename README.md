# FTEC5660 Project 2 - LiTS / BlocksWorld reproducibility

**WEI ZIJUN, 1155244573** | Report prepared 5 November; revised 9 November 2026.

This is a student partial reproduction using the authors' LiTS repository and an accessible replacement model. It is not the authors' repository and does not reproduce their exact 66.7% BlocksWorld number. The main controlled change is ancestor-cycle pruning on a shared engineering-fixed MCTS control. A later low-score-widening pilot is an exploratory negative result.

## Main findings

| Round (same 20 tasks) | Control success | Cycle-pruned success | Calls, control -> pruned |
|---|---:|---:|---:|
| 1 | 7/20 | 8/20 | 1137 -> 721 |
| 2 | 7/20 | 9/20 | 1147 -> 718 |
| 3 | 10/20 | 8/20 | 1049 -> 737 |

Across three rounds: calls fall 34.7%, returned tokens 33.9%, and estimated fees 34.1%. Success is 24/60 vs. 25/60 repeated task-runs, with inconsistent accuracy differences. These are the same 20 tasks, not 60 independent test samples. The control is `UniquePolicy + FrontierMCTS`, not untouched author MCTS. In the two-task low-score pilot both arms solve 1/2; the two added actions are not explored, so no effectiveness gain is demonstrated.

## Report and source

- Formal report: `output/pdf/LiTS_Reproducibility_Report.pdf` (8 pages).
- Primary evidence: `reports/cycle_stability_20261104.json` and three `results/cycle_test_*` directories.
- New pilot: `reports/low_score_dev_20261104.json` and `results/low_score_dev_20261104`.
- Failure audit: `reports/cycle_round3_loss_analysis_20261104.json`.
- Author repository: https://github.com/xinzhel/lits-llm
- Upstream revision: `1ef347bee7b3f2c10dc504342589b8c951794c2d` (LiTS 0.4.0).
- Paper reference: https://arxiv.org/html/2603.00631v2 (Table 3: different model, 30 samples, 10 iterations).

## Installation (no model API calls)

Recommended measured environment: Python 3.12.14, macOS arm64; runtime packages are pinned. A fresh-machine rebuild has not been verified. Other platforms may require compatible binary wheels. Upstream optional backends are outside this scoped installation.

```bash
PYTHON_BIN=python3.12 bash submission_setup.sh
```

This downloads dependencies and the pinned upstream checkout, installs it with `--no-deps`, checks existing data hashes and performs network-disabled regressions. Bootstrap build tools `setuptools`/`wheel` are installed separately; they are not all covered by the runtime lock. The supplied split/manifest is preserved. The original `scripts/setup.sh` is retained byte-for-byte as historical source; use the submission setup above for this package.

The upstream directory is a fetched dependency, ignored by this student repository. It is not an empty Git submodule. The primary search pipeline uses LLM transitions, followed by independent four-operator replay after search. The separately included local symbolic planner was a diagnostic only; its successes are not in the main experiment.

## Recompute existing reports offline

After installation, these use supplied logs and do not call a model:

```bash
.venv/bin/python scripts/cycle_repeat_report.py
.venv/bin/python scripts/analyze_cycle_losses.py
.venv/bin/python scripts/low_score_report.py results/low_score_dev_20261104
```

The three-round validator checks every planned task, local seed, data and source fingerprint. Selected full trees, terminal nodes and actual API ledgers are supplied rather than just summary CSVs. Incomplete tasks would remain failures in the denominator. On some restricted macOS sandboxes, PyTorch/OpenMP cannot access shared memory; run the same offline command in a normal local terminal. This does not require a paid rerun.

## Visual replay for the presentation (no model calls)

A self-contained viewer shows three measured round-three trajectories: a successful pruned run on instance-35, the successful control run on instance-69, and an audited closed-cycle path from its pruned run. Use Previous/Next or Play to inspect the blocks, actions, goals and logged node IDs.

```bash
python3 submission/build_replay_demo.py
# macOS: open the resulting local file
open output/replay/lits-replay.html
```

Building this viewer needs only Python's standard library and the supplied artifacts. Every displayed action is independently replayed and compared with its logged modeled state. A displayed cycle must match the saved audit and an ancestor on that path; successful paths match the selected terminal actions. Source paths and SHA256 hashes are included. The historical whole-task request counts are 23, 52 and 38; playback starts no search or API request. These selected paths illustrate behavior; the main three-round experiment remains the basis for accuracy and cost conclusions.

The timed English recording script now uses a two-minute visual replay and targets a 13-minute presentation. Integrity checks and result-summary commands remain available as pre-recording checks or backup evidence.

## Offline checks and CLI dry runs

```bash
.venv/bin/python scripts/check_api_guard.py
.venv/bin/python scripts/check_search_fixes.py
.venv/bin/python scripts/check_cycle_pruning.py
.venv/bin/python scripts/check_low_score.py
.venv/bin/python scripts/run.py --config cycle_pruned_3 --split dev --limit 1 --output reports/new_dry_cycle
.venv/bin/python scripts/low_score_experiment.py --output reports/new_dry_low_score
```

Output paths must be fresh. The default runner mode disables networking and does not load `.env`. `preflight.py` additionally validates the dataset and independent evaluator using offline fixtures; fixtures are not measured model results.

## Deliberate paid execution

Do not add `--execute-api` unless you intend to pay for a new run. No command above needs credentials.

Copy `.env.example` to a local, ignored `.env`, then supply `OPENAI_API_KEY`, `OPENAI_API_BASE=https://api.deepseek.com` and `LITS_MODEL=openai/deepseek-flash`. Never commit the filled file.

```bash
# One explicit development task using the frozen cycle configuration:
.venv/bin/python scripts/run.py --config cycle_pruned_3 --split dev --limit 1 --output results/new_dev_cycle --execute-api
# Both development tasks, original-cycle versus low-score extension:
.venv/bin/python scripts/low_score_experiment.py --output results/new_dev_low_score --execute-api
```

The primary task limits are 150 API starts, a cumulative returned-token threshold of 150000 checked before each next request, 512 output-token ceiling and 900-second process timeout. Thinking and SDK retries are disabled. The final response may cross the cumulative threshold. No provider seed is sent; local seed 42 does not guarantee API determinism. Fees are estimates using cache usage at the 4 November snapshot, not confirmed bills.

## Code organization and attribution

- `lits_fixes.py`: common finite-action/frontier fixes and the isolated `CycleMCTS` treatment.
- `cycle_state.py`: conservative state canonicalization; uncertain states are not pruned.
- `lits_low_score.py`: separate bounded low-score singleton widening; no backup or recovery change.
- `entry.py`: offline network guard and bounded API ledger.
- `symbolic.py` / `evaluate.py`: independent plan replay and all-planned-task accounting.
- `cycle_experiment.py`, `cycle_repeat_report.py`: frozen main trial and repeated analysis.
- `low_score_experiment.py`, `low_score_report.py`: separate two-development-task pilot.

Frozen experiment code/configuration must not be edited in place: validators intentionally reject changed hashes. Create a new version/protocol for further research. Test cases already inspected for debugging are regression cases for future modifications, not an untouched holdout.

## Distribution and provenance

This is a curated submission copy. `.env`, `.venv`, caches, the upstream nested `.git`, bulk duplicate traces and nonessential checkpoints are omitted. Original personal absolute path labels in selected report/result metadata are normalized to `<PROJECT_ROOT>` / `<COURSE_ROOT>`; they are not runnable commands. Runtime source/config/data, numeric usage, actions and states are preserved. `EVIDENCE_INVENTORY.json` records distributed hashes and normalized files. `verify_distribution.py` checks those hashes and refuses obvious credential patterns.

The initial student-package commits were preparation commits made on 5 November 2026, not a reconstruction of the earlier experiment timeline. Upstream authorship and Apache-2.0 license are documented in `THIRD_PARTY_NOTICES.txt` and `third_party/`. The upstream repository remains unchanged. Empirical claims are backed by measured artifacts.

## Submission items

Student submission repository: **https://github.com/Wayne-12321/FTEC5660-Project2-LiTS**. The repository contains the prepared source, selected evidence, formal report and readable preparation commits.

The assignment requires the report, this student repository link, and a recorded Zoom presentation link of at most 15 minutes. A timed presentation script is included under `submission/`; the actual recording and course-portal submission remain to be completed.
