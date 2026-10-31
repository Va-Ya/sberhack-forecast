# SberHack handoff

## Fresh audit, 2026-10-31 17:49 Moscow

See reports/current_audit/SUMMARY.md and its evidence JSON/XML/CSV files.
- 27 unit tests passed; 27 real HTTP checks passed. Full real POST: 23.00/22.60 seconds.
- Source sales/stock independently rebuilt from ZIP: exact cache equality, difference 0.0.
- All three frozen backtest predictions and independently computed metrics matched the previous report.
- Final model refitted separately: native file SHA256 identical, prediction difference 0.0, fit 78.12 seconds.
- Important quality issue: control-window total prediction 165210.33 versus actual 212336, bias -22.19%.
  Average per-series absolute Bias is a different measure: 67.58%.
- Mean8 baseline beats the chosen model on RMSE/Bias/RMSLE; chosen model wins on the weighted local objective.
- Short 180-day control history is slightly worse on WAPE: 47.4211 vs 47.3994 for full history.
- CLI defect fixed: explicit missing/directory --pairs now fails clearly instead of falling back or crashing.
  Two regressions reproduced the issue before the fix and passed afterward. Main model/submission unchanged.
- Docker freshly started once and failed before engine startup on inaccessible dockerInference runtime socket.
  The old WSL error below belongs to an earlier attempt. No OS/service reset performed in this audit.
- Current runtime is an existing venv with system-site-packages; clean Linux/container operation remains NOT RUN.
- The v1.0.0 release predates this audit; main contains the current CLI fix and audit report.

Status: final model, submission, CLI and running Windows HTTP API are ready and verified.
Dockerfile is ready; Docker build/offline execution remains blocked by WSL startup and administrator permissions.

Goal: submission + reproducible model + FastAPI/Docker by 2026-10-31 23:30 Europe/Moscow.
Data: hakaton.zip (unchanged), two CSVs. Competition cutoff 2026-07-26.
Pair policy: temporary full warehouse/article Cartesian product (11 × 799).
Skills: four ECC skills installed from Shelpess/komanda v2 c8dfa617a2ef1523c713ba79e88189d93b0f1173.

Environment: Python 3.13.9; pandas 2.3.3, numpy 2.3.4, LightGBM 4.7.0.
API/pytest/ruff dependencies installed in .venv (system-site-packages enabled).
Hardware: 4 logical CPUs, 7.84 GiB RAM, initially only 0.5–0.9 GiB available.
Training reduced to 600 sampled pairs/origin, 180 trees/model, 2 threads.
Final tests outside sandbox: 25 passed in 27.55 seconds; ruff passed.
Coverage: app/main 100%, model 99%, features 99%, data 87%, overall 73% (CLI is tested separately).
Repeat training of the synthetic native model produced identical model text.

Docker Desktop 4.49 initially crashed on stale AF_UNIX sockets. Both runtime folders were preserved
under *.sberhack-backup-* names (latest suffix 20261031021020). No images, containers, volumes,
VHDX or settings were reset. Startup now fails at WSL VM creation with HCS_E_CONNECTION_TIMEOUT.
Restart-Service WslService was denied by Windows; no active WSL VM was found at the check.
See reports/environment.md and tools/restart-wsl-admin.ps1. Docker build/offline smoke is NOT verified.
Linux/Python 3.13 pinned wheel dependency resolution passed (25 runtime packages).

Data findings: three missing sales_sum values, 316955 missing discount_sum values.
Money is not used in features. Negative stock is allowed. Exact post-cutoff counts in reports/data_audit.json.

Completed training command:
`.venv/Scripts/python -u -m forecast.cli train --archive hakaton.zip --iterations 180 --pairs-per-origin 600 --threads 2`
Training took 1377.4 seconds; final fit: 420600 rows, 60 features.
Selected log_zero (LightGBM MSE on log1p, predictions below 0.5 set to zero), locked before the control fold.
Control fold 2026-05-25: SMAPE 64.6634, WAPE 47.3994, RMSE 3.3456, Bias 67.5781, RMSLE 0.4875.
Baseline last: SMAPE 74.3701, WAPE 58.4249, RMSE 3.9720, Bias 120.6910, RMSLE 0.6173.
Training can resume from completed reports/folds bundles with the SAME configuration.
data/cache/panel.npz avoids rereading inputs. Full history is not included in the release archive.
Baseline files: outputs/baseline_{last,mean4,mean8,seasonal}.csv (79101 rows each).
Do not use the earlier 1800-pair run; it was stopped after preprocessing to reduce memory.

Final CSV: outputs/submission.csv, 79101 rows, weeks 2026-07-27 through 2026-09-21.
SHA256: 740b36cc1bff8b7b7fecb204fccbdcf9dad1bf561add42415bf33a9290f23d6f
Verified complete keys, finite nonnegative values, no duplicate keys, exact native reload determinism.
Model bundle: model/config.json, model/log.txt, model/pairs.csv.
Local Git repository initialized; implementation commit 185b9ec. Main has been pushed to the private repository.
GitHub publication authorized by the user: https://github.com/Va-Ya/sberhack-forecast (private).
Human guide: docs/HUMAN_GUIDE.md. Agent guide: docs/AGENT_GUIDE.md. Copy-paste prompt: CONTINUE_PROMPT.md.
The private repository is a separate project from Shelpess/komanda; the latter remains the skill source.
Delivery archive: release/sberhack-solution.zip; checksums/source commit in release/manifest.json.
GitHub release: https://github.com/Va-Ya/sberhack-forecast/releases/tag/v1.0.0

Live server: http://127.0.0.1:8000 ; docs: /docs ; health: /health.
Started via hidden user process; launcher PID 14460, uvicorn PID 9220 (check reports/api_process.json/logs).
Cache currently contains the successful 180-day smoke forecast, which is different from full-history submission.
CLI/API TestClient parity on real 180-day files: 79101 rows, 28.90 seconds PASS.
Real HTTP parity including multipart spool and cached GET: exact bytes equal, 57.34 seconds PASS.
Smoke inputs/expected forecast: data/smoke/ (local only; regenerate using tools.smoke_api).

Next required step: restore WSL with administrator help, build Docker, run --network none smoke
using reports/environment.md, then update the verification status and rebuild release archive.
Ready maintenance commands:
`python -m tools.verify_release`
`python -m tools.report_results`
`python -m tools.package_solution`

User asked to warn if account usage exceeds 50%. Last read during publication: Plus Codex 5-hour 45%, weekly 7%.
Check get_usage_limits at milestones; these are shared Codex windows, not total Plus consumption.
Earlier automatic approval review stopped once due to a usage-limit error; after user continuation
ordinary usage was allowed and subsequent approvals/tests completed. No safety denial is pending.
No organizer messages or submission have been sent.
