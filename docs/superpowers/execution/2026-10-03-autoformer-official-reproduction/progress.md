# SDD ledger — plan: docs/superpowers/plans/2026-10-03-autoformer-official-reproduction.md

Execution mode: Native (user approved 2026-10-03).

Ruling: The project root is not a Git repository, so the required worktree helper cannot create its Git-ignored execution workspace. Work proceeds in the user-specified project directory with this ledger under `docs/superpowers/execution/`; cost if wrong: changes cannot be isolated or committed as a branch, but the project has no branch to protect.

Pre-flight: Task 1 produces a verified immutable upstream checkout consumed by Task 2; interface is compatible because both specify `upstream/Autoformer` and `provenance/upstream.json`.

Pre-flight: Task 2 produces formal run arrays consumed by Task 4; interface is compatible because the formal wrapper writes `runs/<run-id>/results/*/{pred,true}.npy`.

Pre-flight: Task 3 produces `evaluate_models(...)` consumed by Task 4; interface is compatible because the comparison CLI accepts official normalized arrays and existing baseline original-scale `.npz` files.

Pre-flight: Task 4 produces comparison JSON/Markdown/PNG consumed by Task 5; interface is compatible because Task 5 only documents and verifies these artifacts.

Task 1: complete (no project Git commit; tests: `python -m pytest reproduction/02_autoformer_official/tests/test_source_boundary.py -q` → 3 passed; source commit `51c7d416ae120b805fd5beef2f4ccf7de496a6ff`; CSV SHA-256 recorded).

Task 2: Ruling: the required one-epoch smoke run was interrupted after about eight minutes without reaching the authors' first 100-batch progress report. The immutable author command had loaded ETTm1 and entered training successfully, but the current CPU implies a many-hour ten-epoch run; cost if wrong: a slow but eventually complete local training was forfeited in favor of avoiding an estimated 10+ hour laptop workload. `runs/ettm1_l96_h96_smoke_seed2021/ABORTED.md` preserves the evidence and excludes it from metrics.

Task 3: complete (tests: `PYTHONPATH=reproduction/02_autoformer_official/src reproduction/02_autoformer_official/.venv/bin/python -m pytest reproduction/02_autoformer_official/tests/test_evaluation.py -q` → 3 passed; evaluator converts scales, rejects shifted labels, and validates result discovery).

Task 4 (infrastructure): complete pending formal remote run (tests: `PYTHONPATH=reproduction/02_autoformer_official/src reproduction/02_autoformer_official/.venv/bin/python -m pytest reproduction/02_autoformer_official/tests/test_comparison_cli.py -q` → 2 passed; comparison artifact includes every model's normalized/original-scale MSE and rejects misaligned labels before writing output).
