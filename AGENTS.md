# Repository Guidelines

Prioritize clarity over action. do not be quick to jump to an action until you’ve achieved clarity, always confirm ambiguous items with me before continuing.

- ALWAYS RUN YOUR TESTS IF YOU CREATE ONE (use `conda run -n seq pytest …`).
- Do not run the whole `pytest -q` unless ordered to do so. Prefer focused invocations such as `conda run -n seq pytest test/test_cli_register.py -k smoke`.
- USE THE `seq` CONDA ENVIRONMENT TO RUN ALL PYTHON COMMANDS INCLUDING PYTEST. Use `conda run -n seq …`; if the environment is missing, use the `cp4` environment.
- Sandbox note: the CLI runs under a seccomp profile; `conda run` can hang if GPU plugins try to register semaphores. When that happens, set `CONDA_NO_PLUGINS=true` or ping the user to loosen sandbox restrictions before proceeding.
- You do not need to verify `git` status after your edits. There can be changes that _I_ made that I want you to keep, but you may still inspect `git status` to confirm what you touched.
- DO NOT create conditional imports or assume that some packages are not going to be available. ALL packages are available, do not try to create a fallback unless explicitly told to do so. It adds bloat and complexity.
- ABSOLUTELY NEVER add fallbacks (silent defaults, alternate codepaths, “best-effort” behavior) unless the user explicitly requests it. Fall-backs hide bugs and waste debugging time.
- “Robustness” must be contract-preserving: do not change output semantics (including edge/error cases) unless the user explicitly asks. Prefer failing loudly over silently substituting defaults.
- DO NOT preemptively handle exceptions, swallowing Exceptions are never acceptable. If you are not sure what to do, ask the user.
- Do not worry about backwards compatibility unless the user indicates so.
- After Python modifications, run `conda run -n seq ruff check --output-format=concise {MODIFIED FILES}` unless told otherwise to check for errors before returning to the user.

- When working with .h5ad files, DO NOT use h5py directly, it can corrupt the h5ad file. Only use anndata or scanpy.
- **DO NOT EVER use %-style formatting for logging. Use f-strings.**

## Quick Start

- Explore CLIs: `fishtools --help` (main entrypoint), `preprocess --help` (image prep), `postprocess --help` (analysis assembly), `segment --help` (Cellpose wrapper), `mkprobes --help` (probe design).

## CLI Structure (Entry Points, Groups, Modules)

- Entry points (pyproject `[project.scripts]`):
  - `fishtools` → `fishtools/cli.py:main` (top‑level tools: `compress`, `decompress`, and nests `postprocess`).
  - `preprocess` → `fishtools/preprocess/cli.py:main` (all preprocessing sub‑apps; lazy‑loaded).
  - `postprocess` → `fishtools/postprocess/cli.py:main` (post‑processing sub‑apps; lazy‑loaded).
  - `segment` → `fishtools/segment:main`.
  - `mkprobes` → `fishtools/mkprobes/cli:main`.

- Aggregators use a lazy‑loading Click group to keep startup fast:
  - `fishtools/preprocess/cli.py` defines `LazyGroup` and a `LAZY_COMMANDS` map.

- Logging, progress, and errors (uniform across CLIs):
  - Always initialize workspace‑scoped logging via `setup_cli_logging` (or `_setup_cli_logging` helpers) so logs write under `<workspace>/analysis/logs` and play nicely with progress bars.

- Adding a new CLI or group (checklist):
  1) Create `cli_<feature>.py` with a Click group and subcommands.
  2) Add to the aggregator’s `LAZY_COMMANDS` (`preprocess` or `postprocess`).
  3) Initialize logging via the shared helpers; accept `<workspace>` first; use `Workspace` APIs for IO.
  4) Provide focused tests in `test/test_<feature>_cli.py` using Click/Typer runners.
  5) Keep help text clear; prefer kebab‑case for option names; pair booleans as `--foo/--no-foo`.

## Core Architecture (what to extend)

- Paths & Workspace: `Workspace` (see `fishtools/io/workspace.py`) manages experiment structure (round/ROI/codebook). Always use `Workspace` functions first instead of manual globbing or string-splitting. Hyphens (`-`) are not allowed in round names (we use `-` as a separator elsewhere); use underscores in round tokens.

### `Workspace` First Policy

- Do not construct paths by hand for common operations. Prefer the methods below for correctness and consistency across CLIs and tests.
- Do not parse ROI/codebook from directory names yourself; use `Workspace.rois` and `Workspace.resolve_rois`.
- For stitching, remember: TileConfiguration is ROI-level; stitched outputs are ROI+codebook-level.

## Testing Practices (strict; see TESTING_STRAT.md)

- Tests must pass. Favor TDD: analyze behavior, write focused tests, then implement. Example: `conda run -n seq pytest test/test_preprocess_config_json.py::test_round_defaults`.
- Prefer synthetic arrays; validate shapes/dtypes; use `np.allclose` for floats and `np.isfinite` checks.
- Add CLI tests for new subcommands; keep coverage stable or rising. Example: `conda run -n seq pytest -vv -k register` using the Click/Typer fixtures in `test/conftest.py`.
- Mock external FS/heavy IO only; do not mock core algorithms.
- **Refactor-resistant assertions:** verify observable outcomes/state, not internal steps; duplicate literals in tests instead of reusing production constants when asserting.
- **Control nondeterminism:** inject time, randomness, and environment explicitly (e.g., pass seeded RNGs, fixed timestamps).
- **Output-first mindset:** default to output- or state-based assertions; reserve communication-based expectations for unmanaged edges through adapters.
- Do not manually parse ROI names — always use `Workspace` or `batch_roi`.
- Keep tests deterministic; seed RNGs (e.g., `rng = np.random.default_rng(0)`); avoid relying on local file structures beyond fixtures/workspace mocks.
