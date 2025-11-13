# Repository Guidelines

## Orientation

- Purpose: End‑to‑end FISH analysis — compression → preprocessing → registration/stitching → segmentation → post‑processing → probe design → data assembly. Expect gigabyte-scale TIFF stacks per round and plan for long-running jobs.
- fishtools is only designed for Linux; Windows and macOS are not supported.
- Read this, then skim `ARCHITECTURE.md` at the repository root before coding.
- If there is a mismatch between your memory of what the code was and the actual code, re-read the code; a quality hook (e.g., pre-commit formatting) or a human update has made the repo authoritative.

- ALWAYS RUN YOUR TESTS IF YOU CREATE ONE (use `conda run -n seq pytest …`).
- DO NOT run the whole `pytest -q` unless ordered to do so. Prefer focused invocations such as `conda run -n seq pytest test/test_cli_register.py -k smoke`.
- VERY IMPORTANT: USE THE `seq` CONDA ENVIRONMENT TO RUN ALL PYTHON COMMANDS INCLUDING PYTEST. Use `conda run -n seq …`; if the environment is missing, create it with `conda env create -n seq -f environment.yml` and retry.
- Sandbox note: the CLI runs under a seccomp profile; `conda run` can hang if GPU plugins try to register semaphores. When that happens, set `CONDA_NO_PLUGINS=true` or ping the user to loosen sandbox restrictions before proceeding.
- You do not need to verify `git` status after your edits. There can be changes that _I_ made that I want you to keep, but you may still inspect `git status` to confirm what you touched.
- Perform `python -m compileall` before returning the results to the user. Run `conda run -n seq python -m compileall fishtools test` unless told otherwise.

- DO NOT create conditional imports or assume that some packages are not going to be available. ALL packages are available, do not try to create a fallback unless explicitly told to do so. It adds bloat and complexity.
- DO NOT preemptively handle exceptions, swallowing Exceptions are never acceptable. If you are not sure what to do, ask the user.

- After modifications, run `conda run -n seq ruff check --output-format=concise {MODIFIED FILES}` unless told otherwise to check for errors before returning to the user.

- **DO NOT EVER use %-style formatting for logging. Use f-strings.**

```python
# BAD
logger.info("CombSpots grid: rois=%d", n_rois, params.dpi)  # This DOES NOT WORK.
# GOOD
logger.info(f"CombSpots grid: rois={n_rois}, dpi={params.dpi}")
```

## Quick Start

- Alt (mamba): `mamba env create -n fishtools -f environment.yml && conda activate fishtools && pip install -e '.[dev]'` (useful for local development before mirroring the commands inside the `seq` environment).
- Sanity check: `conda run -n seq pytest -q test/`
- Explore CLIs: `fishtools --help` (main entrypoint), `preprocess --help` (image prep), `postprocess --help` (analysis assembly), `segment --help` (Cellpose wrapper), `mkprobes --help` (probe design).

## Project Layout

- `fishtools/`: Core package (Python 3.12). Subpackages: `preprocess/`, `postprocess/`, `mkprobes/`, `segment/`, `compression/`, `utils/`, `gpu/`.
- `test/`: Pytest suite (unit + CLI). Use `test_*.py`; keep tests fast and independent of large TIFFs. Fixtures live in `test/conftest.py` and `test/synthetic_fiducials.py`.
- `scripts/`: Ad‑hoc utilities (e.g., `scripts/register_simple.py`). Avoid importing into library code.
- Data: Keep large binaries out of git; prefer `data/`, `results/`, `figures/` (gitignored) for local artifacts and stash raw datasets outside the repository when possible.

## CLI Structure (Entry Points, Groups, Modules)

- Entry points (pyproject `[project.scripts]`):
  - `fishtools` → `fishtools/cli.py:main` (top‑level tools: `compress`, `decompress`, and nests `postprocess`).
  - `preprocess` → `fishtools/preprocess/cli.py:main` (all preprocessing sub‑apps; lazy‑loaded).
  - `postprocess` → `fishtools/postprocess/cli.py:main` (post‑processing sub‑apps; lazy‑loaded).
  - `segment` → `fishtools/segment:main`.
  - `mkprobes` → `fishtools/mkprobes/cli:main`.

- Aggregators use a lazy‑loading Click group to keep startup fast:
  - `fishtools/preprocess/cli.py` defines `LazyGroup` and a `LAZY_COMMANDS` map:
    - Keys → subcommand name; values → module + exported group function.
    - Registered sub‑apps: `basic`, `deconvnew`, `deconv` (legacy), `register`, `stitch`, `spots`, `inspect`, `verify`, `correct-illum`, `check-shifts`, `check-stitch`.
    - Also exposes a wrapper `registerv1` that forwards to the Typer migration module (`cli_register_migrated`).
  - `fishtools/postprocess/cli.py` mirrors the same pattern with `concat` and `extract-intensity`.
  - `fishtools/cli.py` registers its own commands and attaches the `postprocess` group.

- Subcommand module pattern (how to add/organize CLIs):
  - File per feature: `fishtools/<area>/cli_<feature>.py`.
  - Inside the file, define a Click group named after the sub‑app (e.g., `basic`, `stitch`, `deconvnew`). Attach commands with `@<group>.command()`.
  - Register the group in the aggregator’s `LAZY_COMMANDS` map instead of importing eagerly.
  - For new Typer‑based flows that must live under `preprocess`, provide a thin wrapper command (like `registerv1`) that forwards args to the Typer module. This keeps aggregator behavior consistent while we migrate.

- Logging, progress, and errors (uniform across CLIs):
  - Always initialize workspace‑scoped logging via `setup_cli_logging` (or `_setup_cli_logging` helpers) so logs write under `<workspace>/analysis/logs` and play nicely with progress bars.
  - Use Loguru with f‑strings; never `%` formatting. Fail fast with `click.ClickException` for user‑facing errors.

- Deconvolution CLI structure quick reference:
  - Group: `preprocess deconvnew` (modern; `deconv` is legacy per‑tile quantization kept for compatibility).
  - Commands:
    - `prepare` — sample small tile sets to validate settings and emit float32 artifacts/histograms.
    - `precompute` — aggregate per‑tile histograms to global scaling (`analysis/deconv_scaling/<round>.txt`).
    - `quantize` — write uint16 deliverables using the global scaling (adds fid planes).
    - `run` — multi‑GPU deconvolution over rounds/ROIs. Modes: `u16` (preferred), `float32`, `legacy`.
    - `easy` — convenience wrapper that ensures scaling exists, then runs `quantize` and `run` concurrently.
  - Concurrency model: a process per GPU with a three‑stage pipeline (reader → GPU compute → writer) and a central scheduler. Use `--devices` to select GPUs; `auto` = all visible.
  - Idempotency: each backend advertises expected outputs; the CLI filters already‑satisfied tiles unless `--overwrite` is set.
  - Timing fields printed per tile: `basic` (BaSiC + H2D), `dec` (unsynchronized kernel wrapper; often ~0), `quant` (u16 path), `post` (D2H + hist/host work for float32), and `stage_total` (= GPU stages + write). Use `stage_total − (basic+dec+quant+post)` to approximate write time.

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

### Workspace Quick Reference

- `Workspace(path)` → normalized workspace root (accepts root or `analysis/deconv` subdir).
- `ws.deconved` → `analysis/deconv/` directory.
- Discovery: `ws.rounds: list[str]`, `ws.rois: list[str]`.
- Validation: `ws.resolve_rois(rois) -> list[str]` (fails fast on unknown ROI).
- Images:
  - `ws.img(round, roi, idx, read=False)` → Path or ndarray.
  - `ws.registered(roi, codebook)` → Path to `analysis/deconv/registered--ROI+CB`.
  - `ws.regimg(roi, codebook, idx, read=False)` → Path or ndarray.
- Stitching:
  - `ws.tileconfig_dir(roi)` → `<workspace_root>/stitch--ROI` (ROI-level); contains `TileConfiguration.registered.txt`.
  - `ws.tileconfig(roi)` → load `TileConfiguration` (root first; falls back to `analysis/deconv/stitch--ROI/`).
  - `ws.stitch(roi, codebook=None)` → `analysis/deconv/stitch--ROI[+CB]` (stitched outputs).
- Optimization: `ws.opt(codebook)` → optimization result paths (`mse.txt`, `global_scale.txt`).
- Registered file discovery: `ws.registered_file_map(codebook, rois=None)` → `{roi: [tif...]}, missing_rois`.
- TIFF helpers: `Workspace.ensure_tiff_readable(path)` (raises `CorruptedTiffError`).
- Safe writes: `safe_imwrite(path, array, ...)` writes atomically using a `.partial` temp file.

When in doubt, check or extend `Workspace` instead of scattering path logic.

- Preprocess Data Model: `preprocess/base.py` — `Image` (Pydantic) loads TIFF stacks → `nofid/fid`, enforces 2048², tracks `bits/powers`, optional BaSiC; assumes Z stacks ordered as z/c/y/x and fails fast on mismatches.
- Fiducials & Registration: `preprocess/fiducial.py` — spot detection (DAOStarFinder), phase correlation, robust errors (`NotEnoughSpots`, `ResidualTooLarge`, etc.), `TranslationTransform` for pure shifts. DAOStarFinder parameters flow from `RegisterConfig` in `preprocess/config.py`.
- Configs: `preprocess/config.py` — typed configs (`RegisterConfig`, `StitchingConfig`, `BasicConfig`, `DeconvolutionConfig`, …) with validators. Add new knobs here first and document them in the associated class docstrings.
- Post‑processing: `postprocess/concat_pipeline.py` builds AnnData from spots/polygons/intensities with QC; `postprocess/intensity_pipeline.py` extracts intensities from Zarr masks and writes parquet output to the configured workspace.
- Segmentation: `fishtools/segment/` Typer app; `TrainConfig` + `run_train` wrap Cellpose and save checkpoints to the run directory given in the config.
- Probe Design: `mkprobes/` (`ext/`, `codebook/`, `utils/`) — dataset definitions, codebook picking, IDT integration. Add new codebooks under `mkprobes/codebook/`.
- Compression: `compression/compression.py` — TIFF/JP2/DAX ↔ JPEG‑XR TIF via `fishtools` CLI (notable flags: `--quality`, `--threads`).
- Utilities: `utils/utils.py` — `batch_roi`, `check_if_exists`, `noglobal`; `utils/logging.py` — `initialize_logger`, `configure_cli_logging`, `setup_workspace_logging`; `utils/pretty_print.py` — progress/thread‑pool helpers (`progress_bar_threadpool`) and `jprint`. `noglobal` is a decorator preventing accidental reliance on global state.

## Testing Practices (strict; see TESTING_STRAT.md)

- Tests must pass — no exceptions. Favor TDD: analyze behavior, write focused tests, then implement. Example: `conda run -n seq pytest test/test_preprocess_config_json.py::test_round_defaults`.
- Prefer synthetic arrays; validate shapes/dtypes; use `np.allclose` for floats and `np.isfinite` checks.
- Add CLI tests for new subcommands; keep coverage stable or rising. Example: `conda run -n seq pytest -vv -k register` using the Click/Typer fixtures in `test/conftest.py`.
- Mock external FS/heavy IO only; do not mock core algorithms.
- **Refactor-resistant assertions:** verify observable outcomes/state, not internal steps; duplicate literals in tests instead of reusing production constants when asserting.
- **Test value factors:** balance regression protection, speed, maintainability, and refactor-resistance—if any drops to ~0, rewrite the test.
- **Edges only mocking:** stub or spy only at our unmanaged boundaries (SMTP, APIs) via adapters we own; assert both expected and absence of unexpected calls.
- **Control nondeterminism:** inject time, randomness, and environment explicitly (e.g., pass seeded RNGs, fixed timestamps).
- **Test design checklist:** assert observable behavior, pick unit vs integration intentionally, keep one act per test with clear Arrange→Act→Assert, expose hidden inputs, run managed deps for real while mocking unmanaged via adapters, use literals (not prod constants) in assertions, fix seeds/time/concurrency, and verify no unexpected edge calls slip through.
- **Output-first mindset:** default to output- or state-based assertions; reserve communication-based expectations for unmanaged edges through adapters.
- **Functional core, mutable shell:** keep domain logic pure and push IO/side effects to edges to keep unit tests fast and stable.
- **Anti-patterns (fix immediately):** exposing internals for tests, asserting interactions between domain objects, ambient time/random/global config, layering indirection without boundaries, or controllers that are both logic-heavy and dependency-heavy.

## Style, Typing & Tooling

- Formatting: Black (110) + isort (profile=black). Lint: Ruff. Security: Bandit. Spelling: Codespell. Typical commands: `conda run -n seq ruff check`, `conda run -n seq bandit -r fishtools`, `conda run -n seq codespell`.
- Typing is mandatory: annotate every function (params + return). **Use Python 3.12 built-ins (`list`, `dict`, `tuple`, `type[...]`) instead of importing `List`/`Dict`; pull `NDArray[...]` from `numpy.typing` when needed.**

## Common Flows

- Register: `preprocess register /workspace cortex 42 --threshold 3.0 --fwhm 4.0` (ROI `cortex`, reference round `42`).
- Concat ROIs: write JSON config validated against `schemas/config.schema.json`, then run the pipeline to produce `.h5ad` and optional Baysor CSV.
- Intensity: configure Zarr segmentation and intensity channel in JSON; run extraction to parquet (per‑slice, parallel) to the output directory specified in the config.

## Pitfalls & Tips

- Do not manually parse ROI names — always use `Workspace` or `batch_roi`.
- Watch memory: prefer Zarr/chunking for large arrays; free intermediates; validate non‑finite values.
- Keep tests deterministic; seed RNGs (e.g., `rng = np.random.default_rng(0)`); avoid relying on local file structures beyond fixtures/workspace mocks.

## User & Agent Guide

### Audience & Prerequisites

- **Who:** Power users running pipelines and contributors (human or agent).
- **Env:** Linux, Python 3.12, conda/mamba or uv, ImageJ/Fiji for stitching.
- **Data layout:** One workspace per experiment using the round/ROI naming scheme below.

### Workspace Layout & Naming

```
/workspace
├── {round}--{roi}/                     # Raw round data (e.g., 1_9_17--cortex)
│   ├── {round}-0001.tif                # 16‑bit TIFFs per index
│   └── {round}-0002.tif
└── analysis/
    └── deconv/
        ├── {round}--{roi}/             # Deconvolved images
        ├── registered--{roi}+{cb}/     # Register results (fiducials/shifts)
        ├── stitch--{roi}[/+{cb}]/      # Stitched mosaics + TileConfiguration.*
        ├── shifts--{roi}+{cb}/         # Estimated shifts
        └── opt_{cb}/                    # Optimization outputs (mse.txt, global_scale.txt)
```

- **File patterns:** `{round}-{idx:04d}.tif`, directories use `{round}--{roi}` and suffix `+{codebook}`.
- Use `Workspace` for discovery: `Workspace(path).rounds`, `.rois`, `.img()`, `.registered()`, `.stitch()`.

### End‑to‑End Recipe (Minimal)

1) **Setup**
   - `uv sync --all-extras && source .venv/bin/activate && pip install -e '.[dev]'` (ensure `uv` is installed, then mirror the same installs inside `seq`).
   - Or conda: `mamba env create -n fishtools -f environment.yml && conda activate fishtools && pip install -e '.[dev]'` (useful for iterative dev; still execute CLI/tests via `conda run -n seq`).

2) **Optional: Compression**
   - Lossless JPEG‑XR TIF: `fishtools compress /workspace -q 1.0`
   - Decompress JXL/TIF: `fishtools decompress /workspace -n 16 --out tif` (`-n` controls thread count).

3) **BaSiC Templates** (illumination correction)
   - Per round: `preprocess basic run /workspace 1_9_17 --zs 0.5` (`--zs` sets z-step spacing) → writes `analysis/deconv/basic/`
   - Batch: `preprocess basic batch /workspace -t 4 --zs 0.5`

4) **Register Images**
   - Click CLI: `preprocess register /workspace cortex 42 -t 3.0 -f 4.0` (ROI `cortex`, reference round `42`).
   - Typer CLI (new): `preprocess registerv1 /workspace cortex 42 --codebook codebook.json`
   - All in ROI: `preprocess registerv1 /workspace cortex --codebook codebook.json`
   - All: `preprocess registerv1 /workspace --codebook codebook.json`

5) **Stitch**
   - Extract + build tile config + run ImageJ: `preprocess stitch register /workspace cortex --idx 0 --threshold 0.4`
   - Fuse mosaics: `preprocess stitch fuse /workspace cortex --codebook cb1 -d 2 --threads 8` (`-d` for downsample factor, `--threads` for parallelism)
   - Combine to Zarr for analysis: `preprocess stitch combine /workspace cortex cb1 -d 2`

6) **Post‑Processing**
   - Intensity extraction (Zarr→parquet):
     `fishtools postprocess extract-intensity --config intensity_config.json` (JSON validated with `schemas/config.schema.json`)
   - Multi‑ROI concat to H5AD:
     `fishtools postprocess concat --config concat_config.json`

### Agent Playbooks

- **Add a CLI subcommand:**
  - For preprocess: implement in `fishtools/preprocess/cli_*.py`, add to group in `preprocess/cli.py`.
  - For postprocess: add command in `postprocess/cli_*.py` and register in `postprocess/cli.py`.
  - Update CLI help/docs (`--help` text, docs/ if relevant).
  - Tests: add `test/test_<feature>_cli.py` with Click/Typer runner fixtures (see existing tests).
- **Logging:** All preprocess CLIs must initialize logging via `setup_workspace_logging` so runs write to `<workspace>/analysis/logs` with meaningful `component`/`file` tags (e.g., `preprocess.register.run`).

- **Extend configuration:**
  - Add typed fields to the relevant `BaseModel` (e.g., `RegisterConfig`) with validators.
  - Plumb new options through CLI to pipeline classes; update docs and tests.

- **Use the Workspace correctly:**
  - Never glob by hand for ROIs; prefer `Workspace.rois` and `batch_roi` for `roi='*'` fan‑out.
  - Use `.img()/.regimg()` overloads to get `Path` or `np.ndarray` safely; combine with the `batch_roi` decorator from `fishtools.utils.utils` when you need to fan out over `roi='*'`.

- **Memory:** Work slice‑wise; write intermediates; validate non‑finite values (`np.isfinite`); tune chunk sizes via JSON config fields (see `schemas/config.schema.json`).
- **Repro:** Seed RNGs, pin versions in `environment.yml`/`pyproject.toml`, and record CLI invocations in PR descriptions.

## CLI Design Philosophy

- Scope & Structure
  - Each command does one thing well; compose workflows from small commands rather than building monoliths.
  - Use hyphenated command names (kebab‑case). Put required positional arguments first; options follow and use clear defaults.
  - Prefer Click for straightforward commands and Typer when richer sub‑apps or type‑driven help/validation improves UX.

- Paths & IO
  - Accept paths relevant to the task. When a project/workspace abstraction exists, resolve with its API (e.g., `Workspace`) rather than hand‑building paths. Otherwise use `pathlib.Path` consistently.
  - Never write beside raw inputs unless explicitly part of a documented layout; expose `-o/--output` for artifacts. Perform atomic writes where practical.
  - Avoid destructive actions by default; require explicit flags for overwrites.

- Logging & Errors
  - Initialize logging through the shared helpers so console and file logs are consistent when a project path is available.
  - INFO for user‑level progress, DEBUG for internals. Raise `click.ClickException` (or `typer.Exit`) with contextual messages; do not swallow exceptions. Use non‑zero exit codes on failure.

- Configuration & Precedence
  - Prefer typed Pydantic models for configuration. Precedence: CLI option > environment variable > config file > library defaults. Document any implicit defaults.

- Outputs & Naming
  - Default output directory should be predictable (e.g., sibling `output/` to the provided path) and overrideable via `-o/--output`.
  - Filenames include the minimal identifying keys (e.g., `{name}--key1[+key2].ext`) and timestamps only when helpful for uniqueness. Centralize save helpers to keep behavior uniform.

- Units & Labels
  - Always include units in axis labels and titles where relevant. Choose units that match user expectations and data scale; expose a flag only when unit switching is truly needed.

- Plotting & Reports
  - Reuse shared plotting utilities (themes, formatters, size bars). Keep figures readable: clamp DPI sensibly, decimate dense scatters, and provide options to reduce clutter.
  - Offer overlap‑avoidance for annotations where beneficial and make it optional to keep performance predictable.

- UX & Interactivity
  - Default to non‑interactive, reproducible behavior. If a command supports interactive flows, provide equivalent non‑interactive flags and print the paths to all generated artifacts.
  - Use kebab‑case for long options, short aliases only for very common flags (e.g., `-o` for output). Boolean flags should support paired `--foo/--no-foo` form.

- Performance & Determinism
  - Stream or chunk large inputs; avoid loading entire datasets into memory when unnecessary. Make parallelism explicit via `--threads` and select sensible defaults.
  - Ensure outputs are deterministic given the same inputs and flags (seed RNGs where applicable).

- Testing
  - Write focused tests for CLI helpers and behaviors (use Click/Typer runners). Assert observable results (files created, figure labels, exit codes) rather than pixel‑perfect images or internals.
  - Keep tests fast and independent of large binaries by using synthetic data and temporary directories.

- Deprecation
  - Avoid introducing options you intend to remove. When necessary, deprecate explicitly with clear help text and remove in the next release; do not keep no‑op flags.

### Canonical CLI Syntax

Conventions used across CLIs

- Positional order: `<workspace> [ROI ...]` unless the command is naturally tied to a single artifact (e.g., a model file).
- ROIs are positional, optional, and may be variadic (`nargs=-1`) where supported. If omitted, the command operates on all detected ROIs. The wildcard `*` and the token `all` (when accepted) are treated as “all ROIs”.
- Codebooks: pass via `--codebook <label_or_path>`. Labels may be stems (e.g., `cs-base`); paths may be full JSON files.
- File naming: filesystem‑friendly codebook labels replace hyphens/spaces with underscores (see `Workspace.sanitize_codebook_name`). Per‑ROI parquet files: `{roi}+{sanitize(codebook)}.parquet` under `<workspace>/analysis/output/`.
- Batch vs single ROI: single‑ROI commands fail fast; multi‑ROI commands log per‑ROI issues and continue.

Preprocess — spots

- Overlay spots onto segmentation masks
  - `preprocess spots overlay <workspace> [ROI] --codebook <cb> [--seg-codebook <seg_cb>] [--spots <file_or_dir>] [--overwrite]`
  - Skips ROIs missing `stitch--ROI[+CB]/output_segmentation.zarr` or the spots parquet.

- Threshold selection (interactive)
  - `preprocess spots threshold <workspace> -c <codebook.json> [ROI ...]`
  - ROIs are positional; none or any of `*`/`all` → process all.

- Plot all genes (density/hexbins)
  - `preprocess spots plotall <workspace> [ROI] --codebook <label>`
  - ROI is optional (`*` for all). Output defaults to `<workspace>/analysis/output/plots/`.

**Illumination Field Correction (TCYX)**

- Export the correction field (per ROI/codebook) from your model:
  - `preprocess correct-illum export-field <model.npz> --what both --downsample 1 --output <field.zarr>`
    - Produces a single Zarr with `axes="TCYX"`, where `T=[low,range]`. Values outside the union-of-tiles mask are set to `1.0`.

- Apply during spot workflows (before percentiles/spot calling):
  - Percentiles: `preprocess spots find-threshold <workspace> <ROI> --codebook <cb.json> --field-zarr <field.zarr>`
  - Spot calling: `preprocess spots run <tile.tif> --codebook <cb.json> --field-zarr <field.zarr> ...`

- How alignment works:
  - Uses registered tile origin (TileConfiguration.registered.txt) and any local slice offset (e.g., quadrants) to index the field.
  - Uses `model_meta.x0/y0` and `model_meta.downsample` from the Zarr to convert native pixels → downsampled grid indices.
  - Channel mapping uses the TIFF `key` metadata; if missing, a positional mapping is used.
  - Outside the union-of-tiles region, both `low` and `range` are `1.0` (identity) so images are unchanged there.

- GPU behavior and performance:
  - For each plane: slice downsampled field once per (tile, channel), upsample and apply on GPU in a single block, transfer only the corrected plane back to CPU, free GPU memory immediately.
  - The downsampled field slices are cached per (channel, x_off, y_off) and reused across Z for the tile to minimize IO.

- Constraints & tips:
  - Use `--downsample 1` when exporting the field for simplest alignment (other factors work but require resampling).
  - Ensure the TCYX store attributes include `t_labels=['low','range']` and `model_meta` with `x0, y0, downsample, channels`.

Preprocess — stitch

- Register tiles (simple, using existing TileConfiguration)
  - `preprocess stitch register-simple <registered_dir> --tileconfig <TileConfiguration.txt> [--fuse] [-d <downsample>] [--config <json>]`

- Register tiles per ROI (channel extraction + ImageJ run)
  - `preprocess stitch register <workspace> [ROI] [--codebook <label>] [--idx <channel>|--max-proj] [--fid] [--threshold <float>] [--config <json>] [--overwrite]`

- Fuse registered tiles (extract + fuse)
  - `preprocess stitch fuse <workspace> [ROI] --codebook <label> [-d <downsample>] [--threads N] [--subsample-z N] [--is-2d] [--max-proj] [--debug] [--config <json>] [--field-zarr <TCYX.zarr>]`
  - Field correction uses a single TCYX store exported by `correct-illum export-field --what both --downsample 1`. When `--field-zarr` is provided, set `-d/--downsample 1` for correct alignment.

- Combine fused tiles to Zarr
  - `preprocess stitch combine <workspace> [ROI] --codebook <label> [--chunk-size 2048] [--overwrite]`

- N4 bias-field correction on stitched mosaics
  - `preprocess stitch n4 <workspace> [ROI] --codebook <label> --z-index <z> [--channels ...] [--shrink N] [--threshold <method|value>] [--overwrite] [--debug]`

- One-shot pipeline wrapper (extract→fuse→combine)
  - `preprocess stitch run <workspace> [ROI] --codebook <label> [-d <downsample>] [--threads N] [--channels "-3,-2,-1"] [--overwrite]`

Preprocess — register

- Typer wrapper (newer UI)
  - `preprocess registerv1 <workspace> [ROI] --codebook <codebook.json>`
  - Examples in `preprocess registerv1 --help` show single‑ROI and all‑ROI forms.

- Click commands (legacy; retains explicit `--roi`)
  - Single index: `preprocess register run <workspace> <idx> --codebook <codebook.json> [--roi ROI|'*'] [--reference <round>] [--threshold F] [--fwhm F] [--overwrite]`
  - Batch: `preprocess register batch <workspace> --codebook <codebook.json> [--ref <round>] [--threads N] [--overwrite] [--verify]`

Preprocess — basic (illumination templates)

- Per round: `preprocess basic run <workspace> <round> [--zs 0.5] [--overwrite] [--seed N]`
- Batch all rounds: `preprocess basic batch <workspace> [-t N] [--zs 0.5] [--overwrite] [--seed N]`

Preprocess — deconvolution (new)

- Precompute global quantization: `preprocess deconv precompute <workspace> <round> [--bins 8192] [--p-low F] [--p-high F] [--gamma F] [--i-max N]`
- Quantize float32 tiles: `preprocess deconv quantize <workspace> <round> [--roi ROI ...] [--n-fids 2] [--overwrite]`
  - Note: this subcommand uses `--roi` (repeatable) instead of positional ROIs.

Preprocess — illumination correction (RBF)

- Render global field PNG from a model: `preprocess correct-illum render-global <model.tiff.json> [--workspace <ws>] [--roi ROI] [--codebook <label>] [--output <png>]`
- Compute subtile percentiles: `preprocess correct-illum calculate-percentiles <workspace> <roi> [idx] [--codebook <label>] [--percentiles Plo Phi] [--grid N] [--overwrite]`
- Generate field stores (Zarr): `preprocess correct-illum field-generate <workspace> <roi> --codebook <label> [--grid-step F] [--neighbors N] [--smoothing F] [--output <zarr>]`
- Plot field: `preprocess correct-illum plot-field <model.tiff.json> [--workspace <ws>] [--roi ROI] [--codebook <label>] [--output <png>]`
- Export field to Zarr (TCYX): `preprocess correct-illum export-field <model.npz> [--workspace <ws>] [--roi ROI] [--codebook <label>] [--downsample N] [--what range|low|both] [--output <field.zarr>]`
  - Exports a single Zarr array with `axes="TCYX"`. `T` indexes field kinds in order `[low, range]` (when `--what=both`; otherwise `T=1`). Values outside the union-of-tiles mask are set to `1.0`.
  - Use this store directly in stitching via `preprocess stitch fuse ... --field-zarr <field.zarr> --downsample 1`.

Preprocess — verification and QC

- Verify registered TIFFs by codebook: `preprocess verify codebook <workspace> <codebook.json> [--roi ROI ...]`
- Verify arbitrary paths (delete by default): `preprocess verify path <paths...> [--no-delete]`
- Stitch layout check: `preprocess check-stitch <workspace> [--roi ROI ...] [-o <outdir>] [--per-roi]`

Segment

- Extract training slices/projections: `segment extract z|ortho <workspace> [ROI] --codebook <label> [--n N] [--dz N] [--anisotropy N] [--channels ...] [--out <dir>]`
- Train model: `segment train <path> <name>` (reads `<path>/models/<name>.json`).

Postprocess

- Intensity extraction: `fishtools postprocess extract-intensity --config <intensity.toml> [--workspace <ws>] [--roi ROI] [--channel NAME] [--max-workers N] [--overwrite] [--dry-run]`

Notes and exceptions

- Some legacy subcommands (e.g., `preprocess register run`, `preprocess deconv quantize`, `preprocess check-stitch`) still accept ROI via `--roi` options rather than positional ROIs. Follow each command’s `--help` for authoritative usage while we migrate toward the unified positional‑ROI style.
- When both a label and a JSON path are accepted for `--codebook`, the stem of the JSON is treated as the label.

### Frequently Asked Questions

- **Where do I put raw data?**
  - `/workspace/{round}--{roi}/{round}-0001.tif` etc. Rounds use `a_b_c` bit naming; ROI is free‑form.
- **Which CLI to use for registration?**
  - `preprocess register` (Click) is stable; `preprocess registerv1` wraps the newer Typer app.
- **How do I get a `stitch--` folder?**
  - Run `preprocess stitch register` then `preprocess stitch fuse` (and optionally `combine`).
- **How do I create the H5AD?**
  - Write `concat_config.json` (validated with `schemas/config.schema.json`) and run `fishtools postprocess concat --config concat_config.json`.

### Scientific Computing Test Considerations

**NumPy Array Testing**:

- Use `np.allclose()` for floating point comparisons
- Set explicit tolerances (e.g., `np.allclose(a, b, rtol=1e-5, atol=1e-8)`)
- Check `np.isfinite()` for NaN/inf detection
- Verify array shapes and dtypes explicitly
- Test with synthetic data with known ground truth
- Prefer `np.random.default_rng(0)` for reproducible data synthesis

**Image Processing Validation**:

- Create synthetic images with known features
- Verify that processing enhances expected regions
- Track simple metrics (mean intensity, SNR) before/after processing
- Test edge cases: zeros, uniform data, extreme values
- Check that algorithms handle different input shapes (2D vs 3D)

AGAIN DO NOT EVER use %s formatting for logging. ALWAYS use f-strings.
