# File Path Contracts

## Design Goal: Treat Filesystem Layout as a First‑Class API

- File paths and directory layouts are part of the public API contract between CLIs, Python modules, and external tools, not an implementation detail.
- All producers and consumers of on‑disk artifacts must use shared helpers (for example, `Workspace` or dedicated path utilities) instead of hand‑rolled string concatenation or duplicated literals.
- There must be a single source of truth for each path contract (code + docs). When a location changes, the abstraction is updated, not every call‑site.
- File location contracts should be treated like any other API: changes are versioned and, when necessary, deprecated with a compatibility window rather than silently broken.
- Runtime validation and tests should enforce the contract: producers must write to the canonical location, consumers must auto‑discover from that location, and failures must be loud and contextual.
- Avoid “stringly‑typed” file contracts: prefer `Path` objects and shared helpers over scattered string patterns in CLIs, library code, and notebooks.

## Submodule‑Scoped Path Contracts

- For file pairs or layouts used only between a small number of modules (e.g. a single feature or workflow), define a **feature‑scoped path helper**, not a global `Workspace` method.
- Represent these helpers as small classes or dataclasses that encapsulate path logic and behavior, for example `SalvagePaths` or `StitchPaths`, rather than Pydantic models.
- Both the producer and consumer must depend on the same helper class (or module‑level functions) to derive paths, avoiding duplicated literals and hand‑rolled path assembly.
- Promote a feature‑scoped path helper into `Workspace` (or another shared namespace) only when the same path contract is clearly used across multiple domains or CLIs.
