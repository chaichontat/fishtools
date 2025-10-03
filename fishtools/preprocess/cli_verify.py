from __future__ import annotations

import sys
from pathlib import Path

import rich_click as click
from loguru import logger
from tqdm.auto import tqdm

from fishtools.io.workspace import CorruptedTiffError, Workspace


@click.group()
def verify() -> None:
    """Validate TIFF artifacts for corruption.

    Subcommands:
    - codebook: scan all registered TIFFs for a given codebook (no deletion).
    - path: scan one or more filesystem paths; deletes corrupted files by default.

    Behavior:
    - Emits one logger.error line per corrupted file; otherwise stays quiet on success.
    - Shows a progress bar on TTY; auto-disables in non-interactive runs.
    - Exit status 0 when all files are readable; 1 if any file is corrupted or inputs are invalid.
    """


@verify.command("codebook")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("codebook", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.option("--roi", "rois", multiple=True, help="Restrict verification to one or more ROI identifiers.")
def verify_codebook(path: Path, codebook: Path, rois: tuple[str, ...]) -> None:
    """Scan registered TIFFs for a codebook and report corruption.

    Notes:
    - This command never deletes files; it only reports issues.
    - Output is a single logger.error line per corrupted file: "<filename> corrupted.".
    - Optionally restrict scanning to specific ROI(s) with --roi.
    """

    workspace = Workspace(path)
    codebook_name = codebook.stem

    if rois:
        try:
            roi_param: tuple[str, ...] | None = tuple(workspace.resolve_rois(rois))
        except ValueError as exc:  # pragma: no cover - exercised in CLI tests
            raise click.BadParameter(str(exc), param_hint="roi") from exc
    else:
        roi_param = None

    file_map, missing_rois = workspace.registered_file_map(codebook_name, rois=roi_param)

    if rois and missing_rois:
        missing = ", ".join(missing_rois)
        raise click.ClickException(
            f"No registered data found for ROI(s) {missing} with codebook '{codebook_name}'."
        )

    files: list[Path] = [path for paths in file_map.values() for path in paths]

    if not files:
        raise click.ClickException(f"No registered TIFF files found for codebook '{codebook_name}'.")

    corrupted: list[tuple[Path, str]] = []
    disable_pb = not sys.stderr.isatty()
    with tqdm(total=len(files), unit="file", desc="Verifying", disable=disable_pb) as progress:
        for file_path in files:
            try:
                Workspace.ensure_tiff_readable(file_path)
            except CorruptedTiffError as error:
                corrupted.append((file_path, str(error)))
            finally:
                progress.update(1)

    if corrupted:
        for path_checked, _ in corrupted:
            logger.error(f"{path_checked.name} corrupted.")
        raise SystemExit(1)


@verify.command("path")
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.option(
    "--no-delete",
    is_flag=True,
    help="Keep corrupted files (default deletes them).",
)
def verify_path(paths: tuple[Path, ...], no_delete: bool) -> None:
    """Recursively scan one or more paths for TIFF corruption.

    What it checks:
    - Directories: recursively traverses and checks all .tif/.tiff files under each.
    - Files: checks that single TIFF file.

    Output and exit codes:
    - Prints one logger.error line per corrupted file.
    - Deletes corrupted files by default; use --no-delete to keep them.
    - Exits 0 if all files are readable; exits 1 if any are corrupted or inputs are invalid.

    Examples:
    - preprocess verify path /data/runA /data/runB
    - preprocess verify path /data/**/*.tif  (shell-expanded)
    - preprocess verify path /data/runA --no-delete
    """

    def list_tiffs(root: Path) -> list[Path]:
        if root.is_dir():
            return sorted([*root.rglob("*.tif"), *root.rglob("*.tiff")])
        # Single file
        if root.suffix.lower() in (".tif", ".tiff"):
            return [root]
        raise click.ClickException(f"Path {root} is not a TIFF file.")

    if not paths:
        raise click.BadParameter("Provide at least one PATH to scan.")

    # Aggregate and de-duplicate discovered TIFFs while preserving order
    seen: set[Path] = set()
    files: list[Path] = []
    for root in paths:
        for f in list_tiffs(root):
            if f not in seen:
                seen.add(f)
                files.append(f)
    if not files:
        if len(paths) == 1:
            raise click.ClickException(f"No TIFF files found under {paths[0]}.")
        raise click.ClickException("No TIFF files found under any provided path(s).")

    corrupted: list[tuple[Path, str]] = []
    disable_pb = not sys.stderr.isatty()
    with tqdm(total=len(files), unit="file", desc="Verifying", disable=disable_pb) as progress:
        for file_path in files:
            try:
                Workspace.ensure_tiff_readable(file_path)
            except CorruptedTiffError as error:
                corrupted.append((file_path, str(error)))
            finally:
                progress.update(1)

    # Optionally delete corrupted files (default behavior)
    deleted = 0
    if corrupted and not no_delete:
        for path_checked, _ in corrupted:
            try:
                path_checked.unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:  # pragma: no cover - OS permission edge
                logger.warning(f"Failed to delete {path_checked}: {exc}")

    if corrupted:
        for path_checked, _ in corrupted:
            if not no_delete:
                logger.error(f"{path_checked.name} corrupted. Deleted.")
            else:
                logger.error(f"{path_checked.name} corrupted.")
        raise SystemExit(1)
