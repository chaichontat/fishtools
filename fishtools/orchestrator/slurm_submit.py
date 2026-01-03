"""SLURM job array submission utilities for fishtools workflows.

This module provides functions to generate and submit SLURM batch scripts
for parallel processing. Intended to be called from bigrun.py or similar
orchestration scripts.
"""

import math
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import click
import yaml
from pydantic import BaseModel

from fishtools.io.workspace import Workspace

PROFILES_PATH = Path(__file__).parent / "slurm_profiles.yaml"
LOGS_DIR = Path.home() / "logs"


class SlurmConfig(BaseModel):
    partition: str = "l40s"
    gpus: int = 2
    cpus: int = 8
    mem: str = "64G"
    time: str = "02:00:00"
    job_name: str = "fishtools"


class SlurmProfiles(BaseModel):
    default: SlurmConfig = SlurmConfig()
    profiles: dict[str, SlurmConfig] = {}

    def get(self, profile: str) -> SlurmConfig:
        """Get config for profile, merging with defaults."""
        if profile not in self.profiles:
            return self.default.model_copy()
        # Merge: start with default, override with profile values
        merged = self.default.model_dump()
        merged.update(self.profiles[profile].model_dump(exclude_unset=True))
        return SlurmConfig(**merged)


def load_profiles() -> SlurmProfiles:
    """Load SLURM profiles from YAML config."""
    if not PROFILES_PATH.exists():
        return SlurmProfiles()
    with open(PROFILES_PATH) as f:
        data = yaml.safe_load(f)
    return SlurmProfiles.model_validate(data)


def make_job_name(ws_path: Path, job_type: str = "deconv") -> str:
    """Create a unique job name encoding the workspace.

    Format: {workspace_name}_{job_type}
    """
    return f"{ws_path.resolve().name}_{job_type}"


def get_running_job_count(job_name: str) -> int:
    """Get count of running/pending jobs with this name.

    Args:
        job_name: The SLURM job name to check

    Returns:
        Number of running or pending jobs with this name
    """
    try:
        result = subprocess.run(
            f"squeue -u $USER -n {job_name} -h | wc -l",
            capture_output=True,
            text=True,
            shell=True,
        )
        if result.returncode != 0:
            return 0
        return int(result.stdout.strip())
    except Exception:
        return 0


def confirm_or_abort(prompt: str, confirm: bool) -> bool:
    """Prompt the user to continue, unless confirmation is disabled."""
    if not confirm:
        return True
    reply = input(prompt).strip().lower()
    if reply != "y":
        print("Aborted.")
        return False
    return True


def warn_if_running(ws: Workspace, job_name: str, confirm: bool) -> bool:
    """Warn if jobs already exist and optionally confirm continuation."""
    running_count = get_running_job_count(job_name)
    if running_count > 0:
        print(f"Warning: {running_count} job(s) already running/pending for {ws.path.name}")
        return confirm_or_abort("Submit anyway? [y/N] ", confirm)
    return True


def append_extra_args(command: str, extra_args: tuple[str, ...]) -> str:
    """Append extra CLI args to a command string with shell-safe quoting."""
    if not extra_args:
        return command
    return f"{command} {shlex.join(extra_args)}"


def parse_depends_on(depends_on: str | None) -> str | None:
    """Parse comma-separated SLURM job IDs into an afterok dependency."""
    if not depends_on:
        return None
    job_ids = [job_id.strip() for job_id in depends_on.split(",") if job_id.strip()]
    if not job_ids:
        return None
    invalid = [job_id for job_id in job_ids if not job_id.isdigit()]
    if invalid:
        raise click.ClickException(
            f"Invalid --depends-on entries (must be numeric job IDs): {', '.join(invalid)}"
        )
    return f"afterok:{':'.join(job_ids)}"


def split_rounds_and_extra(
    rounds: tuple[str, ...],
    extra_args: tuple[str, ...],
) -> tuple[list[str], tuple[str, ...]]:
    """Split variadic rounds from pass-through args without requiring `--`."""
    if extra_args:
        return list(rounds), extra_args
    for idx, value in enumerate(rounds):
        if value.startswith("-"):
            return list(rounds[:idx]), tuple(rounds[idx:])
    return list(rounds), ()


def split_roi_and_extra(
    roi_and_extra: tuple[str, ...],
    extra_args: tuple[str, ...],
) -> tuple[str | None, tuple[str, ...]]:
    rois, merged_extra = split_rounds_and_extra(roi_and_extra, extra_args)
    if not rois:
        return None, merged_extra
    if len(rois) != 1:
        raise click.ClickException(f"Expected at most one ROI, got: {', '.join(rois)}")
    return rois[0], merged_extra


def submit_single_task(
    command: str,
    config: SlurmConfig,
    *,
    dry_run: bool,
    confirm: bool,
    task_label: str,
    submit_prompt: str,
    dependency: str | None = None,
) -> None:
    """Print a single-task summary and submit it."""
    print(f"Task: {task_label}")
    print(f"  Command: {command}")
    print()

    if dry_run:
        print("Dry run - not submitting")
        return

    if not confirm_or_abort(submit_prompt, confirm):
        return

    job_id = submit_array([command], config, dry_run=dry_run, dependency=dependency)
    if job_id:
        print(f"Submitted job {job_id}")


def submit_labeled_tasks(
    commands: list[str],
    labels: list[str],
    config: SlurmConfig,
    *,
    dry_run: bool,
    confirm: bool,
    header: str,
    empty_message: str = "No ROIs found",
    submit_prompt: str | None = None,
    dependency: str | None = None,
) -> None:
    """Print a labeled task list and submit as a job array."""
    if not commands:
        print(empty_message)
        return

    print(header)
    for label in labels:
        print(f"  - {label}")
    print()

    if dry_run:
        print("Dry run - not submitting")
        return

    if submit_prompt is None:
        submit_prompt = f"Submit {len(commands)} jobs? [y/N] "
    if not confirm_or_abort(submit_prompt, confirm):
        return

    job_id = submit_array(commands, config, dry_run=dry_run, dependency=dependency)
    if job_id:
        print(f"Submitted job {job_id}")


def submit_one(
    ws: Workspace,
    *,
    profile_name: str,
    job_type: str,
    command: str,
    ctx_args: tuple[str, ...],
    dry_run: bool,
    yes: bool,
    depends_on: str | None,
    task_label: str,
    submit_prompt: str,
    config_transform: Callable[[SlurmConfig], None] | None = None,
) -> None:
    config = load_profiles().get(profile_name)
    job_name = make_job_name(ws.path, job_type)
    config.job_name = job_name
    if config_transform is not None:
        config_transform(config)

    if not warn_if_running(ws, job_name, confirm=not yes):
        return

    submit_single_task(
        append_extra_args(command, ctx_args),
        config,
        dry_run=dry_run,
        confirm=not yes,
        task_label=task_label,
        submit_prompt=submit_prompt,
        dependency=parse_depends_on(depends_on),
    )


def submit_per_roi(
    ws: Workspace,
    *,
    profile_name: str,
    job_type: str,
    label_prefix: str,
    header: str,
    build_cmd_for_roi: Callable[[str], str],
    ctx_args: tuple[str, ...],
    dry_run: bool,
    yes: bool,
    depends_on: str | None,
    config_transform: Callable[[SlurmConfig], None] | None = None,
    rois: Iterable[str] | None = None,
) -> None:
    config = load_profiles().get(profile_name)
    job_name = make_job_name(ws.path, job_type)
    config.job_name = job_name
    if config_transform is not None:
        config_transform(config)

    if not warn_if_running(ws, job_name, confirm=not yes):
        return

    try:
        resolved_rois = ws.resolve_rois(rois)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    commands: list[str] = []
    labels: list[str] = []
    for roi in resolved_rois:
        labels.append(f"{label_prefix}:{roi}")
        commands.append(append_extra_args(build_cmd_for_roi(roi), ctx_args))

    submit_labeled_tasks(
        commands,
        labels,
        config,
        dry_run=dry_run,
        confirm=not yes,
        header=f"{header} ({len(resolved_rois)} ROIs)",
        dependency=parse_depends_on(depends_on),
    )


_MEM_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[KMGTPkmgpt]?)$")


def _scale_mem(mem: str, factor: float) -> str:
    match = _MEM_RE.match(mem.strip())
    if match is None:
        raise click.ClickException(f"Unsupported mem format: {mem!r} (expected e.g. '64G')")

    value = float(match.group("value"))
    unit = match.group("unit")

    scaled = value * factor
    scaled_int = int(math.ceil(scaled)) if factor >= 1 else int(math.floor(scaled))
    if scaled_int < 1:
        scaled_int = 1
    return f"{scaled_int}{unit}"


def generate_batch_script(
    commands: list[str],
    config: SlurmConfig,
) -> str:
    """Generate SLURM batch script content.

    Args:
        commands: List of shell commands, one per array task
        config: SLURM resource configuration

    Returns:
        Batch script content as string
    """
    case_branches = []
    for i, cmd in enumerate(commands):
        case_branches.append(f'    {i}) {cmd} ;;')

    return f'''#!/bin/bash
#SBATCH --job-name={config.job_name}
#SBATCH --output={LOGS_DIR}/%A_{config.job_name}_%a.out
#SBATCH --error={LOGS_DIR}/%A_{config.job_name}_%a.err
#SBATCH --partition={config.partition}
#SBATCH --gres=gpu:{config.gpus}
#SBATCH --cpus-per-task={config.cpus}
#SBATCH --mem={config.mem}
#SBATCH --time={config.time}

set -euo pipefail

module load GCC/13.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cp4

export DATA_PATH=~/data
export IMAGEJ_PATH=~/Fiji/fiji-headless.sh

case $SLURM_ARRAY_TASK_ID in
{chr(10).join(case_branches)}
    *) echo "Unknown task ID: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
'''


def submit_array(
    commands: list[str],
    config: SlurmConfig | None = None,
    dry_run: bool = False,
    dependency: str | None = None,
) -> str | None:
    """Submit a SLURM job array and return the job ID.

    Args:
        commands: List of shell commands, one per array task
        config: SLURM resource configuration (uses defaults if None)
        dry_run: If True, print script instead of submitting
        dependency: Optional SLURM dependency string (e.g., "afterok:12345")

    Returns:
        Job ID string if submitted, None if dry_run or no commands
    """
    result = submit_array_result(commands, config, dry_run=dry_run, dependency=dependency)
    return result.job_id if result is not None else None


class SubmitResult(NamedTuple):
    job_id: str
    stdout: str
    stderr: str
    script: str


def submit_array_result(
    commands: list[str],
    config: SlurmConfig | None = None,
    *,
    dry_run: bool = False,
    dependency: str | None = None,
) -> SubmitResult | None:
    """Submit a SLURM job array and return job_id plus sbatch stdout/stderr.

    Useful for orchestrators that want to persist submission diagnostics.
    """
    if not commands:
        return None

    if config is None:
        config = SlurmConfig()

    script_content = generate_batch_script(commands, config)

    if dry_run:
        print(f"[Dry run] Would submit {len(commands)} tasks")
        return None

    LOGS_DIR.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        n = len(commands) - 1
        sbatch_args = ["sbatch", f"--array=0-{n}"]
        if dependency:
            sbatch_args.append(f"--dependency={dependency}")
        sbatch_args.append(script_path)

        result = subprocess.run(
            sbatch_args,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"SLURM error (exit code {result.returncode}):")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, sbatch_args, result.stdout, result.stderr)

        stdout = result.stdout.strip()
        stderr = (result.stderr or "").strip()
        print(stdout)
        job_id = stdout.split()[-1] if stdout else ""
        if not job_id:
            raise RuntimeError(f"Failed to parse sbatch job ID from output: {stdout!r}")
        return SubmitResult(job_id=job_id, stdout=stdout, stderr=stderr, script=script_content)
    finally:
        Path(script_path).unlink()


def submit_deconv(
    ws: Workspace,
    config: SlurmConfig | None = None,
    dry_run: bool = False,
    confirm: bool = True,
    include_nonbit: bool = False,
    rounds: list[str] | None = None,
    extra_args: tuple[str, ...] = (),
    dependency: str | None = None,
) -> list[str]:
    """Submit deconvolution jobs for pending tasks.

    Submits jobs in phases with SLURM dependencies:
    1. Phase 1: bit round "run" tasks + non-bit "prepare" tasks (no dependencies)
    2. Phase 2: non-bit "run_u16" tasks (depends on Phase 1 completion)

    Args:
        ws: Workspace instance
        config: SLURM config (uses "deconv" profile from slurm_profiles.yaml if None)
        dry_run: If True, print commands without submitting
        confirm: If True, prompt user for confirmation before submitting
        include_nonbit: If True, also submit prepare/precompute for non-bit rounds
        rounds: Optional list of round names to filter tasks

    Returns:
        List of submitted job IDs
    """
    job_name = make_job_name(ws.path, "deconv")

    if config is None:
        config = load_profiles().get("deconv")
    config.job_name = job_name

    if not warn_if_running(ws, job_name, confirm):
        return []

    # Validate round names if specified
    if rounds:
        invalid = set(rounds) - set(ws.rounds)
        if invalid:
            raise click.ClickException(f"Unknown round(s): {', '.join(sorted(invalid))}")

    # If specific rounds given, auto-enable nonbit for any non-bit rounds in the list
    effective_include_nonbit = include_nonbit
    if rounds:
        has_nonbit = any(not Workspace.is_bit_round(r) for r in rounds)
        if has_nonbit:
            effective_include_nonbit = True

    tasks = ws.pending_deconv_tasks(include_nonbit=effective_include_nonbit)

    # Filter by round names if specified
    if rounds:
        rounds_set = set(rounds)
        tasks = [(r, roi, t) for r, roi, t in tasks if r in rounds_set]

    # Separate tasks into phases
    # Phase 1: run (bit) + prepare (non-bit) - no dependencies
    # Phase 2: run_u16 (non-bit) - depends on phase 1
    phase1_labels: list[str] = []
    phase1_commands: list[str] = []
    phase2_labels: list[str] = []
    phase2_commands: list[str] = []

    # Track which rounds have prepare tasks - we'll generate run_u16 for all their ROIs
    prepare_rounds: set[str] = set()

    for round_name, roi, task_type in tasks:
        if task_type == "run":
            phase1_labels.append(f"{round_name}:{roi}")
            phase1_commands.append(
                append_extra_args(
                    f"preprocess deconvnew run {ws.path} {round_name} --roi={roi} --devices=auto --mode=legacy",
                    extra_args,
                )
            )
        elif task_type == "prepare":
            phase1_labels.append(f"{round_name} (prepare)")
            prepare_cmd = append_extra_args(
                f"preprocess deconvnew prepare {ws.path} {round_name}",
                extra_args,
            )
            precompute_cmd = append_extra_args(
                f"preprocess deconvnew precompute {ws.path} {round_name}",
                extra_args,
            )
            phase1_commands.append(f"{prepare_cmd} && {precompute_cmd}")
            prepare_rounds.add(round_name)
        elif task_type == "run_u16":
            phase2_labels.append(f"{round_name}:{roi} (u16)")
            phase2_commands.append(
                append_extra_args(
                    f"preprocess deconvnew run {ws.path} {round_name} --roi={roi} --devices=auto --mode=u16",
                    extra_args,
                )
            )

    # For rounds with prepare tasks, generate run_u16 for all ROIs with source tiles
    for round_name in prepare_rounds:
        for roi in ws.rois:
            src_count = len(list(ws.path.glob(f"{round_name}--{roi}/{round_name}-*.tif")))
            if src_count > 0:
                phase2_labels.append(f"{round_name}:{roi} (u16)")
                phase2_commands.append(
                    append_extra_args(
                        f"preprocess deconvnew run {ws.path} {round_name} --roi={roi} --devices=auto --mode=u16",
                        extra_args,
                    )
                )

    # Show summary
    if not phase1_commands and not phase2_commands:
        print("No pending tasks")
        return []

    print(f"Pending tasks for {ws.path.name}:")
    if phase1_labels:
        print(f"\n  Phase 1 ({len(phase1_labels)} tasks):")
        for label in phase1_labels:
            print(f"    - {label}")
    if phase2_labels:
        print(f"\n  Phase 2 ({len(phase2_labels)} tasks) - depends on Phase 1:")
        for label in phase2_labels:
            print(f"    - {label}")
    print()

    if dry_run:
        print("Dry run - not submitting")
        return []

    total = len(phase1_commands) + len(phase2_commands)
    if not confirm_or_abort(f"Submit {total} tasks? [y/N] ", confirm):
        return []

    job_ids: list[str] = []

    # Submit phase 1
    if phase1_commands:
        config.job_name = f"{job_name}_p1"
        job_id = submit_array(phase1_commands, config, dry_run=dry_run, dependency=dependency)
        if job_id:
            job_ids.append(job_id)

    # Submit phase 2 with dependency on phase 1
    if phase2_commands:
        config.job_name = f"{job_name}_p2"
        phase2_dependency = f"afterok:{job_ids[0]}" if job_ids else dependency
        job_id = submit_array(phase2_commands, config, dry_run=dry_run, dependency=phase2_dependency)
        if job_id:
            job_ids.append(job_id)

    return job_ids


@click.group()
def cli():
    """SLURM job array submission for fishtools workflows."""
    pass


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("rounds", nargs=-1)
@click.option("--include-nonbit", is_flag=True, help="Include non-bit rounds (prepare/precompute + run_u16)")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--partition", default=None, help="SLURM partition (overrides profile)")
@click.option("--gpus", default=None, type=int, help="Number of GPUs per task (overrides profile)")
@click.option("--cpus", default=None, type=int, help="CPUs per task (overrides profile)")
@click.option("--mem", default=None, help="Memory per task (overrides profile)")
@click.option("--time", default=None, help="Time limit per task (overrides profile)")
@click.pass_context
def deconv(
    ctx: click.Context,
    path: Path,
    rounds: tuple[str, ...],
    include_nonbit: bool,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
    partition: str | None,
    gpus: int | None,
    cpus: int | None,
    mem: str | None,
    time: str | None,
):
    """Submit deconvolution jobs for a workspace.

    ROUNDS: Optional round names to filter (e.g., 1_9_17 dapi). If not specified, all pending rounds.

    Uses 'deconv' profile from slurm_profiles.yaml. CLI options override profile values.
    """
    ws = Workspace(path)

    # Load from profile, then apply CLI overrides
    config = load_profiles().get("deconv")
    if partition is not None:
        config.partition = partition
    if gpus is not None:
        config.gpus = gpus
    if cpus is not None:
        config.cpus = cpus
    if mem is not None:
        config.mem = mem
    if time is not None:
        config.time = time

    resolved_rounds, extra_args = split_rounds_and_extra(rounds, tuple(ctx.args))

    submit_deconv(
        ws,
        config=config,
        dry_run=dry_run,
        confirm=not yes,
        include_nonbit=include_nonbit,
        rounds=resolved_rounds if resolved_rounds else None,
        extra_args=extra_args,
        dependency=parse_depends_on(depends_on),
    )


@cli.command("basic", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def basic(
    ctx: click.Context,
    path: Path,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit basic preprocessing job for a workspace.

    Runs `preprocess basic run` on all ROIs.
    """
    ws = Workspace(path)
    submit_one(
        ws,
        profile_name="basic",
        job_type="basic",
        command=f"preprocess basic run {ws.path} all",
        ctx_args=tuple(ctx.args),
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        task_label=f"basic preprocessing for {ws.path.name}",
        submit_prompt="Submit job? [y/N] ",
    )


@cli.command("compute-range", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def compute_range(
    ctx: click.Context,
    path: Path,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit compute-range job for a workspace.

    Runs `preprocess deconv compute-range` on the deconvolved directory.
    """
    ws = Workspace(path)
    submit_one(
        ws,
        profile_name="compute-range",
        job_type="compute-range",
        command=f"preprocess deconv compute-range {ws.deconved}",
        ctx_args=tuple(ctx.args),
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        task_label=f"compute-range for {ws.path.name}",
        submit_prompt="Submit job? [y/N] ",
    )


@cli.command("register", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--threads", default=None, type=int, help="Number of threads (defaults to cpus from profile)")
@click.option(
    "--intensity",
    is_flag=True,
    help="Use register-batch-intensity profile instead of register-batch.",
)
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def register(
    ctx: click.Context,
    path: Path,
    codebook: Path,
    roi_and_extra: tuple[str, ...],
    threads: int | None,
    intensity: bool,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit register batch jobs for a workspace, sharded by ROI.

    Runs `preprocess register batch` on each ROI in the deconvolved directory.
    """
    ws = Workspace(path)
    profile_name = "register-batch-intensity" if intensity else "register-batch"

    # Default to 18 threads (less than cpus to leave headroom)
    if threads is None:
        threads = 24

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    submit_per_roi(
        ws,
        profile_name=profile_name,
        job_type=profile_name,
        label_prefix="register",
        header=f"Tasks: {profile_name} for {ws.path.name}",
        build_cmd_for_roi=(
            lambda roi: (
                f"preprocess register batch {ws.deconved} {roi} --codebook={codebook} --threads={threads}"
            )
        ),
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("spots-optimize", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--rounds", default=10, type=int, help="Number of optimization rounds")
@click.option("--threads", default=None, type=int, help="Number of threads (defaults to cpus from profile)")
@click.option("--config", "json_config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional project config JSON")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def spots_optimize(
    ctx: click.Context,
    path: Path,
    codebook: Path,
    rounds: int,
    threads: int | None,
    json_config: Path | None,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit spots optimize job for a workspace.

    Runs `preprocess spots optimize` on all ROIs.
    """
    ws = Workspace(path)

    # Default threads to profile cpus
    if threads is None:
        threads = 24

    cmd = (
        f"preprocess spots optimize {ws.deconved} '*' --codebook={codebook} --rounds={rounds} "
        f"--threads={threads}"
    )
    if json_config:
        cmd += f" --config={json_config}"

    submit_one(
        ws,
        profile_name="spots-optimize",
        job_type="spots-optimize",
        command=cmd,
        ctx_args=tuple(ctx.args),
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        task_label=f"spots-optimize for {ws.path.name}",
        submit_prompt="Submit job? [y/N] ",
    )


@cli.command("spots-batch", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--threads", default=None, type=int, help="Number of threads (defaults to cpus from profile)")
@click.option("--split", is_flag=True, default=True, help="Split into quadrants")
@click.option("--config", "json_config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional project config JSON")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def spots_batch(
    ctx: click.Context,
    path: Path,
    codebook: Path,
    roi_and_extra: tuple[str, ...],
    threads: int | None,
    split: bool,
    json_config: Path | None,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit spots batch jobs for a workspace, sharded by ROI.

    Runs `preprocess spots batch` on each ROI in the deconvolved directory.
    """
    ws = Workspace(path)

    # Default threads to profile cpus
    if threads is None:
        threads = 24

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    def build_cmd_for_roi(roi_name: str) -> str:
        cmd = f"preprocess spots batch {ws.deconved} {roi_name} --codebook={codebook} --threads={threads}"
        if split:
            cmd += " --split"
        if json_config:
            cmd += f" --config={json_config}"
        return cmd

    submit_per_roi(
        ws,
        profile_name="spots-batch",
        job_type="spots-batch",
        label_prefix="spots-batch",
        header=f"Tasks: spots-batch for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("stitch-register", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--codebook", default=None, help="Codebook name (auto-detected if only one)")
@click.option("--fid/--no-fid", default=False, show_default=True, help="Use fiducial channel for registration")
@click.option(
    "--max-proj/--no-max-proj",
    default=True,
    show_default=True,
    help="Use max projection for registration",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing TileConfiguration.registered.txt")
@click.option("--config", "json_config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional project config JSON")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def stitch_register(
    ctx: click.Context,
    path: Path,
    roi_and_extra: tuple[str, ...],
    codebook: str | None,
    fid: bool,
    max_proj: bool,
    overwrite: bool,
    json_config: Path | None,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit stitch register jobs for a workspace, sharded by ROI.

    Runs `preprocess stitch register` on each ROI in the deconvolved directory.
    """
    ws = Workspace(path)

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    def build_cmd_for_roi(roi: str) -> str:
        cmd = f"preprocess stitch register {ws.deconved} {roi} --debug"
        if max_proj:
            cmd += " --max-proj"
        else:
            cmd += " --idx=0"
        if fid:
            cmd += " --fid"
        if codebook:
            cmd += f" --codebook={codebook}"
        if overwrite:
            cmd += " --overwrite"
        if json_config:
            cmd += f" --config={json_config}"
        return cmd

    submit_per_roi(
        ws,
        profile_name="stitch-register",
        job_type="stitch-register",
        label_prefix="stitch-register",
        header=f"Tasks: stitch-register for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("stitch-fuse", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--codebook", default="pi", help="Codebook name (default: pi)")
@click.option("--config", "json_config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Optional project config JSON")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def stitch_fuse(
    ctx: click.Context,
    path: Path,
    roi_and_extra: tuple[str, ...],
    codebook: str,
    json_config: Path | None,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit stitch fuse jobs for a workspace, sharded by ROI.

    Runs `preprocess stitch fuse` on each ROI in the deconvolved directory.
    """
    ws = Workspace(path)

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    def build_cmd_for_roi(roi: str) -> str:
        cmd = f"preprocess stitch fuse {ws.deconved} {roi} --codebook={codebook} --downsample=2"
        if json_config:
            cmd += f" --config={json_config}"
        return cmd

    submit_per_roi(
        ws,
        profile_name="stitch-fuse",
        job_type="stitch-fuse",
        label_prefix="stitch-fuse",
        header=f"Tasks: stitch-fuse for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("n4", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--codebook", default="pi", help="Codebook name (default: pi)")
@click.option("--z-index", default=5, type=int, help="Z index for n4 correction (default: 5)")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def n4(
    ctx: click.Context,
    path: Path,
    roi_and_extra: tuple[str, ...],
    codebook: str,
    z_index: int,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit N4 bias field correction jobs for a workspace, sharded by ROI.

    Runs `preprocess stitch n4` on each ROI in the deconvolved directory.
    """
    ws = Workspace(path)

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    submit_per_roi(
        ws,
        profile_name="n4",
        job_type="n4",
        label_prefix="n4",
        header=f"Tasks: n4 for {ws.path.name}",
        build_cmd_for_roi=(
            lambda roi: f"preprocess stitch n4 {ws.deconved} {roi} --codebook={codebook} --z-index={z_index}"
        ),
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("spots-stitch", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("codebook", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--threads", default=None, type=int, help="Number of threads (defaults to cpus from profile)")
@click.option("--no-filter", is_flag=True, help="Disable filtering")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def spots_stitch(
    ctx: click.Context,
    path: Path,
    codebook: Path,
    roi_and_extra: tuple[str, ...],
    threads: int | None,
    no_filter: bool,
    overwrite: bool,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit spots stitch jobs for a workspace, sharded by ROI.

    Runs `preprocess spots stitch` on each ROI in the deconvolved directory.
    """
    ws = Workspace(path)

    # Default threads to profile cpus
    if threads is None:
        threads = 8

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    def build_cmd_for_roi(roi_name: str) -> str:
        cmd = f"preprocess spots stitch {ws.deconved} {roi_name} --codebook={codebook} --threads={threads}"
        if no_filter:
            cmd += " --no-filter"
        if overwrite:
            cmd += " --overwrite"
        return cmd

    submit_per_roi(
        ws,
        profile_name="spots-stitch",
        job_type="spots-stitch",
        label_prefix="spots-stitch",
        header=f"Tasks: spots-stitch for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("overlay", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def overlay(
    ctx: click.Context,
    path: Path,
    roi_and_extra: tuple[str, ...],
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Submit overlay jobs for a workspace, sharded by ROI.

    Runs `segment overlay all . <ROI> --threads=32` from the workspace root.
    """
    ws = Workspace(path)
    workspace_root = shlex.quote(str(ws.path))

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    def build_cmd_for_roi(roi: str) -> str:
        return f"cd {workspace_root} && segment overlay all . {shlex.quote(roi)} --threads=16"

    submit_per_roi(
        ws,
        profile_name="overlay",
        job_type="overlay",
        label_prefix="overlay",
        header=f"Tasks: overlay for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


@cli.command("export", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def segment_export(
    ctx: click.Context,
    path: Path,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Submit segment export job for a workspace."""
    ws = Workspace(path)
    submit_one(
        ws,
        profile_name="export",
        job_type="export",
        command=f"segment export {shlex.quote(str(ws.path))}",
        ctx_args=tuple(ctx.args),
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        task_label=f"segment export for {ws.path.name}",
        submit_prompt="Submit job? [y/N] ",
    )


@cli.command("dist-seg", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("codebook", type=str)
@click.argument("roi_and_extra", nargs=-1)
@click.option("--l40s", "use_l40s", is_flag=True, help="Use l40s profile (lower resources)")
@click.option(
    "--n-gpu",
    "n_gpu",
    default=None,
    type=int,
    help="Override GPUs per task; scales cpus/mem linearly from the profile baseline.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing segmentation outputs")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def dist_seg(
    ctx: click.Context,
    path: Path,
    codebook: str,
    roi_and_extra: tuple[str, ...],
    use_l40s: bool,
    n_gpu: int | None,
    overwrite: bool,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit distributed segmentation jobs for a workspace, sharded by ROI."""
    ws = Workspace(path)
    script_path = (
        Path(__file__).resolve().parents[1]
        / "fishtools"
        / "segmentation"
        / "distributed"
        / "distributed_segmentation.py"
    )
    profile_name = "dist-seg-l40s" if use_l40s else "dist-seg"
    workers_per_gpu = 4 if use_l40s else 8

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    if n_gpu is not None and n_gpu < 1:
        raise click.ClickException("--n-gpu must be >= 1")

    def config_transform(config: SlurmConfig) -> None:
        if n_gpu is None:
            return

        base_gpus = config.gpus
        if base_gpus < 1:
            raise click.ClickException(f"Invalid profile gpus={base_gpus} for {profile_name!r}; expected >= 1")

        factor = n_gpu / base_gpus
        config.gpus = n_gpu
        config.cpus = max(
            1,
            int(math.ceil(config.cpus * factor)) if factor >= 1 else int(math.floor(config.cpus * factor)),
        )
        config.mem = _scale_mem(config.mem, factor)

    def build_cmd_for_roi(roi: str) -> str:
        cmd = (
            f"python {script_path} "
            f"run {ws.path} {roi} --workers-per-gpu={workers_per_gpu} --codebook={codebook} -c ~/config.json"
        )
        if overwrite:
            cmd += " --overwrite"
        return cmd

    submit_per_roi(
        ws,
        profile_name=profile_name,
        job_type="dist-seg",
        label_prefix="dist-seg",
        header=f"Tasks: dist-seg for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        config_transform=config_transform,
        rois=[roi] if roi is not None else None,
    )


@cli.command("dist-postproc", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("roi_and_extra", nargs=-1)
@click.option("--overwrite", is_flag=True, help="Overwrite existing postproc outputs")
@click.option(
    "--depends-on",
    default=None,
    help="Comma-separated SLURM job IDs to depend on.",
)
@click.option("--dry-run", is_flag=True, help="Print tasks without submitting")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def dist_postproc(
    ctx: click.Context,
    path: Path,
    roi_and_extra: tuple[str, ...],
    overwrite: bool,
    depends_on: str | None,
    dry_run: bool,
    yes: bool,
):
    """Submit distributed post-processing jobs for a workspace, sharded by ROI."""
    ws = Workspace(path)
    script_path = (
        Path(__file__).resolve().parents[1]
        / "fishtools"
        / "segmentation"
        / "distributed"
        / "distributed_postproc.py"
    )

    roi, extra_args = split_roi_and_extra(roi_and_extra, tuple(ctx.args))

    def build_cmd_for_roi(roi: str) -> str:
        cmd = f"python {script_path} {ws.path} {roi} --workers-per-gpu=10"
        if overwrite:
            cmd += " --overwrite"
        return cmd

    submit_per_roi(
        ws,
        profile_name="dist-postproc",
        job_type="dist-postproc",
        label_prefix="dist-postproc",
        header=f"Tasks: dist-postproc for {ws.path.name}",
        build_cmd_for_roi=build_cmd_for_roi,
        ctx_args=extra_args,
        dry_run=dry_run,
        yes=yes,
        depends_on=depends_on,
        rois=[roi] if roi is not None else None,
    )


if __name__ == "__main__":
    cli()
