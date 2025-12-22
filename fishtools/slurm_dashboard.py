from __future__ import annotations

import curses
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, Sequence

import rich_click as click
from loguru import logger

DEFAULT_SQUEUE_FORMAT = "%i|%j|%T|%M|%l|%P|%D|%C|%b|%m|%R|%o|%e"
DEFAULT_SQUEUE_FORMAT_WITH_USER = f"%u|{DEFAULT_SQUEUE_FORMAT}"
DEFAULT_SACCT_FIELDS = [
    "JobIDRaw",
    "User",
    "JobName",
    "State",
    "Elapsed",
    "Timelimit",
    "Partition",
    "ReqNodes",
    "ReqCPUS",
    "ReqTRES",
    "ReqMem",
    "NodeList",
    "WorkDir",
    "StdOut",
    "StdErr",
    "End",
]
RUNNING_STATES = {"RUNNING", "COMPLETING"}
TERMINAL_PREFIXES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


class CommandRunner(Protocol):
    def __call__(self, args: Sequence[str]) -> str: ...


def _run_command(args: Sequence[str]) -> str:
    proc = subprocess.run(list(args), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _sanitize_error_message(message: str) -> str:
    # Keep curses stable: single line, no control chars.
    cleaned = " ".join(message.split())
    return cleaned


def _expand_slurm_path(
    path_str: str,
    job_id: str,
    *,
    job_name: str | None = None,
    user: str | None = None,
) -> str:
    if not path_str:
        return path_str
    base_id, _, array_id = job_id.partition("_")
    expanded = (
        path_str.replace("%j", job_id)
        .replace("%A", base_id)
        .replace("%a", array_id or "")
        .replace("%u", user or os.environ.get("USER", ""))
        .replace("%x", job_name or "")
    )
    expanded = os.path.expandvars(expanded)
    expanded = os.path.expanduser(expanded)
    return expanded


def _resolve_relative(
    path_str: str,
    workdir: str | None,
    job_id: str,
    *,
    job_name: str | None = None,
    user: str | None = None,
) -> str:
    if not path_str:
        return path_str
    expanded = _expand_slurm_path(path_str, job_id, job_name=job_name, user=user)
    p = Path(expanded)
    if p.is_absolute() or workdir is None:
        return expanded
    return str(Path(workdir) / p)


def _parse_gpus(gres: str) -> str:
    if not gres:
        return ""
    for pat in (
        r"gres/gpu(?::[^:,=]+)?[:=](\d+)",
        r"gpu(?::[^:,=]+)?[:=](\d+)",
        r"gpu=(\d+)",
    ):
        m = re.search(pat, gres)
        if m:
            return m.group(1)
    return ""


def _parse_slurm_duration(duration: str) -> timedelta | None:
    duration = duration.strip()
    if not duration or duration in {"N/A", "UNLIMITED", "UNLIMITED*", "Unknown", "None"}:
        return None

    days = 0
    if "-" in duration:
        day_str, rest = duration.split("-", 1)
        if not day_str.isdigit():
            return None
        days = int(day_str)
        duration = rest

    parts = duration.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(p) for p in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(p) for p in parts)
        elif len(parts) == 1:
            hours = 0
            minutes = int(parts[0])
            seconds = 0
        else:
            return None
    except ValueError:
        return None

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _format_iso8601_local(time_str: str) -> str:
    try:
        dt = datetime.fromisoformat(time_str)
    except ValueError:
        return time_str
    return dt.replace(microsecond=0).isoformat(timespec="seconds")


def _parse_time_epoch(time_str: str | None) -> float:
    if not time_str or time_str in {"Unknown", "None"}:
        return 0.0
    try:
        return datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return 0.0


def _search_home_logs(job_id: str, *, logs_dir: Path | None = None) -> tuple[str | None, str | None]:
    logs_root = logs_dir or (Path.home() / "logs")
    if not logs_root.exists():
        return None, None

    ids: list[str] = [job_id]
    base_id = job_id.split("_", 1)[0]
    if base_id != job_id:
        ids.append(base_id)

    matches: list[Path] = []
    for candidate in ids:
        matches.extend(logs_root.glob(f"{candidate}*"))
    if not matches:
        for candidate in ids:
            matches.extend(logs_root.glob(f"*{candidate}*"))

    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None, None

    lower_names = {p: p.name.lower() for p in matches}
    stdout_candidates = [
        p for p in matches if ("out" in lower_names[p] or "stdout" in lower_names[p]) and "err" not in lower_names[p]
    ]
    stderr_candidates = [p for p in matches if ("err" in lower_names[p] or "stderr" in lower_names[p])]

    if not stdout_candidates:
        stdout_candidates = [p for p in matches if p.suffix in {".out", ".log"} and p.suffix != ".err"]
    if not stderr_candidates:
        stderr_candidates = [p for p in matches if p.suffix == ".err"]

    def _pick(cands: list[Path]) -> str | None:
        if not cands:
            return None
        best = max(cands, key=lambda p: p.stat().st_mtime)
        return str(best)

    stdout_path = _pick(stdout_candidates)
    stderr_path = _pick(stderr_candidates)
    if stdout_path is None and stderr_path is None:
        stdout_path = _pick(matches)
    return stdout_path, stderr_path


@dataclass(frozen=True)
class JobInfo:
    job_id: str
    user: str | None
    name: str
    state: str
    elapsed: str
    time_limit: str
    partition: str
    nodes: str
    cpus: str
    gres: str
    mem: str
    reason_or_node: str
    stdout_path: str
    stderr_path: str
    end_time: str | None = None
    workdir: str | None = None

    @property
    def is_running(self) -> bool:
        return self.state in RUNNING_STATES

    @property
    def gpus_requested(self) -> str:
        return _parse_gpus(self.gres)

    @property
    def end_epoch(self) -> float:
        return _parse_time_epoch(self.end_time)


def parse_squeue_line(line: str) -> JobInfo:
    parts = line.split("|")
    user: str | None = None
    if len(parts) == 14:
        user = parts[0].strip()
        parts = parts[1:]
    if len(parts) != 13:
        raise ValueError(f"Unexpected squeue line with {len(parts)} fields: {line}")
    (
        job_id,
        name,
        state,
        elapsed,
        time_limit,
        partition,
        nodes,
        cpus,
        gres,
        mem,
        reason,
        stdout_path,
        stderr_path,
    ) = parts
    return JobInfo(
        job_id=job_id.strip(),
        user=user,
        name=name.strip(),
        state=state.strip(),
        elapsed=elapsed.strip(),
        time_limit=time_limit.strip(),
        partition=partition.strip(),
        nodes=nodes.strip(),
        cpus=cpus.strip(),
        gres=gres.strip(),
        mem=mem.strip(),
        reason_or_node=reason.strip(),
        stdout_path=stdout_path.strip(),
        stderr_path=stderr_path.strip(),
    )


def parse_sacct_line(line: str) -> JobInfo:
    parts = line.split("|")
    if len(parts) != len(DEFAULT_SACCT_FIELDS):
        raise ValueError(f"Unexpected sacct line with {len(parts)} fields: {line}")

    (
        job_id,
        user,
        name,
        state,
        elapsed,
        time_limit,
        partition,
        nodes,
        cpus,
        tres,
        mem,
        nodelist,
        workdir,
        stdout_path,
        stderr_path,
        end_time,
    ) = parts

    end_time_val = end_time.strip()
    if end_time_val in {"", "Unknown", "None", "N/A"}:
        end_time_val = ""

    return JobInfo(
        job_id=job_id.strip(),
        user=user.strip() or None,
        name=name.strip(),
        state=state.strip(),
        elapsed=elapsed.strip(),
        time_limit=time_limit.strip(),
        partition=partition.strip(),
        nodes=nodes.strip(),
        cpus=cpus.strip(),
        gres=tres.strip(),
        mem=mem.strip(),
        reason_or_node=nodelist.strip(),
        stdout_path=_resolve_relative(
            stdout_path.strip(),
            workdir.strip() or None,
            job_id.strip(),
            job_name=name.strip() or None,
            user=user.strip() or None,
        ),
        stderr_path=_resolve_relative(
            stderr_path.strip(),
            workdir.strip() or None,
            job_id.strip(),
            job_name=name.strip() or None,
            user=user.strip() or None,
        ),
        end_time=end_time_val or None,
        workdir=workdir.strip() or None,
    )


class SlurmClient:
    def __init__(
        self,
        *,
        user: str | None = None,
        runner: CommandRunner = _run_command,
        squeue_format: str = DEFAULT_SQUEUE_FORMAT,
    ) -> None:
        self.user = user or self._infer_default_user()
        if not self.user:
            raise ValueError("Could not determine SLURM user; set --user or $USER.")
        self.runner = runner
        self.squeue_format = squeue_format
        self.user_tokens = self._infer_user_tokens(self.user)
        self._use_squeue_user_filter = True
        self._use_sacct_user_filter = True

    def fetch_jobs(self) -> list[JobInfo]:
        fmt = f"%u|{self.squeue_format}"
        if self._use_squeue_user_filter:
            try:
                out = self.runner(["squeue", "-u", self.user, "-h", "-o", fmt])
                filter_after = False
            except RuntimeError:
                self._use_squeue_user_filter = False
                out = self.runner(["squeue", "-h", "-o", fmt])
                filter_after = True
        else:
            out = self.runner(["squeue", "-h", "-o", fmt])
            filter_after = True
        jobs: list[JobInfo] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            job = parse_squeue_line(line)
            if filter_after and job.user not in self.user_tokens:
                continue
            jobs.append(job)
        return jobs

    def fetch_recent_completed(self, days: int = 7) -> list[JobInfo]:
        fields = ",".join(DEFAULT_SACCT_FIELDS)
        cmd_base = [
            "sacct",
            "--starttime",
            f"now-{days}days",
            "-X",
            "-n",
            "-P",
            "-o",
            fields,
        ]
        if self._use_sacct_user_filter:
            try:
                out = self.runner([*cmd_base, "-u", self.user])
                filter_after = False
            except RuntimeError:
                self._use_sacct_user_filter = False
                out = self.runner(cmd_base)
                filter_after = True
        else:
            out = self.runner(cmd_base)
            filter_after = True
        jobs: list[JobInfo] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                job = parse_sacct_line(line)
            except ValueError:
                continue
            if filter_after and job.user not in self.user_tokens:
                continue
            state_prefix = re.split(r"[+ ]", job.state, maxsplit=1)[0]
            if state_prefix in TERMINAL_PREFIXES:
                jobs.append(job)
        jobs.sort(key=lambda j: j.end_epoch, reverse=True)
        return jobs

    @staticmethod
    def _infer_default_user() -> str | None:
        for key in ("USER", "LOGNAME"):
            val = os.environ.get(key)
            if val and not val.isdigit():
                return val
        home = os.environ.get("HOME")
        if home:
            candidate = Path(home).name
            if candidate and not candidate.isdigit():
                return candidate
        return None

    @staticmethod
    def _infer_user_tokens(user: str) -> set[str]:
        tokens: set[str] = {user}
        for key in ("USER", "LOGNAME"):
            val = os.environ.get(key)
            if val:
                tokens.add(val)
        home = os.environ.get("HOME")
        if home:
            tokens.add(Path(home).name)
        tokens.discard("")
        return tokens

    def get_job_details(self, job_id: str) -> dict[str, str]:
        out = self.runner(["scontrol", "show", "job", "-dd", job_id])
        details: dict[str, str] = {}
        for match in re.finditer(r"(\w+)=([^\s]+)", out):
            details[match.group(1)] = match.group(2)
        return details

    def resolve_log_paths(self, job: JobInfo) -> JobInfo:
        stdout_expanded = _expand_slurm_path(job.stdout_path, job.job_id, job_name=job.name, user=job.user)
        stderr_expanded = _expand_slurm_path(job.stderr_path, job.job_id, job_name=job.name, user=job.user)

        stdout_is_abs = bool(stdout_expanded) and Path(stdout_expanded).is_absolute()
        stderr_is_abs = bool(stderr_expanded) and Path(stderr_expanded).is_absolute()

        # For non-running jobs, avoid scontrol (may fail for PENDING/finished).
        if not job.is_running:
            stdout_resolved = _resolve_relative(
                stdout_expanded, job.workdir, job.job_id, job_name=job.name, user=job.user
            )
            stderr_resolved = _resolve_relative(
                stderr_expanded, job.workdir, job.job_id, job_name=job.name, user=job.user
            )
            state_prefix = re.split(r"[+ ]", job.state, maxsplit=1)[0]
            if state_prefix in TERMINAL_PREFIXES:
                stdout_ok = bool(stdout_resolved) and Path(stdout_resolved).exists()
                stderr_ok = bool(stderr_resolved) and Path(stderr_resolved).exists()
                if not stdout_ok or not stderr_ok:
                    found_out, found_err = _search_home_logs(job.job_id)
                    if found_out and not stdout_ok:
                        stdout_resolved = found_out
                    if found_err and not stderr_ok:
                        stderr_resolved = found_err
            return replace(job, stdout_path=stdout_resolved, stderr_path=stderr_resolved)

        needs_details = any(
            (
                not stdout_expanded,
                not stderr_expanded,
                not stdout_is_abs,
                not stderr_is_abs,
                job.workdir is None,
            )
        )
        if not needs_details:
            return replace(job, stdout_path=stdout_expanded, stderr_path=stderr_expanded)

        details = self.get_job_details(job.job_id)
        workdir = details.get("WorkDir") or job.workdir
        stdout_path = details.get("StdOut", stdout_expanded)
        stderr_path = details.get("StdErr", stderr_expanded)

        stdout_resolved = _resolve_relative(stdout_path, workdir, job.job_id, job_name=job.name, user=job.user)
        stderr_resolved = _resolve_relative(stderr_path, workdir, job.job_id, job_name=job.name, user=job.user)
        return replace(job, stdout_path=stdout_resolved, stderr_path=stderr_resolved, workdir=workdir)


@dataclass
class FileTailer:
    path: Path | None
    pos: int = 0
    inode: tuple[int, int] | None = None
    partial: str = ""

    def set_path(self, path: Path | None, *, prefill_lines: int = 200) -> list[str]:
        self.path = path
        self.pos = 0
        self.inode = None
        self.partial = ""
        return self.tail(prefill_lines)

    def tail(self, n_lines: int) -> list[str]:
        if self.path is None or not self.path.exists():
            return []
        dq: deque[str] = deque(maxlen=n_lines)
        with self.path.open("r", errors="replace") as f:
            for line in f:
                dq.append(line.rstrip("\n"))
        stat = self.path.stat()
        self.inode = (stat.st_ino, stat.st_dev)
        self.pos = stat.st_size
        return list(dq)

    def read_new(self) -> list[str]:
        if self.path is None or not self.path.exists():
            return []
        stat = self.path.stat()
        inode = (stat.st_ino, stat.st_dev)
        if self.inode is None or inode != self.inode or stat.st_size < self.pos:
            self.inode = inode
            self.pos = 0
            self.partial = ""

        with self.path.open("r", errors="replace") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()

        if not data:
            return []

        data = self.partial + data
        chunks = data.splitlines(keepends=True)
        complete: list[str] = []
        self.partial = ""
        for chunk in chunks:
            if chunk.endswith("\n"):
                complete.append(chunk.rstrip("\n"))
            else:
                self.partial = chunk
        return complete


@dataclass(frozen=True)
class LogEvent:
    stream: str  # "out" or "err"
    line: str


class CombinedLogTailer:
    def __init__(self, *, max_events: int = 5000, prefill_lines: int = 200) -> None:
        self.out_tailer = FileTailer(None)
        self.err_tailer = FileTailer(None)
        self.events: deque[LogEvent] = deque(maxlen=max_events)
        self.prefill_lines = prefill_lines

    def set_paths(self, stdout_path: str, stderr_path: str) -> None:
        self.events.clear()
        out_lines = self.out_tailer.set_path(Path(stdout_path) if stdout_path else None, prefill_lines=self.prefill_lines)
        err_lines = self.err_tailer.set_path(Path(stderr_path) if stderr_path else None, prefill_lines=self.prefill_lines)
        for line in out_lines:
            self.events.append(LogEvent("out", line))
        for line in err_lines:
            self.events.append(LogEvent("err", line))

    def update(self) -> None:
        for line in self.out_tailer.read_new():
            self.events.append(LogEvent("out", line))
        for line in self.err_tailer.read_new():
            self.events.append(LogEvent("err", line))

    def view(self, mode: str) -> list[LogEvent]:
        if mode == "out":
            return [ev for ev in self.events if ev.stream == "out"]
        if mode == "err":
            return [ev for ev in self.events if ev.stream == "err"]
        return list(self.events)


class SlurmDashboard:
    def __init__(
        self,
        stdscr: curses.window,
        client: SlurmClient,
        *,
        jobs_refresh: float = 2.0,
        logs_refresh: float = 1.0,
        max_log_events: int = 5000,
    ) -> None:
        self.stdscr = stdscr
        self.client = client
        self.jobs_refresh = jobs_refresh
        self.logs_refresh = logs_refresh
        self.running_filter = True
        self.other_filter = True
        self.log_mode = "both"
        self.follow_logs = True

        self.jobs: list[JobInfo] = []
        self.filtered_jobs: list[JobInfo] = []
        self.selected_index = 0
        self.selected_job_id: str | None = None
        self.current_job: JobInfo | None = None

        self.job_scroll = 0
        self.log_top = 0
        self.tailer = CombinedLogTailer(max_events=max_log_events)
        self.last_jobs_fetch = 0.0
        self.last_logs_fetch = 0.0
        self.error_message: str | None = None

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # stdout
        curses.init_pair(2, curses.COLOR_RED, -1)  # stderr
        curses.init_pair(3, curses.COLOR_CYAN, -1)  # header

    def run(self) -> None:
        while True:
            now = time.monotonic()
            if now - self.last_jobs_fetch >= self.jobs_refresh:
                self.refresh_jobs()
                self.last_jobs_fetch = now
            if self.current_job and now - self.last_logs_fetch >= self.logs_refresh:
                self.refresh_logs()
                self.last_logs_fetch = now
            self.draw()
            key = self.stdscr.getch()
            if key != -1:
                if not self.handle_key(key):
                    return
            time.sleep(0.05)

    def refresh_jobs(self) -> None:
        errors: list[str] = []
        current: list[JobInfo] = []
        recent: list[JobInfo] = []

        try:
            current = self.client.fetch_jobs()
        except Exception as e:  # expected boundary error
            msg = _sanitize_error_message(f"squeue failed: {e}")
            errors.append(msg)
            logger.error(msg)

        if (not self.running_filter) and self.other_filter:
            try:
                recent = self.client.fetch_recent_completed(days=7)
            except Exception as e:  # expected boundary error
                msg = _sanitize_error_message(f"sacct failed: {e}")
                errors.append(msg)
                logger.error(msg)

        recent_ids = {job.job_id for job in recent}
        jobs = [*recent, *[job for job in current if job.job_id not in recent_ids]]

        self.error_message = "; ".join(errors) if errors else None
        self.jobs = jobs
        self.apply_filters()

    def apply_filters(self) -> None:
        filtered: list[JobInfo] = []
        for job in self.jobs:
            if job.is_running and self.running_filter:
                filtered.append(job)
            elif (not job.is_running) and self.other_filter:
                filtered.append(job)
        self.filtered_jobs = filtered

        if self.selected_job_id:
            for i, job in enumerate(self.filtered_jobs):
                if job.job_id == self.selected_job_id:
                    self.selected_index = i
                    break
            else:
                self.selected_index = 0
                self.selected_job_id = None

        self.ensure_selection_valid()

    def ensure_selection_valid(self) -> None:
        if not self.filtered_jobs:
            self.selected_index = 0
            self.current_job = None
            return
        self.selected_index = max(0, min(self.selected_index, len(self.filtered_jobs) - 1))
        job = self.filtered_jobs[self.selected_index]
        if job.job_id != self.selected_job_id:
            self.select_job(job)

    def select_job(self, job: JobInfo) -> None:
        try:
            resolved = self.client.resolve_log_paths(job)
        except Exception as e:
            self.error_message = _sanitize_error_message(
                f"Failed to resolve logs for job {job.job_id}: {e}"
            )
            logger.error(self.error_message)
            resolved = job

        self.selected_job_id = resolved.job_id
        self.current_job = resolved
        self.tailer.set_paths(resolved.stdout_path, resolved.stderr_path)
        self.follow_logs = resolved.is_running
        self.log_top = 0
        self.last_logs_fetch = 0.0

    def refresh_logs(self) -> None:
        self.tailer.update()
        if self.follow_logs:
            self.log_top = self._max_log_top()

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False

        if key == curses.KEY_UP:
            self.move_selection(-1)
        elif key == curses.KEY_DOWN:
            self.move_selection(1)
        elif key == ord("r"):
            self.running_filter = not self.running_filter
            self.last_jobs_fetch = 0.0
            self.refresh_jobs()
        elif key == ord("n"):
            self.other_filter = not self.other_filter
            self.last_jobs_fetch = 0.0
            self.refresh_jobs()
        elif key == ord("o"):
            self.log_mode = "out"
            self.follow_logs = True
        elif key == ord("e"):
            self.log_mode = "err"
            self.follow_logs = True
        elif key == ord("b"):
            self.log_mode = "both"
            self.follow_logs = True
        elif key in (curses.KEY_PPAGE,):
            self.scroll_logs(-self._log_scroll_amount())
        elif key in (curses.KEY_NPAGE,):
            self.scroll_logs(self._log_scroll_amount())
        elif key == curses.KEY_END:
            self.follow_logs = True
            self.log_top = self._max_log_top()
        elif key in (curses.KEY_ENTER, ord("\n")):
            self.follow_logs = not self.follow_logs
            if self.follow_logs:
                self.log_top = self._max_log_top()
        return True

    def move_selection(self, delta: int) -> None:
        if not self.filtered_jobs:
            return
        new_index = max(0, min(self.selected_index + delta, len(self.filtered_jobs) - 1))
        if new_index != self.selected_index:
            self.selected_index = new_index
            self.ensure_selection_valid()
        self.ensure_job_scroll_visible()

    def ensure_job_scroll_visible(self) -> None:
        h, _w = self.stdscr.getmaxyx()
        list_height = max(1, h - 2)
        if self.selected_index < self.job_scroll:
            self.job_scroll = self.selected_index
        elif self.selected_index >= self.job_scroll + list_height:
            self.job_scroll = self.selected_index - list_height + 1

    def scroll_logs(self, delta: int) -> None:
        max_top = self._max_log_top()
        self.log_top = max(0, min(self.log_top + delta, max_top))
        self.follow_logs = self.log_top >= max_top

    def _log_lines(self) -> list[LogEvent]:
        return self.tailer.view(self.log_mode)

    def _log_scroll_amount(self) -> int:
        h, _w = self.stdscr.getmaxyx()
        visible = max(1, h - 3)
        return max(1, visible // 2)

    def _max_log_top(self) -> int:
        h, _w = self.stdscr.getmaxyx()
        visible = max(1, h - 3)
        lines = self._log_lines()
        return max(0, len(lines) - visible)

    def draw(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        left_w = max(50, int(w * 0.45))
        right_w = w - left_w - 1

        left_win = self.stdscr.derwin(h, left_w, 0, 0)
        right_win = self.stdscr.derwin(h, right_w, 0, left_w + 1)

        for y in range(h):
            try:
                self.stdscr.addch(y, left_w, "|")
            except curses.error:
                pass

        self.draw_left(left_win)
        self.draw_right(right_win)
        self.stdscr.refresh()

    def draw_left(self, win: curses.window) -> None:
        win.erase()
        h, w = win.getmaxyx()

        filters_line = (
            f"[{'x' if self.running_filter else ' '}] Running (r)  "
            f"[{'x' if self.other_filter else ' '}] Other (n)"
        )
        if (not self.running_filter) and self.other_filter:
            filters_line += "  (history 7d)"
        win.addnstr(0, 0, filters_line, w - 1, curses.color_pair(3))
        win.hline(1, 0, "-", w - 1)

        if not self.filtered_jobs:
            win.addnstr(2, 0, "No jobs match filter.", w - 1)
            win.noutrefresh()
            return

        list_height = h - 2
        start = self.job_scroll
        for i, job in enumerate(self.filtered_jobs[start : start + list_height]):
            row_index = start + i
            selected = row_index == self.selected_index
            style = curses.A_REVERSE if selected else curses.A_NORMAL
            line = self._format_job_line(job, w - 1)
            try:
                win.addnstr(2 + i, 0, line, w - 1, style)
            except curses.error:
                pass
        win.noutrefresh()

    def _format_job_line(self, job: JobInfo, width: int) -> str:
        job_id = job.job_id
        if len(job_id) > 8:
            job_id = job_id[:7] + "…"

        state_prefix = re.split(r"[+ ]", job.state, maxsplit=1)[0]
        state_map = {
            "RUNNING": "R",
            "COMPLETING": "R+",
            "PENDING": "PD",
            "COMPLETED": "OK",
            "FAILED": "FL",
            "CANCELLED": "CA",
            "TIMEOUT": "TO",
            "OUT_OF_MEMORY": "OM",
            "NODE_FAIL": "NF",
            "PREEMPTED": "PR",
            "BOOT_FAIL": "BF",
            "DEADLINE": "DL",
            "REVOKED": "RV",
        }
        state = state_map.get(state_prefix, state_prefix[:2] or "--")

        part = (job.partition or "-")[:6].ljust(6)
        gpus = job.gpus_requested or "-"
        gpu_token = f"G{gpus}"
        if len(gpu_token) > 4:
            gpu_token = gpu_token[:4]
        gpu_token = gpu_token.ljust(4)

        end_time = "-"
        if job.end_time:
            end_time = _format_iso8601_local(job.end_time)
        elif job.is_running:
            elapsed = _parse_slurm_duration(job.elapsed)
            limit = _parse_slurm_duration(job.time_limit)
            if elapsed is not None and limit is not None:
                remaining = limit - elapsed
                if remaining.total_seconds() < 0:
                    remaining = timedelta(0)
                end_time = (datetime.now() + remaining).replace(microsecond=0).isoformat(timespec="seconds")
        end_time = end_time[:19].ljust(19)

        prefix = f"{job_id:>8} {state:<2} {part} {gpu_token} {end_time} "
        remaining = max(0, width - len(prefix))
        name = job.name
        if remaining > 0:
            if len(name) > remaining:
                if remaining <= 1:
                    name = name[:remaining]
                else:
                    name = name[: remaining - 1] + "…"
            line = prefix + name
        else:
            line = prefix.rstrip()

        if len(line) > width:
            return line[: max(0, width - 1)]
        return line.ljust(width)

    def draw_right(self, win: curses.window) -> None:
        win.erase()
        h, w = win.getmaxyx()

        job_header = "No job selected."
        if self.current_job:
            job_header = f"Job {self.current_job.job_id} {self.current_job.name} [{self.current_job.state}]"
        win.addnstr(0, 0, job_header, w - 1, curses.A_BOLD)

        controls = f"Logs: out(o)/err(e)/both(b) | Follow: {'ON' if self.follow_logs else 'OFF'} (Enter) | PgUp/PgDn scroll"
        win.addnstr(1, 0, controls, w - 1)
        win.hline(2, 0, "-", w - 1)

        if self.error_message:
            win.addnstr(3, 0, _sanitize_error_message(self.error_message), w - 1, curses.color_pair(2))

        lines = self._log_lines()
        visible_h = h - 3
        max_top = max(0, len(lines) - visible_h)
        self.log_top = max(0, min(self.log_top, max_top))

        for i, ev in enumerate(lines[self.log_top : self.log_top + visible_h]):
            prefix = ""
            color = curses.A_NORMAL
            if self.log_mode == "both":
                if ev.stream == "out":
                    prefix = "[OUT] "
                    color = curses.color_pair(1)
                else:
                    prefix = "[ERR] "
                    color = curses.color_pair(2)
            text = f"{prefix}{ev.line}"
            try:
                win.addnstr(3 + i, 0, text, w - 1, color)
            except curses.error:
                pass

        win.noutrefresh()


def run_dashboard(
    *,
    user: str | None = None,
    jobs_refresh: float = 2.0,
    logs_refresh: float = 1.0,
    max_log_events: int = 5000,
) -> None:
    client = SlurmClient(user=user)

    def _wrapped(stdscr: curses.window) -> None:
        SlurmDashboard(
            stdscr,
            client,
            jobs_refresh=jobs_refresh,
            logs_refresh=logs_refresh,
            max_log_events=max_log_events,
        ).run()

    curses.wrapper(_wrapped)


@click.command()
@click.option("--user", default=None, help="SLURM user to query (default: $USER).")
@click.option("--jobs-refresh", default=2.0, type=float, show_default=True, help="Seconds between squeue polls.")
@click.option("--logs-refresh", default=1.0, type=float, show_default=True, help="Seconds between log tail polls.")
@click.option("--max-log-events", default=5000, type=int, show_default=True, help="Max lines kept in log buffer.")
def main(user: str | None, jobs_refresh: float, logs_refresh: float, max_log_events: int) -> None:
    """Two-pane SLURM TUI: job list + live stdout/stderr tail."""
    run_dashboard(
        user=user,
        jobs_refresh=jobs_refresh,
        logs_refresh=logs_refresh,
        max_log_events=max_log_events,
    )


if __name__ == "__main__":
    main()
