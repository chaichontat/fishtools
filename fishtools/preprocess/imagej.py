from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from loguru import logger

from fishtools.preprocess.config import StitchingConfig
from fishtools.utils.pretty_print import (
    TaskCancelledException,
    get_cancel_event,
    register_subprocess,
    unregister_subprocess,
)


def run_imagej(
    path: Path,
    *,
    compute_overlap: bool = False,
    fuse: bool = True,
    threshold: float | None = None,
    name: str = "TileConfiguration.txt",
    capture_output: bool = False,
    sc: StitchingConfig | None = None,
) -> None:
    """Execute ImageJ's Grid/Collection stitching macro in headless mode.

    Centralized wrapper so CLIs can reuse consistent defaults and behavior.
    """
    options = "subpixel_accuracy"
    if compute_overlap:
        options += " compute_overlap"
    fusion = "Linear Blending" if fuse else "Do not fuse images (only write TileConfiguration)"

    if not (imagej_path := Path.home() / "Fiji.app/ImageJ-linux64").exists():
        raise FileNotFoundError(
            f"ImageJ not found at {imagej_path}. Please install ImageJ in your home directory."
        )

    max_mem = sc.max_memory_mb if sc else 102400
    par_thr = sc.parallel_threads if sc else 32
    reg_thr = threshold if threshold is not None else (sc.fusion_thresholds.get("regression") if sc else 0.4)
    disp_max = sc.fusion_thresholds.get("displacement_max") if sc else 1.5
    disp_abs = sc.fusion_thresholds.get("displacement_abs") if sc else 2.5

    macro = f"""
    run(\"Memory & Threads...\", \"maximum={max_mem} parallel={par_thr}\");
    run(\"Grid/Collection stitching\", \"type=[Positions from file] \
    order=[Defined by TileConfiguration] directory={path.resolve()} \
    layout_file={name}{"" if compute_overlap else ".registered"}.txt \
    fusion_method=[{fusion}] regression_threshold={reg_thr} \
    max/avg_displacement_threshold={disp_max} absolute_displacement_threshold={disp_abs} {options} \
    computation_parameters=[Save computation time (but use more RAM)] \
    image_output=[Write to disk] \
    output_directory={path.resolve()}\");
    """

    with NamedTemporaryFile("wt") as f:
        f.write(macro)
        f.flush()

        # Build command without shell to control process group reliably
        cmd = [imagej_path.as_posix(), "--headless", "--console", "-macro", f.name]

        creationflags = 0
        preexec_fn = None
        if os.name == "posix":
            # Start new process group so we can signal the entire JVM tree
            preexec_fn = os.setsid  # type: ignore[assignment]
        else:
            # Windows: CREATE_NEW_PROCESS_GROUP allows CTRL_BREAK_EVENT later if needed
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        stdout = subprocess.PIPE if capture_output else None
        stderr = subprocess.STDOUT if capture_output else None

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                text=False,
                preexec_fn=preexec_fn,
                creationflags=creationflags,
            )
        except Exception as e:
            logger.critical(f"Failed to launch ImageJ: {e}")
            raise

        # Register and wait cooperatively so SIGINT can stop the JVM
        register_subprocess(proc)
        cancel = get_cancel_event()
        try:
            # Periodically poll to observe cancellation and KeyboardInterrupt
            while True:
                try:
                    ret = proc.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel.is_set():
                        # Ask JVM to exit; escalate if it ignores us
                        try:
                            if os.name == "posix":
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            else:
                                try:
                                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                                except Exception:
                                    proc.terminate()
                        except Exception:
                            ...
                        # Give it a moment to exit
                        deadline = time.time() + 2.0
                        while time.time() < deadline and proc.poll() is None:
                            time.sleep(0.05)
                        if proc.poll() is None:
                            try:
                                if os.name == "posix":
                                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                                else:
                                    proc.kill()
                            except Exception:
                                ...
                        raise TaskCancelledException("ImageJ cancelled by SIGINT")
                except KeyboardInterrupt:
                    # Ensure child is torn down when user interrupts here
                    try:
                        if os.name == "posix":
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        else:
                            try:
                                proc.send_signal(signal.CTRL_BREAK_EVENT)
                            except Exception:
                                proc.terminate()
                    finally:
                        raise

            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
        finally:
            unregister_subprocess(proc)
