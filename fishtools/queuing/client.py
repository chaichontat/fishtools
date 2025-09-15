"""Queue-gated command runner (client for scripts/baseserver.py).

Usage examples
--------------
- Run a command when it reaches the head of the queue:
    python scripts/queuewrap.py -- echo "hello"

- Same, with a fixed ID and custom server settings:
    SCHED_AUTHKEY=change-me \
    python scripts/queuewrap.py --id worker-A --host 127.0.0.1 --port 50001 -- \
      bash -lc 'sleep 2 && echo done'

Behavior
--------
- Submits a unique ID to the shared IPC queue.
- Waits until it is at the head, then atomically pops itself and runs the command.
- On exit (success, failure, Ctrl-C), it removes its queue entry if still present.
"""

from __future__ import annotations

import argparse
import atexit
import getpass
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import AuthenticationError
from multiprocessing.managers import BaseManager
from typing import Any


class QueueManager(BaseManager):
    pass


def gen_default_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    rnd = uuid.uuid4().hex[:8]
    return f"{host}:{pid}:{rnd}"


def connect_manager(host: str, port: int, authkey: bytes) -> Any:
    QueueManager.register("get_queue")
    m = QueueManager(address=(host, port), authkey=authkey)
    m.connect()
    return m


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gate a command behind a shared IPC queue")
    p.add_argument("--id", dest="id", default=os.getenv("QUEUE_ID", gen_default_id()))
    p.add_argument("--host", default=os.getenv("SCHED_HOST", "127.0.0.1"))
    p.add_argument("--port", default=int(os.getenv("SCHED_PORT", "50001")), type=int)
    p.add_argument(
        "--authkey",
        default=os.getenv("SCHED_AUTHKEY", "change-me"),
        help="Auth key shared with the server",
    )
    p.add_argument("--interval", type=float, default=1.0, help="Poll interval while waiting")
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Max seconds to wait for turn (exit 124 on timeout)",
    )
    p.add_argument(
        "--no-shell",
        action="store_true",
        help="Run command without a shell (default: use /bin/bash -lc)",
    )
    p.add_argument("--", dest="dashdash", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    ns = p.parse_args(argv)
    if not ns.command:
        p.error("provide a command after --")

    # Normalize command: argparse keeps the leading '--' token in command; drop it if present
    cmd = ns.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    cmd_str = " ".join(cmd)

    # Connect to manager and queue, with friendly errors if server is unavailable
    try:
        mgr = connect_manager(ns.host, ns.port, ns.authkey.encode())
        q = mgr.get_queue()
    except AuthenticationError:
        print(
            f"[queuewrap] error: authentication failed connecting to {ns.host}:{ns.port}.\n"
            "  - Check SCHED_AUTHKEY/--authkey matches the server.\n"
            "  - Example: SCHED_AUTHKEY=change-me qserver",
            file=sys.stderr,
        )
        return 111
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        msg = getattr(e, "strerror", str(e))
        print(
            f"[queuewrap] error: cannot connect to server at {ns.host}:{ns.port}: {msg}.\n"
            "  - Is the server running? Start it with: qserver \n"
            "  - Or adjust connection with --host/--port/--authkey.",
            file=sys.stderr,
        )
        return 111

    base_id: str = str(ns.id)
    user = os.getenv("QUEUE_USER") or getpass.getuser()
    ident: str = f"{user}:{base_id}"
    placed = False
    removed = False

    def cleanup() -> None:
        # Remove our entry if it still exists
        if not placed:
            return
        try:
            if not removed:
                # Prefer atomic head pop at exit; else remove wherever it is
                if not q.get_if_head(ident):
                    q.remove(ident)
        except Exception:
            pass

    atexit.register(cleanup)

    # Forward SIGINT/SIGTERM to exit promptly (atexit will clean the queue)
    def _sig_handler(_sig, _frame):
        sys.exit(130 if _sig == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Submit our id
    q.put(ident)
    placed = True
    print(f"[queuewrap] queued id={ident}")

    # Heartbeat loop to keep our queue entry alive while we run/wait
    hb_stop = threading.Event()

    def _hb_loop():
        interval = float(os.getenv("QUEUE_HEARTBEAT_INTERVAL", "2"))
        while not hb_stop.is_set():
            try:
                q.heartbeat(ident)
            except Exception:
                pass
            # Use event.wait to allow prompt exit
            hb_stop.wait(interval)

    hb_thread = threading.Thread(target=_hb_loop, name="queuewrap-heartbeat", daemon=True)
    hb_thread.start()

    # Wait until at head, then atomically pop ourselves
    last_report = 0.0
    deadline = (time.time() + float(ns.timeout)) if ns.timeout else None
    while True:
        try:
            # Hold the head while running: do not pop until exit
            if q.peek() == ident:
                print("[queuewrap] acquired turn; running command...")
                break
            # Not yet at head; emit occasional position updates
            now = time.time()
            if deadline is not None and now >= deadline:
                print("[queuewrap] timed out waiting for turn; cleaning up and exiting (124)")
                hb_stop.set()
                hb_thread.join(timeout=1)
                return 124
            if now - last_report >= 5.0:
                try:
                    pos = q.position(ident)
                    size = q.size()
                    if pos >= 0:
                        print(f"[queuewrap] waiting… position={pos} size={size}")
                    else:
                        print("[queuewrap] waiting… (not currently in queue?)")
                except Exception:
                    pass
                last_report = now
            time.sleep(ns.interval)
        except KeyboardInterrupt:
            # Ensure atexit runs cleanup
            raise SystemExit(130)

    # Run the command
    try:
        if ns.no_shell:
            # Exec without shell; pass tokens directly
            proc = subprocess.run(cmd, check=False)
        else:
            # Use bash for shell features (pipes, redirects, env, etc.)
            proc = subprocess.run(cmd_str, shell=True, check=False, executable="/bin/bash")
        # After command finishes, remove our queue entry
        try:
            removed = bool(q.get_if_head(ident)) or bool(q.remove(ident))
        except Exception:
            pass
        # Stop heartbeat thread
        try:
            hb_stop.set()
            hb_thread.join(timeout=1)
        except Exception:
            pass
        return proc.returncode
    finally:
        # Nothing else to pop now; we already popped when starting
        pass


if __name__ == "__main__":
    raise SystemExit(main())
