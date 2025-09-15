import os
import getpass
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from multiprocessing.managers import BaseManager
from typing import Any

import pytest

# Skip the entire module if sockets cannot be created in this environment
try:
    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _s.close()
except PermissionError:  # pragma: no cover - environment specific
    pytest.skip("Skipping queuewrap tests: socket creation not permitted in sandbox.", allow_module_level=True)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class QueueManager(BaseManager):
    pass


def _connect_mgr(host: str, port: int, authkey: bytes):
    QueueManager.register("get_queue")
    m = QueueManager(address=(host, port), authkey=authkey)
    m.connect()
    return m


# State verification utilities for faster, more reliable tests
def wait_until_queued(q: Any, item_id: str, timeout: float = 2.0) -> bool:
    """Wait until item appears in queue at any position."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if q.position(item_id) >= 0:
            return True
        time.sleep(0.01)  # Tight polling loop
    raise TimeoutError(f"{item_id} not queued within {timeout}s")


def wait_until_position(q: Any, item_id: str, expected_pos: int, timeout: float = 2.0) -> bool:
    """Wait until item reaches specific position in queue."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if q.position(item_id) == expected_pos:
            return True
        time.sleep(0.01)
    raise TimeoutError(f"{item_id} not at position {expected_pos} within {timeout}s")


def wait_until_removed(q: Any, item_id: str, timeout: float = 2.0) -> bool:
    """Wait until item is removed from queue."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if q.position(item_id) == -1:
            return True
        time.sleep(0.01)
    raise TimeoutError(f"{item_id} not removed within {timeout}s")


def wait_for_output(proc: subprocess.Popen, expected_text: str, timeout: float = 2.0) -> bool:
    """Wait for specific output from a subprocess."""
    import select
    deadline = time.time() + timeout
    buffer = ""

    # Make stdout non-blocking for efficient polling
    import fcntl
    import os as os_module
    flags = fcntl.fcntl(proc.stdout.fileno(), fcntl.F_GETFL)
    fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, flags | os_module.O_NONBLOCK)

    while time.time() < deadline:
        # Use select to check if data is available
        ready, _, _ = select.select([proc.stdout], [], [], 0.01)
        if ready:
            try:
                chunk = proc.stdout.read(1024)
                if chunk:
                    buffer += chunk
                    if expected_text in buffer:
                        return True
            except:
                pass
        time.sleep(0.001)  # Very short sleep
    raise TimeoutError(f"'{expected_text}' not found in output within {timeout}s")


def verify_queue_state(q: Any, expected_items: list) -> None:
    """Verify queue contains exactly the expected items in order."""
    actual = q.list()
    assert actual == expected_items, f"Queue state mismatch: {actual} != {expected_items}"


@pytest.fixture(scope="module")
def server():
    host = "127.0.0.1"
    port = _free_port()
    authkey = f"pytest-{os.getpid()}".encode()

    env = os.environ.copy()
    env.update({
        "SCHED_HOST": host,
        "SCHED_PORT": str(port),
        "SCHED_AUTHKEY": authkey.decode(),
        "SCHED_LOG_LEVEL": "WARNING",
        "SCHED_HEARTBEAT_TIMEOUT": "1",  # Reduced from 5s - still safe for tests
        "SCHED_SWEEP_INTERVAL": "0.1",  # Reduced from 0.2s - 10Hz is plenty
    })

    proc = subprocess.Popen(
        [sys.executable, "-m", "fishtools.queuing.server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait until the server is ready to accept connections
    deadline = time.time() + 2.0  # Reduced from 5s
    last_err = None
    while time.time() < deadline:
        try:
            m = _connect_mgr(host, port, authkey)
            q = m.get_queue()
            # Touch the queue to confirm proxy works
            _ = q.size()
            break
        except Exception as e:  # pragma: no cover - only on startup flake
            last_err = e
            time.sleep(0.01)  # Reduced from 0.05s for faster startup detection
    else:  # pragma: no cover
        proc.terminate()
        out, err = proc.communicate(timeout=1)
        raise RuntimeError(f"server failed to start: {last_err}\nSTDOUT:{out}\nSTDERR:{err}")

    try:
        yield {"host": host, "port": port, "authkey": authkey}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)  # Reduced from 2s
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


@pytest.fixture(autouse=True)
def clear_queue(server):
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    q.clear()
    yield
    q.clear()


def _run_client(server, args, timeout=3):  # Reduced default timeout from 5s
    env = os.environ.copy()
    env.update(
        {
            "SCHED_HOST": server["host"],
            "SCHED_PORT": str(server["port"]),
            "SCHED_AUTHKEY": server["authkey"].decode(),
        }
    )
    p = subprocess.Popen(
        [sys.executable, "-m", "fishtools.queuing.client", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        raise
    return p.returncode, out


def _python_one_liner(label: str, log_path: str, sleep_s: float) -> list[str]:
    code = (
        "import time, os, pathlib; "
        f"p=pathlib.Path(r'{log_path}'); p.parent.mkdir(parents=True,exist_ok=True); "
        "f=p.open('a'); "
        f"f.write(f'{{time.time()}} START {label}\\n'); f.flush(); "
        f"time.sleep({sleep_s}); f.write(f'{{time.time()}} END {label}\\n'); f.flush(); "
        "f.close()"
    )
    return [sys.executable, "-c", code]


def test_single_client_runs_and_clears(server):
    user = getpass.getuser()
    ret, out = _run_client(
        server,
        [
            "--id",
            "A",
            "--timeout",
            "2",  # Reduced from 5s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "print('A-start')",
        ],
        timeout=2,  # Reduced from default 3s
    )
    assert ret == 0
    assert f"[queuewrap] queued id={user}:A" in out
    assert "[queuewrap] acquired turn" in out

    # Explicit state verification instead of relying on output
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    assert q.position(f"{user}:A") == -1
    assert q.size() == 0


def test_waits_until_anchor_removed_then_runs(server):
    user = getpass.getuser()
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    q.put("ANCHOR")

    env = os.environ.copy()
    env.update(
        {
            "SCHED_HOST": server["host"],
            "SCHED_PORT": str(server["port"]),
            "SCHED_AUTHKEY": server["authkey"].decode(),
        }
    )

    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "C",
            "--interval",
            "0.01",  # Reduced from 0.05s for faster polling
            "--timeout",
            "2",  # Reduced from 5s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "print('C-run')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Use state verification instead of time-based wait
    wait_until_position(q, f"{user}:C", 1, timeout=2.0)  # Reduced from 3s

    # Remove anchor so C can acquire and run
    q.remove("ANCHOR")
    out, _ = p.communicate(timeout=3)  # Reduced from 5s with safety margin
    assert p.returncode == 0
    assert "[queuewrap] acquired turn" in out


def test_interrupt_while_waiting_removes_entry(server):
    user = getpass.getuser()
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()

    # Insert an anchor so the client cannot become head
    q.put("ANCHOR")

    env = os.environ.copy()
    env.update(
        {
            "SCHED_HOST": server["host"],
            "SCHED_PORT": str(server["port"]),
            "SCHED_AUTHKEY": server["authkey"].decode(),
        }
    )

    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "2",
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.1)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Wait for explicit state instead of reading output line by line
    wait_for_output(p, f"queued id={user}:B", timeout=2.0)  # Reduced from 3s

    # Send Ctrl-C (SIGINT)
    p.send_signal(signal.SIGINT)
    ret = p.wait(timeout=1)  # Reduced from 4s
    assert ret in (124, 130, 143, 1, -2)  # 124 timeout; 130/143 mapped; -2 raw SIGINT

    # Verify removal with explicit check
    wait_until_removed(q, f"{user}:B", timeout=0.5)
    q.remove("ANCHOR")
    q.clear()


def test_multi_client_exclusive_run(tmp_path, server):
    user = getpass.getuser()
    env = os.environ.copy()
    env.update(
        {
            "SCHED_HOST": server["host"],
            "SCHED_PORT": str(server["port"]),
            "SCHED_AUTHKEY": server["authkey"].decode(),
        }
    )

    # Start A with shorter sleep
    pA = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "A",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "3",  # Reduced from 10s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.3)",  # Reduced from 0.6s
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Verify A is running before starting B
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    wait_until_position(q, f"{user}:A", 0, timeout=1.0)

    pB = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "3",  # Reduced from 10s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "print('B')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Wait for B to acquire using state verification
    wait_for_output(pB, "acquired turn", timeout=2.0)  # Reduced from 10s

    # At this exact moment, A must be removed (explicit verification)
    assert q.position(f"{user}:A") == -1

    outA, _ = pA.communicate(timeout=2)  # Reduced from 10s
    outB, _ = pB.communicate(timeout=1)
    assert pA.returncode == 0
    assert pB.returncode == 0


def test_sigkill_head_removal_allows_next(server):
    user = getpass.getuser()
    env = os.environ.copy()
    env.update(
        {
            "SCHED_HOST": server["host"],
            "SCHED_PORT": str(server["port"]),
            "SCHED_AUTHKEY": server["authkey"].decode(),
        }
    )

    # A acquires head and sleeps; B waits behind A
    pA = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "A",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "10",  # Reduced from 30s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",  # Long sleep but will be killed
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Explicit state: wait for A to reach head
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    wait_until_position(q, f"{user}:A", 0, timeout=1.0)  # Reduced from 5s

    # Start B now
    pB = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "5",  # Reduced from 30s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "print('B-run')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Explicit state: wait for B to queue
    wait_until_position(q, f"{user}:B", 1, timeout=1.0)  # Reduced from 5s

    # Kill A hard (simulate crash)
    pA.kill()
    pA.wait(timeout=1)  # Reduced from 3s

    # B should proceed after heartbeat timeout (~1s with our settings)
    outB, _ = pB.communicate(timeout=2.5)  # Reduced from 12s
    assert pB.returncode == 0
    assert "acquired turn" in outB


def test_killed_waiting_client_removed(server):
    user = getpass.getuser()
    env = os.environ.copy()
    env.update(
        {
            "SCHED_HOST": server["host"],
            "SCHED_PORT": str(server["port"]),
            "SCHED_AUTHKEY": server["authkey"].decode(),
        }
    )

    # A grabs head; B queues then is SIGKILLed while waiting
    pA = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "A",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "10",  # Reduced from 30s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(1)",  # Reduced from 5s
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Wait for A to acquire using explicit state
    wait_for_output(pA, "acquired turn", timeout=3.0)  # Reduced from 5s but with safety margin

    pB = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.01",  # Reduced from 0.05s
            "--timeout",
            "10",  # Reduced from 30s
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "print('B')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Wait for B to queue using explicit state
    wait_for_output(pB, f"queued id={user}:B", timeout=2.0)  # Reduced from 3s

    # Kill B then verify it disappears from queue within ~heartbeat timeout
    pB.kill()
    pB.wait(timeout=0.5)  # Reduced from 2s

    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()

    # Explicit state: wait for removal after heartbeat timeout
    wait_until_removed(q, f"{user}:B", timeout=2.0)  # Reduced from 6s with safety margin
