import os
import getpass
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from multiprocessing.managers import BaseManager

import pytest


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
        "SCHED_HEARTBEAT_TIMEOUT": "5",
        "SCHED_SWEEP_INTERVAL": "0.2",
    })

    proc = subprocess.Popen(
        [sys.executable, "-m", "fishtools.queuing.server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait until the server is ready to accept connections
    deadline = time.time() + 5.0
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
            time.sleep(0.05)
    else:  # pragma: no cover
        proc.terminate()
        out, err = proc.communicate(timeout=2)
        raise RuntimeError(f"server failed to start: {last_err}\nSTDOUT:{out}\nSTDERR:{err}")

    try:
        yield {"host": host, "port": port, "authkey": authkey}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


@pytest.fixture(autouse=True)
def clear_queue(server):
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    q.clear()
    yield
    q.clear()


def _run_client(server, args, timeout=5):
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
            "5",
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "print('A-start')",
        ],
    )
    assert ret == 0
    assert f"[queuewrap] queued id={user}:A" in out
    assert "[queuewrap] acquired turn" in out

    # Verify no leftover entries
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
            "0.05",
            "--timeout",
            "5",
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

    # Ensure it's queued behind ANCHOR and still waiting
    deadline = time.time() + 3
    while time.time() < deadline:
        pos = q.position(f"{user}:C")
        if pos == 1:
            break
        time.sleep(0.02)
    assert q.position(f"{user}:C") == 1

    # Remove anchor so C can acquire and run
    q.remove("ANCHOR")
    out, _ = p.communicate(timeout=5)
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
            "0.05",
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

    # Wait until it has queued itself
    deadline = time.time() + 3
    queued = False
    while time.time() < deadline:
        line = p.stdout.readline()
        if not line:
            time.sleep(0.01)
            continue
        if f"queued id={user}:B" in line:
            queued = True
            break
    assert queued, "client did not reach queued state"

    # Send Ctrl-C (SIGINT)
    p.send_signal(signal.SIGINT)
    ret = p.wait(timeout=4)
    assert ret in (124, 130, 143, 1, -2)  # 124 timeout; 130/143 mapped; -2 raw SIGINT

    # Ensure B was removed; cleanup anchor
    assert q.position(f"{user}:B") == -1
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

    # Start A which will run for a short time
    pA = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "A",
            "--interval",
            "0.05",
            "--timeout",
            "10",
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(0.6)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Start B shortly after A
    time.sleep(0.05)
    pB = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.05",
            "--timeout",
            "10",
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

    # Wait until B reports acquisition; at that exact time A must have been removed
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    acquired_B = False
    deadline = time.time() + 10
    while time.time() < deadline:
        line = pB.stdout.readline()
        if not line:
            time.sleep(0.01)
            continue
        if "acquired turn" in line:
            assert q.position(f"{user}:A") == -1
            acquired_B = True
            break
    assert acquired_B

    outA, _ = pA.communicate(timeout=10)
    outB, _ = pB.communicate(timeout=10)
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
            "0.05",
            "--timeout",
            "30",
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    # Wait until A is at head (position 0)
    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    deadline = time.time() + 5
    while time.time() < deadline and q.position(f"{user}:A") != 0:
        time.sleep(0.05)
    assert q.position(f"{user}:A") == 0, "A did not reach head in time"

    # Start B now
    pB = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.05",
            "--timeout",
            "30",
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

    # Ensure B is queued (position 1 behind A)
    deadline = time.time() + 5
    while time.time() < deadline and q.position(f"{user}:B") != 1:
        time.sleep(0.05)
    assert q.position(f"{user}:B") == 1, "B did not queue"

    # Kill A hard (simulate crash)
    pA.kill()
    pA.wait(timeout=3)

    # B should proceed after heartbeat timeout (~3s)
    outB, _ = pB.communicate(timeout=12)
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
            "0.05",
            "--timeout",
            "30",
            "--no-shell",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Wait until A acquired
    deadline = time.time() + 5
    while time.time() < deadline:
        line = pA.stdout.readline()
        if "acquired turn" in (line or ""):
            break

    pB = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fishtools.queuing.client",
            "--id",
            "B",
            "--interval",
            "0.05",
            "--timeout",
            "30",
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
    # Wait until B queues
    deadline = time.time() + 3
    while time.time() < deadline:
        line = pB.stdout.readline()
        if f"queued id={user}:B" in (line or ""):
            break
    # Kill B then verify it disappears from queue within ~heartbeat timeout
    pB.kill()
    try:
        pB.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pB.terminate()
        pB.wait(timeout=2)

    m = _connect_mgr(server["host"], server["port"], server["authkey"])
    q = m.get_queue()
    # Poll for removal
    deadline = time.time() + 6
    removed = False
    while time.time() < deadline:
        if q.position(f"{user}:B") == -1:
            removed = True
            break
        time.sleep(0.2)
    assert removed, "B was not removed after heartbeat timeout"
