"""Local WebUI process discovery, including servers started before these commands existed."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


class LifecycleError(RuntimeError):
    pass


def _command(args: list[str], *, allow_empty: bool = False) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LifecycleError(f"Cannot inspect the WebUI process: {exc}") from exc
    if result.returncode and not (allow_empty and result.returncode == 1 and not result.stdout):
        raise LifecycleError(f"Cannot inspect the WebUI process: {result.stderr.strip()}")
    return result.stdout.strip()


@dataclass(frozen=True)
class Instance:
    pid: int
    identity: str


def _identity(pid: int) -> str:
    return _command(["ps", "-ww", "-p", str(pid), "-o", "uid=,lstart=,args="], allow_empty=True)


def find_instance(host: str, port: int, script: Path) -> Instance | None:
    """Only recognize this user's Python script in this checkout, never just a port."""
    output = _command([
        "lsof", "-nP", "-a", f"-iTCP@{host}:{port}", "-sTCP:LISTEN", "-Fp",
    ], allow_empty=True)
    pids = {int(line[1:]) for line in output.splitlines() if line.startswith("p")}
    if not pids:
        return None
    if len(pids) != 1:
        raise LifecycleError(f"Port {port} has multiple listeners. No process was stopped.")
    pid = pids.pop()
    identity = _identity(pid)
    fields = identity.split(None, 6)  # uid, five start-time fields, command
    if len(fields) != 7 or fields[0] != str(os.getuid()):
        raise LifecycleError(f"Port {port} belongs to an unrecognized process. No process was stopped.")
    try:
        argv = shlex.split(fields[6])
    except ValueError:
        argv = []
    # Restrict recognition to Python's script form. In particular, a python -c
    # argument mentioning webui.py is not a WebUI server.
    candidate = None
    if argv and Path(argv[0]).name.startswith("python"):
        remaining = argv[1:]
        while remaining and remaining[0] in {"-u", "-B", "-E", "-s"}:
            remaining = remaining[1:]
        if remaining and not remaining[0].startswith("-"):
            candidate = Path(remaining[0])
    if candidate is not None and not candidate.is_absolute():
        cwd_output = _command(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"])
        cwd = next((line[1:] for line in cwd_output.splitlines() if line.startswith("n")), None)
        candidate = Path(cwd) / candidate if cwd else None
    if candidate is None or candidate.resolve() != script.resolve():
        raise LifecycleError(
            f"Port {port} is used by another application or checkout (PID {pid}). "
            "No process was stopped. Choose another --port."
        )
    return Instance(pid, identity)


def stop_instance(instance: Instance, host: str, port: int, script: Path, timeout: float = 10) -> None:
    # Repeat discovery just before signalling to avoid trusting stale PID data.
    if find_instance(host, port, script) != instance:
        raise LifecycleError("The listener changed. No process was stopped; run the command again.")
    try:
        os.kill(instance.pid, signal.SIGTERM)  # only the UI, never its pipeline process group
    except ProcessLookupError:
        return
    except OSError as exc:
        raise LifecycleError(f"Cannot stop WebUI PID {instance.pid}: {exc}") from exc
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _identity(instance.pid) != instance.identity:
            return
        time.sleep(0.1)
    raise LifecycleError(
        f"WebUI PID {instance.pid} has not exited yet. Wait and retry --status. "
        "It was not force-killed."
    )
