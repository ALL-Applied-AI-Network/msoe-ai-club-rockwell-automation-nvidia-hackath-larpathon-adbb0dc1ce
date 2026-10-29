"""Instrumented standard-library-first launcher for one or more 4K seeds."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import platform
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def durable_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def marker(progress: Path, event: str, **fields) -> None:
    record = {"event": event, "utc": utc_now(), **fields}
    line = json.dumps(record, sort_keys=True)
    print(f"[phase4c-diagnostic] {line}", flush=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
        return {"returncode": completed.returncode, "output": completed.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def environment_report() -> dict:
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key] = value.strip()
    except OSError as exc:
        memory["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "recorded_utc": utc_now(),
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ACCEPT_EULA",
                "PRIVACY_CONSENT",
                "PYTHONUNBUFFERED",
                "ISAAC",
                "FANUC_ROBOT_USD",
                "PHASE4C_DIAGNOSTIC_PROGRESS",
            )
        },
        "gpu_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "kit_log_search_roots": [
            "/root/.nvidia-omniverse/logs",
            "/root/.cache/ov/Kit",
            "/root/.local/share/ov/logs",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    progress = Path(
        os.environ.get("PHASE4C_DIAGNOSTIC_PROGRESS", run / "diagnostic_progress.jsonl")
    ).resolve()
    os.environ["PHASE4C_DIAGNOSTIC_PROGRESS"] = str(progress)

    marker(progress, "python_process_started", pid=os.getpid())
    durable_json(run / "startup_environment.json", environment_report())

    trace_path = run / "faulthandler.log"
    trace_stream = trace_path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=trace_stream, all_threads=True)
    faulthandler.dump_traceback_later(300, repeat=True, file=trace_stream)

    def terminate(signum, _frame):
        marker(progress, "termination_signal_received", signal=signum)
        faulthandler.dump_traceback(file=trace_stream, all_threads=True)
        trace_stream.flush()
        os.fsync(trace_stream.fileno())
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        marker(progress, "before_project_imports")
        import phase4b_capture
        marker(progress, "after_project_imports")
        seeds = args.seeds if args.seeds is not None else [args.seed]
        sys.argv = [
            "phase4b_capture.py",
            "--out",
            str(run),
            "--seeds",
            *(str(seed) for seed in seeds),
            "--resolution",
            "3840",
            "2160",
            "--maximum-attempts",
            "1",
        ]
        phase4b_capture.main()
        marker(progress, "python_capture_completed")
    except BaseException as exc:
        failure = {
            "utc": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        durable_json(run / "diagnostic_python_failure.json", failure)
        marker(progress, "python_process_failed", error=failure["error"])
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        trace_stream.flush()
        os.fsync(trace_stream.fileno())
        trace_stream.close()


if __name__ == "__main__":
    main()
