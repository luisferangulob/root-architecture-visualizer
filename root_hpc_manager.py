#!/usr/bin/env python3
"""Five-replicate Slurm job-array manager for massive Schema-v26 runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from root_hpc_storage import (
    ENGINE_COMPATIBILITY_ID,
    atomic_write_json,
    sha256_file,
)


MANIFEST_VERSION = 1
ALLOWED_PARTITIONS = ("standard", "windfall")
MAX_MASSIVE_STEPS = 100_000
MAX_MASSIVE_POINTS = 20_000_000
MAX_WALL_SECONDS = 72 * 60 * 60
MIN_WALL_SECONDS = 10 * 60


def parse_slurm_time(value: str) -> int:
    """Convert a Slurm wall-time value to seconds."""

    match = re.fullmatch(
        r"(?:(?P<days>\d+)-)?(?P<hours>\d{1,2}):"
        r"(?P<minutes>\d{2}):(?P<seconds>\d{2})",
        value,
    )
    if match is None:
        raise ValueError("wall time must be [days-]HH:MM:SS")
    parts = {key: int(raw or 0) for key, raw in match.groupdict().items()}
    if parts["minutes"] >= 60 or parts["seconds"] >= 60:
        raise ValueError("wall-time minutes and seconds must be below 60")
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def task_index_from_parameters(
    *,
    rain_probability: float,
    branch_probability: float,
    thickness_increment: float,
    replicate: int,
) -> int:
    """Map fixed-grid parameters and a replicate to a task index."""

    rain_index = int(round(rain_probability * 100.0)) - 1
    branch_index = int(round(branch_probability * 100.0)) - 1
    thickness_index = int(round(thickness_increment * 10.0)) - 1
    if not 0 <= rain_index < 99:
        raise ValueError("rain_probability must be 0.01 through 0.99")
    if not 0 <= branch_index < 99:
        raise ValueError("branch_probability must be 0.01 through 0.99")
    if not 0 <= thickness_index < 70:
        raise ValueError("thickness_increment must be 0.10 through 7.00")
    if not 0 <= replicate < 5:
        raise ValueError("replicate must be 0 through 4")
    return int((((thickness_index * 99 + rain_index) * 99 + branch_index) * 5) + replicate)


def default_runs_root() -> Path:
    """Return the configured or platform-appropriate HPC run directory."""

    configured = os.environ.get("ROOT_HPC_RUNS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    user = os.environ.get("USER", "root")
    xdisk = Path("/xdisk") / user / "elastic-root-schema26"
    if xdisk.parent.exists() and os.access(xdisk.parent, os.W_OK):
        return xdisk
    return (Path.cwd() / "hpc_runs").resolve()


def validate_massive_request(
    *,
    steps: int,
    point_cap: int,
    wall_time: str,
    partition: str,
    memory_gb: int,
    cpus_per_task: int,
) -> None:
    """Validate resource limits for a five-replicate HPC request."""

    if not 1 <= int(steps) <= MAX_MASSIVE_STEPS:
        raise ValueError(f"developmental steps must be 1..{MAX_MASSIVE_STEPS}")
    if not 2 <= int(point_cap) <= MAX_MASSIVE_POINTS:
        raise ValueError(f"sampled-point cap must be 2..{MAX_MASSIVE_POINTS}")
    wall_seconds = parse_slurm_time(wall_time)
    if not MIN_WALL_SECONDS <= wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("wall time must be between 10 minutes and 72 hours")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"partition must be one of {ALLOWED_PARTITIONS}")
    if not 1 <= int(memory_gb) <= 1024:
        raise ValueError("memory_gb must be 1..1024")
    if not 1 <= int(cpus_per_task) <= 64:
        raise ValueError("cpus_per_task must be 1..64")


def create_run_manifest(
    *,
    simulator_path: Path,
    app_path: Path,
    config: Any,
    grid_values: Mapping[str, float],
    partition: str,
    wall_time: str,
    memory_gb: int,
    cpus_per_task: int,
    checkpoint_interval_steps: int,
    rendering_lod: str,
    runs_root: Path | None = None,
    master_seed: int = 20260617,
) -> Path:
    """Create an immutable run manifest and its working directories."""

    config_values = (
        dataclasses.asdict(config)
        if dataclasses.is_dataclass(config) else dict(config)
    )
    steps = int(config_values["steps"])
    point_cap = min(
        int(config_values["max_nodes"]),
        int(config_values["max_sampled_points"]),
        int(config_values["interactive_safety_cap"]),
    )
    validate_massive_request(
        steps=steps,
        point_cap=point_cap,
        wall_time=wall_time,
        partition=partition,
        memory_gb=memory_gb,
        cpus_per_task=cpus_per_task,
    )
    if checkpoint_interval_steps < 1:
        raise ValueError("checkpoint interval must be at least one step")
    root = (runs_root or default_runs_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_payload = {
        "config": config_values,
        "grid_values": dict(grid_values),
        "master_seed": int(master_seed),
        "created_ns": time.time_ns(),
        "nonce": uuid.uuid4().hex,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{request_hash}"
    run_dir = root / run_id
    run_dir.mkdir()
    for subdir in ("checkpoints", "progress", "results", "logs"):
        (run_dir / subdir).mkdir()
    replicate_tasks = [
        {
            "replicate": replicate,
            "task_index": task_index_from_parameters(
                rain_probability=float(grid_values["rain_probability"]),
                branch_probability=float(grid_values["branch_probability"]),
                thickness_increment=float(grid_values["thickness_increment"]),
                replicate=replicate,
            ),
        }
        for replicate in range(5)
    ]
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": run_id,
        "immutable": True,
        "schema_version": 26,
        "engine_compatibility_id": ENGINE_COMPATIBILITY_ID,
        "simulator_path": str(simulator_path.resolve()),
        "simulator_sha256": sha256_file(simulator_path.resolve()),
        "app_path": str(app_path.resolve()),
        "app_sha256": sha256_file(app_path.resolve()),
        "python_executable": sys.executable,
        "master_seed": int(master_seed),
        "fixed_grid_total_tasks": 3_430_350,
        "replicate_count": 5,
        "slurm_array": "0-4",
        "replicate_tasks": replicate_tasks,
        "grid_values": dict(grid_values),
        "config": config_values,
        "checkpoint_interval_steps": int(checkpoint_interval_steps),
        "rendering_lod": rendering_lod,
        "slurm": {
            "partition": partition,
            "wall_time": wall_time,
            "wall_seconds": parse_slurm_time(wall_time),
            "memory_gb": int(memory_gb),
            "cpus_per_task": int(cpus_per_task),
        },
        "created_unix_time": time.time(),
        "production_grid_sweep": False,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    write_slurm_script(run_dir)
    (run_dir / "manifest.json").chmod(0o440)
    return run_dir


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Load the JSON manifest for an HPC run directory."""

    return json.loads((run_dir.resolve() / "manifest.json").read_text())


def write_slurm_script(run_dir: Path) -> Path:
    """Write the five-task Slurm array submission script for a run."""

    manifest = load_manifest(run_dir)
    slurm = manifest["slurm"]
    worker = Path(__file__).resolve().with_name("root_hpc_worker.py")
    script = run_dir.resolve() / "submit.slurm"
    body = f"""#!/bin/bash
#SBATCH --job-name=root26-{manifest['run_id'][-12:]}
#SBATCH --array=0-4
#SBATCH --partition={slurm['partition']}
#SBATCH --time={slurm['wall_time']}
#SBATCH --mem={slurm['memory_gb']}G
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --output={run_dir.resolve()}/logs/%A_%a.out
#SBATCH --error={run_dir.resolve()}/logs/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
"{manifest['python_executable']}" "{worker}" --manifest "{run_dir.resolve() / 'manifest.json'}" --resume
"""
    script.write_text(body)
    script.chmod(0o750)
    return script


def submit_run(run_dir: Path, *, dry_run: bool | None = None) -> str:
    """Submit a run through Slurm, or record a dry-run submission."""

    directory = run_dir.resolve()
    if dry_run is None:
        dry_run = shutil.which("sbatch") is None
    if dry_run:
        job_id = f"DRYRUN-{int(time.time())}"
    else:
        completed = subprocess.run(
            ["sbatch", "--parsable", str(directory / "submit.slurm")],
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = completed.stdout.strip().split(";")[0]
    submission = {
        "job_id": job_id,
        "dry_run": bool(dry_run),
        "submitted_unix_time": time.time(),
    }
    atomic_write_json(directory / "submission.json", submission)
    return job_id


def resume_incomplete_run(
    run_dir: Path,
    *,
    dry_run: bool | None = None,
) -> str | None:
    """Submit only unfinished array tasks; workers resume exact checkpoints."""

    directory = run_dir.resolve()
    status = poll_run(directory)
    incomplete = [
        int(row["replicate"])
        for row in status["replicates"]
        if not (
            row.get("status") == "developmental_steps_complete"
            and row.get("result_available")
        )
    ]
    if not incomplete:
        return None
    array_spec = ",".join(str(value) for value in incomplete)
    if dry_run is None:
        dry_run = shutil.which("sbatch") is None
    if dry_run:
        job_id = f"DRYRUN-RESUME-{int(time.time())}"
    else:
        completed = subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--array={array_spec}",
                str(directory / "submit.slurm"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = completed.stdout.strip().split(";")[0]
    atomic_write_json(
        directory / "resume_submission.json",
        {
            "job_id": job_id,
            "dry_run": bool(dry_run),
            "array": array_spec,
            "submitted_unix_time": time.time(),
        },
    )
    return job_id


def cancel_run(run_dir: Path) -> bool:
    """Cancel a submitted run and record the cancellation."""

    submission_path = run_dir.resolve() / "submission.json"
    if not submission_path.exists():
        return False
    submission = json.loads(submission_path.read_text())
    job_id = str(submission["job_id"])
    if not submission.get("dry_run", False):
        subprocess.run(["scancel", job_id], check=True)
    atomic_write_json(
        run_dir.resolve() / "cancellation.json",
        {"job_id": job_id, "cancelled_unix_time": time.time()},
    )
    return True


def poll_run(run_dir: Path) -> dict[str, Any]:
    """Collect current progress and artifact availability for all replicates."""

    directory = run_dir.resolve()
    manifest = load_manifest(directory)
    replicates: list[dict[str, Any]] = []
    for task in manifest["replicate_tasks"]:
        replicate = int(task["replicate"])
        progress_path = directory / "progress" / f"replicate_{replicate}.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
        else:
            progress = {
                "status": "queued",
                "completed_steps": 0,
                "requested_steps": int(manifest["config"]["steps"]),
                "axes": 0,
                "branches": 0,
                "sampled_points": 0,
                "runtime_sec": 0.0,
                "resident_memory_bytes": 0,
            }
        progress["replicate"] = replicate
        progress["task_index"] = int(task["task_index"])
        progress["result_available"] = (
            directory / "results" / f"replicate_{replicate}"
        ).is_dir()
        progress["checkpoint_available"] = (
            directory / "checkpoints" / f"replicate_{replicate}.checkpoint"
        ).is_file()
        replicates.append(progress)
    submission_path = directory / "submission.json"
    submission = (
        json.loads(submission_path.read_text())
        if submission_path.exists() else None
    )
    return {
        "run_id": manifest["run_id"],
        "submission": submission,
        "replicates": replicates,
        "all_complete": all(
            row.get("status") == "developmental_steps_complete"
            and row.get("result_available")
            for row in replicates
        ),
    }
