#!/usr/bin/env python3
"""One deterministic Schema-v26 replicate for a five-task Slurm array."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import resource
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from root_hpc_storage import (
    save_result_bundle,
    sha256_file,
    write_progress_atomic,
)


def resident_memory_bytes() -> int:
    """Return current process memory usage in bytes when available."""

    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return usage if sys.platform == "darwin" else usage * 1024


def load_engine(path: Path) -> ModuleType:
    """Load a simulator module directly from an immutable source path."""

    name = f"root_hpc_engine_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import simulator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry_hash(coords: np.ndarray, parent: np.ndarray) -> str:
    """Hash exact topology and coordinates for provenance."""

    digest = hashlib.blake2b(digest_size=12)
    digest.update(np.asarray(coords.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(coords, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(parent, dtype="<i4").tobytes())
    return digest.hexdigest()


def array_hash(values: np.ndarray, dtype: str, digest_size: int = 8) -> str:
    """Hash an array after normalizing it to a specified dtype."""

    digest = hashlib.blake2b(digest_size=digest_size)
    digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def scientific_radius_profile_hash(
    material_arcs: list[np.ndarray],
    radius_profiles: list[np.ndarray],
) -> str:
    """Hash all scientific material-arc and radius profiles."""

    digest = hashlib.blake2b(digest_size=12)
    digest.update(np.asarray([len(material_arcs)], dtype="<i8").tobytes())
    for arcs, radii in zip(material_arcs, radius_profiles):
        arc_array = np.ascontiguousarray(arcs, dtype="<f8")
        radius_array = np.ascontiguousarray(radii, dtype="<f8")
        digest.update(np.asarray(arc_array.shape, dtype="<i8").tobytes())
        digest.update(arc_array.tobytes())
        digest.update(radius_array.tobytes())
    return digest.hexdigest()


def branch_generation(
    parent: np.ndarray,
    is_anchor: np.ndarray,
    is_axis_continuation: np.ndarray,
) -> np.ndarray:
    """Compute per-node branch generations from stored topology."""

    generation = np.zeros(parent.shape[0], dtype=np.int32)
    for node_id in range(1, parent.shape[0]):
        parent_id = int(parent[node_id])
        if bool(is_anchor[node_id]):
            generation[node_id] = 0
        elif bool(is_axis_continuation[node_id]) and parent_id >= 0:
            generation[node_id] = generation[parent_id]
        elif parent_id >= 0 and bool(is_anchor[parent_id]):
            generation[node_id] = 1
        elif parent_id >= 0:
            generation[node_id] = generation[parent_id] + 1
    return generation


def main() -> None:
    """Run one Slurm-array replicate and persist its result bundle."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    replicate = (
        int(args.replicate)
        if args.replicate is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    if not 0 <= replicate < 5:
        raise ValueError("replicate must be 0 through 4")
    task = manifest["replicate_tasks"][replicate]
    progress_path = run_dir / "progress" / f"replicate_{replicate}.json"
    checkpoint_path = (
        run_dir / "checkpoints" / f"replicate_{replicate}.checkpoint"
    )
    result_path = run_dir / "results" / f"replicate_{replicate}"

    try:
        simulator_path = Path(manifest["simulator_path"])
        if sha256_file(simulator_path) != manifest["simulator_sha256"]:
            raise ValueError(
                "simulator source changed after immutable manifest creation"
            )
        sim = load_engine(simulator_path)
        if int(sim.SCHEMA_VERSION) != 26:
            raise ValueError("massive worker requires Schema v26")
        field_names = {
            field.name for field in dataclasses.fields(sim.SimulationConfig)
        }
        config = sim.SimulationConfig(**{
            key: value
            for key, value in manifest["config"].items()
            if key in field_names
        })
        parameters = sim.parameters_for_task(
            int(task["task_index"]),
            int(manifest["master_seed"]),
        )
        resume_path = (
            checkpoint_path
            if args.resume and checkpoint_path.exists() else None
        )
        write_progress_atomic(progress_path, {
            "status": "starting" if resume_path is None else "resuming",
            "completed_steps": 0,
            "requested_steps": int(config.steps),
            "axes": 0,
            "branches": 0,
            "sampled_points": 0,
            "runtime_sec": 0.0,
            "resident_memory_bytes": 0,
        })
        result, store = sim.run_simulation(
            parameters,
            config,
            return_store=True,
            checkpoint_path=checkpoint_path,
            checkpoint_interval_steps=int(
                manifest["checkpoint_interval_steps"]
            ),
            resume_checkpoint_path=resume_path,
            progress_path=progress_path,
            progress_interval_steps=1,
        )
        size = int(store.size)
        coords = np.asarray(store.position[:size], dtype=np.float64)
        parent = np.asarray(store.parent[:size], dtype=np.int32)
        radius = np.asarray(store.radius[:size], dtype=np.float64)
        anchors = np.asarray(store.is_anchor[:size], dtype=np.bool_)
        continuations = np.asarray(
            store.is_axis_continuation[:size], dtype=np.bool_
        )
        generations = branch_generation(parent, anchors, continuations)
        strahler = np.asarray(
            sim.compute_strahler_orders(store), dtype=np.int32
        )
        store.axis_metadata["node_branch_generation"] = generations
        store.axis_metadata["node_strahler_orders"] = strahler
        provenance: dict[str, Any] = {
            "replicate": replicate,
            "task_index": int(task["task_index"]),
            "seed": int(parameters.seed),
            "simulator_sha256": manifest["simulator_sha256"],
            "geometry_hash": geometry_hash(coords, parent),
            "topology_hash": array_hash(parent, "<i4"),
            "radius_hash": array_hash(radius, "<f8"),
            "scientific_radius_profile_hash": (
                scientific_radius_profile_hash(
                    store.axis_metadata["axis_material_arcs"],
                    store.axis_metadata["axis_radii"],
                )
            ),
            "branch_generation_hash": array_hash(generations, "<i4"),
            "strahler_hash": array_hash(strahler, "<i4"),
        }
        save_result_bundle(
            result_path,
            result=result,
            store=store,
            provenance=provenance,
        )
        write_progress_atomic(progress_path, {
            "status": result["stop_reason"],
            "completed_steps": int(result["developmental_steps_completed"]),
            "requested_steps": int(result["developmental_steps_requested"]),
            "axes": int(result["axis_count"]),
            "branches": int(result["accepted_branches"]),
            "sampled_points": int(result["sampled_point_count"]),
            "runtime_sec": float(result["execution_time_sec"]),
            "resident_memory_bytes": resident_memory_bytes(),
            "result_path": str(result_path),
            **provenance,
        })
    except BaseException as exc:
        write_progress_atomic(progress_path, {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
