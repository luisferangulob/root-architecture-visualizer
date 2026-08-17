#!/usr/bin/env python3
"""Atomic checkpoints and lossless lazy storage for Schema-v26 HPC runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CHECKPOINT_FORMAT_VERSION = 1
RESULT_BUNDLE_FORMAT_VERSION = 1
ENGINE_COMPATIBILITY_ID = "schema26-exact-event-sequence-v1"


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot continue the requested exact run."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize a value using stable JSON formatting."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=str,
    )


def configuration_payload(config: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping configuration to a plain dictionary."""

    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("configuration must be a dataclass or mapping")


def configuration_hash(config: Any) -> str:
    """Return a stable SHA-256 digest for a simulation configuration."""

    return hashlib.sha256(
        canonical_json(configuration_payload(config)).encode("utf-8")
    ).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a destination with the supplied bytes."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a human-readable JSON mapping."""

    atomic_write_bytes(
        path,
        (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=True,
                default=str,
            )
            + "\n"
        ).encode("utf-8"),
    )


def checkpoint_header(
    *,
    simulator_path: Path,
    schema_version: int,
    config: Any,
    seed: int,
    task_index: int,
    completed_step: int,
) -> dict[str, Any]:
    """Build provenance metadata for a resumable checkpoint."""

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "engine_compatibility_id": ENGINE_COMPATIBILITY_ID,
        "schema_version": int(schema_version),
        "simulator_sha256": sha256_file(simulator_path),
        "configuration": configuration_payload(config),
        "configuration_sha256": configuration_hash(config),
        "seed": int(seed),
        "task_index": int(task_index),
        "completed_step": int(completed_step),
        "created_unix_time": time.time(),
    }


def save_checkpoint_atomic(
    path: Path,
    *,
    header: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    """Atomically save checkpoint metadata and simulation state."""

    envelope = {
        "header": dict(header),
        "state": dict(state),
    }
    atomic_write_bytes(
        path,
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL),
    )


def load_checkpoint(
    path: Path,
    *,
    simulator_path: Path,
    schema_version: int,
    config: Any,
    seed: int,
    task_index: int,
) -> dict[str, Any]:
    """Load a checkpoint after validating its simulator and run identity."""

    with path.open("rb") as stream:
        envelope = pickle.load(stream)
    if not isinstance(envelope, dict):
        raise CheckpointCompatibilityError("invalid checkpoint envelope")
    header = envelope.get("header")
    state = envelope.get("state")
    if not isinstance(header, dict) or not isinstance(state, dict):
        raise CheckpointCompatibilityError("checkpoint header/state is missing")
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "engine_compatibility_id": ENGINE_COMPATIBILITY_ID,
        "schema_version": int(schema_version),
        "simulator_sha256": sha256_file(simulator_path),
        "configuration_sha256": configuration_hash(config),
        "seed": int(seed),
        "task_index": int(task_index),
    }
    mismatches = {
        key: {"expected": value, "actual": header.get(key)}
        for key, value in expected.items()
        if header.get(key) != value
    }
    if mismatches:
        raise CheckpointCompatibilityError(
            "checkpoint compatibility mismatch: "
            + canonical_json(mismatches)
        )
    return {"header": header, "state": state}


def write_progress_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a progress record."""

    atomic_write_json(path, payload)


def _atomic_replace_directory(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def save_result_bundle(
    destination: Path,
    *,
    result: Mapping[str, Any],
    store: Any,
    provenance: Mapping[str, Any],
) -> Path:
    """Save lossless arrays as .npy files for mmap-backed lazy loading."""

    final_path = destination.resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    ))
    try:
        size = int(store.size)
        metadata = store.axis_metadata
        axis_points = [
            np.asarray(values, dtype=np.float64)
            for values in metadata.get("axis_points", [])
        ]
        axis_arcs = [
            np.asarray(values, dtype=np.float64)
            for values in metadata.get("axis_material_arcs", [])
        ]
        axis_radii = [
            np.asarray(values, dtype=np.float64)
            for values in metadata.get("axis_radii", [])
        ]
        axis_node_ids = [
            np.asarray(values, dtype=np.int32)
            for values in metadata.get("axis_node_ids", [])
        ]

        def offsets(lengths: list[int]) -> np.ndarray:
            output = np.empty(len(lengths) + 1, dtype=np.int64)
            output[0] = 0
            np.cumsum(np.asarray(lengths, dtype=np.int64), out=output[1:])
            return output

        point_offsets = offsets([values.shape[0] for values in axis_points])
        node_offsets = offsets([values.shape[0] for values in axis_node_ids])
        arrays = {
            "position": np.asarray(store.position[:size], dtype=np.float64),
            "parent": np.asarray(store.parent[:size], dtype=np.int32),
            "radius": np.asarray(store.radius[:size], dtype=np.float64),
            "is_anchor": np.asarray(store.is_anchor[:size], dtype=np.bool_),
            "is_axis_continuation": np.asarray(
                store.is_axis_continuation[:size], dtype=np.bool_
            ),
            "axis_point_offsets": point_offsets,
            "axis_points": (
                np.concatenate(axis_points, axis=0)
                if axis_points else np.empty((0, 3), dtype=np.float64)
            ),
            "axis_material_arcs": (
                np.concatenate(axis_arcs)
                if axis_arcs else np.empty(0, dtype=np.float64)
            ),
            "axis_radii": (
                np.concatenate(axis_radii)
                if axis_radii else np.empty(0, dtype=np.float64)
            ),
            "axis_node_offsets": node_offsets,
            "axis_node_ids": (
                np.concatenate(axis_node_ids)
                if axis_node_ids else np.empty(0, dtype=np.int32)
            ),
            "axis_parent_ids": np.asarray(
                metadata.get("axis_parent_ids", []), dtype=np.int32
            ),
            "axis_parent_arc_lengths": np.asarray(
                metadata.get("axis_parent_arc_lengths", []),
                dtype=np.float64,
            ),
            "axis_parent_local_azimuths": np.asarray(
                metadata.get("axis_parent_local_azimuths", []),
                dtype=np.float64,
            ),
            "node_branch_generation": np.asarray(
                metadata.get(
                    "node_branch_generation",
                    np.zeros(size, dtype=np.int32),
                ),
                dtype=np.int32,
            ),
            "node_strahler_orders": np.asarray(
                metadata.get(
                    "node_strahler_orders",
                    np.ones(size, dtype=np.int32),
                ),
                dtype=np.int32,
            ),
        }
        for name, values in arrays.items():
            np.save(temporary / f"{name}.npy", values, allow_pickle=False)
        atomic_write_bytes(
            temporary / "scientific_metadata.pkl",
            pickle.dumps(
                metadata,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )
        atomic_write_json(temporary / "result.json", dict(result))
        manifest = {
            "format_version": RESULT_BUNDLE_FORMAT_VERSION,
            "engine_compatibility_id": ENGINE_COMPATIBILITY_ID,
            "scientific_point_count": size,
            "scientific_axis_count": int(
                metadata.get("axis_count", 0)
            ),
            "arrays": {
                name: {
                    "file": f"{name}.npy",
                    "dtype": str(values.dtype),
                    "shape": list(values.shape),
                }
                for name, values in arrays.items()
            },
            "provenance": dict(provenance),
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        _atomic_replace_directory(temporary, final_path)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final_path


def load_result_bundle(
    source: Path,
    *,
    mmap_mode: str = "r",
    load_scientific_metadata: bool = False,
) -> dict[str, Any]:
    """Load a result bundle, optionally mapping arrays without copying them."""

    bundle = source.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    result = json.loads((bundle / "result.json").read_text())
    arrays = {
        name: np.load(
            bundle / details["file"],
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
        for name, details in manifest["arrays"].items()
    }
    output: dict[str, Any] = {
        "manifest": manifest,
        "result": result,
        **arrays,
    }
    if load_scientific_metadata:
        with (bundle / "scientific_metadata.pkl").open("rb") as stream:
            output["axis_metadata"] = pickle.load(stream)
    return output


LOD_AXIS_LIMITS = {
    "Preview": 2_000,
    "Medium": 10_000,
    "High": 50_000,
    "Full": None,
}


def deterministic_lod_axis_ids(
    axis_count: int,
    level: str,
) -> np.ndarray:
    """Select a deterministic parent-safe prefix of axes for rendering."""

    if level not in LOD_AXIS_LIMITS:
        raise ValueError(f"unknown rendering level: {level}")
    count = max(0, int(axis_count))
    limit = LOD_AXIS_LIMITS[level]
    if limit is None or count <= limit:
        return np.arange(count, dtype=np.int32)
    # Axes are appended parent-before-child. A prefix therefore preserves every
    # attachment ancestor without scanning the full lineage graph.
    return np.arange(int(limit), dtype=np.int32)
