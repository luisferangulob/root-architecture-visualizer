#!/usr/bin/env python3
"""Transactional Streamlit viewer for five deterministic root replicates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from root_hpc_manager import (
    ALLOWED_PARTITIONS,
    cancel_run,
    create_run_manifest,
    parse_slurm_time,
    poll_run,
    resume_incomplete_run,
    submit_run,
)
from root_hpc_storage import (
    LOD_AXIS_LIMITS,
    deterministic_lod_axis_ids,
    load_result_bundle,
)


APP_DIR = Path(__file__).resolve().parent
SIM_PATH = Path(
    os.environ.get("SINGLE_ROOT_SIM_PATH", str(APP_DIR / "single_root_sim.py"))
).expanduser().resolve()
if not SIM_PATH.is_file():
    raise FileNotFoundError(
        f"single_root_sim.py was not found at {SIM_PATH}. Set SINGLE_ROOT_SIM_PATH."
    )

# Read and execute the file on every Streamlit script run. A content-addressed
# module name prevents Python's module cache from serving an older simulator
# after the source file changes.
SIM_SOURCE_BYTES = SIM_PATH.read_bytes()
SIM_SOURCE_HASH = hashlib.sha256(SIM_SOURCE_BYTES).hexdigest()[:16]
SIM_MODULE_NAME = f"elastic_root_sim_{SIM_SOURCE_HASH}"
spec = importlib.util.spec_from_file_location(SIM_MODULE_NAME, SIM_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"could not construct an import specification for {SIM_PATH}")
sim = importlib.util.module_from_spec(spec)
sys.modules[SIM_MODULE_NAME] = sim
spec.loader.exec_module(sim)

SIM_SCHEMA_VERSION = getattr(sim, "SCHEMA_VERSION", 4)
CURVE_MODEL_VERSION = getattr(sim, "CURVE_MODEL_VERSION", "unknown")
DIRECTION_MODEL_VERSION = getattr(sim, "DIRECTION_MODEL_VERSION", "unknown")
INITIATION_MODEL_VERSION = getattr(sim, "INITIATION_MODEL_VERSION", "unknown")
RENDERING_MODEL_VERSION = "parent-surface-attached-global-radius-v26"
MAX_SAFE_FULL_RENDER_AXES = 50_000
MAX_SAFE_FULL_RENDER_AXIS_POINTS = 500_000
MAX_REPORTED_STRAHLER_ORDER = getattr(sim, "MAX_REPORTED_STRAHLER_ORDER", 8)
STRAHLER_ORDER_LABELS = [str(i) for i in range(1, MAX_REPORTED_STRAHLER_ORDER + 1)] + [
    f">{MAX_REPORTED_STRAHLER_ORDER}"
]
DEFAULT_STRAHLER_COLORS = {
    "1": "#1F77B4",
    "2": "#E377C2",
    "3": "#2CA02C",
    "4": "#FF7F0E",
    "5": "#9467BD",
    "6": "#8C564B",
    "7": "#17BECF",
    "8": "#BCBD22",
    f">{MAX_REPORTED_STRAHLER_ORDER}": "#D62728",
}
MAX_REPORTED_BRANCH_GENERATION = 8
BRANCH_GENERATION_LABELS = ["Main path"] + [
    f"Generation {i}" for i in range(1, MAX_REPORTED_BRANCH_GENERATION + 1)
] + [f">{MAX_REPORTED_BRANCH_GENERATION}"]
DEFAULT_BRANCH_GENERATION_COLORS = {
    "Main path": "#003366",
    "Generation 1": "#1F77B4",
    "Generation 2": "#E377C2",
    "Generation 3": "#2CA02C",
    "Generation 4": "#FF7F0E",
    "Generation 5": "#9467BD",
    "Generation 6": "#8C564B",
    "Generation 7": "#17BECF",
    "Generation 8": "#BCBD22",
    f">{MAX_REPORTED_BRANCH_GENERATION}": "#D62728",
}


st.set_page_config(
    page_title="Elastic Root Replicate Viewer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .hero-card {
        padding: 22px 28px; border-radius: 22px; color: white;
        background: linear-gradient(135deg, #003366 0%, #0A4C7A 55%, #FFBF00 150%);
        margin-bottom: 18px; box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    }
    .hero-card h1 { color: white; font-size: 2rem; margin-bottom: .25rem; }
    .hero-card p { color: rgba(255,255,255,.92); margin-bottom: 0; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0,51,102,.08), rgba(255,191,0,.08));
        border: 1px solid rgba(0,51,102,.13); padding: 12px;
        border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="hero-card">
      <h1>Elastic Root 5-Replicate Viewer · Schema v{SIM_SCHEMA_VERSION}</h1>
      <p>Five replicates are recomputed and committed together only when Run is pressed.</p>
      <p style="margin-top:6px; opacity:.78;">Curve model: {CURVE_MODEL_VERSION} · Direction version: {DIRECTION_MODEL_VERSION} · Initiation: {INITIATION_MODEL_VERSION} · Rendering: {RENDERING_MODEL_VERSION}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def task_index_from_parameters(
    rain_probability: float,
    branch_probability: float,
    thickness_increment: float,
    replicate: int,
) -> int:
    """Map displayed grid parameters and a replicate to a fixed task index."""

    rain_index = int(round(rain_probability * 100)) - 1
    branch_index = int(round(branch_probability * 100)) - 1
    thickness_index = int(round(thickness_increment * 10)) - 1
    if not 0 <= rain_index < sim.GRID_RAIN_COUNT:
        raise ValueError("rain_probability must be from 0.01 to 0.99")
    if not 0 <= branch_index < sim.GRID_BRANCH_COUNT:
        raise ValueError("branch_probability must be from 0.01 to 0.99")
    if not 0 <= thickness_index < sim.GRID_THICKNESS_COUNT:
        raise ValueError("thickness_increment must be from 0.10 to 7.00")
    if not 0 <= replicate < sim.GRID_REPLICATES:
        raise ValueError("replicate must be 0 to 4")
    return int(
        (((
            thickness_index * sim.GRID_RAIN_COUNT + rain_index
        ) * sim.GRID_BRANCH_COUNT + branch_index) * sim.GRID_REPLICATES)
        + replicate
    )


def geometry_hash(coords: np.ndarray, parent: np.ndarray) -> str:
    """Hash exact topology and IEEE-754 coordinates in a platform-stable layout."""

    digest = hashlib.blake2b(digest_size=12)
    digest.update(np.asarray(coords.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(coords, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(parent, dtype="<i4").tobytes())
    return digest.hexdigest()


def radius_hash(radius: np.ndarray) -> str:
    """Hash an exact IEEE-754 radius array."""

    digest = hashlib.blake2b(digest_size=8)
    digest.update(np.ascontiguousarray(radius, dtype="<f8").tobytes())
    return digest.hexdigest()


def scientific_radius_profile_hash(
    material_arcs: list[np.ndarray],
    radius_profiles: list[np.ndarray],
) -> str:
    """Hash the complete per-axis material/radius profiles."""

    digest = hashlib.blake2b(digest_size=12)
    digest.update(np.asarray([len(material_arcs)], dtype="<i8").tobytes())
    for arcs, radii in zip(material_arcs, radius_profiles):
        arc_array = np.ascontiguousarray(arcs, dtype="<f8")
        radius_array = np.ascontiguousarray(radii, dtype="<f8")
        digest.update(np.asarray(arc_array.shape, dtype="<i8").tobytes())
        digest.update(arc_array.tobytes())
        digest.update(radius_array.tobytes())
    return digest.hexdigest()


def int_array_hash(values: np.ndarray) -> str:
    """Hash an integer array in a platform-stable layout."""

    digest = hashlib.blake2b(digest_size=8)
    digest.update(np.ascontiguousarray(values, dtype="<i4").tobytes())
    return digest.hexdigest()


def compute_branch_generation(
    parent: np.ndarray,
    is_anchor: np.ndarray,
    is_axis_continuation: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-node branching generation from the main downward path.

    Anchor/main-path nodes are generation 0. A lateral growing directly from an
    anchor is generation 1; laterals growing from those laterals increment
    recursively. Smooth continuation nodes keep their parent's generation.
    Edges are colored by the child node's generation.
    """

    if is_axis_continuation is None:
        is_axis_continuation = np.zeros(parent.shape[0], dtype=np.bool_)
    generation = np.zeros(parent.shape[0], dtype=np.int32)
    for node_id in range(1, parent.shape[0]):
        p = int(parent[node_id])
        if bool(is_anchor[node_id]):
            generation[node_id] = 0
        elif bool(is_axis_continuation[node_id]) and p >= 0:
            generation[node_id] = generation[p]
        elif p >= 0 and bool(is_anchor[p]):
            generation[node_id] = 1
        elif p >= 0:
            generation[node_id] = generation[p] + 1
        else:
            generation[node_id] = 0
    return generation


def geometry_comparison(
    previous: dict[str, Any] | None,
    current_coords: np.ndarray,
    current_parent: np.ndarray,
    current_hash: str,
) -> tuple[str, float | None]:
    """Compare current geometry with a previously completed replicate."""

    if previous is None:
        return "first completed run", None
    if previous["geometry_hash"] == current_hash:
        return "identical", 0.0
    old_coords = previous["coords"]
    old_parent = previous["parent"]
    if old_coords.shape != current_coords.shape or old_parent.shape != current_parent.shape:
        return "changed (node count/topology)", None
    parent_equal = np.array_equal(old_parent, current_parent)
    delta = float(np.max(np.abs(old_coords - current_coords)))
    return ("changed (coordinates)" if parent_equal else "changed (topology)"), delta


def run_one_replicate(
    replicate: int,
    grid_values: dict[str, float],
    config: Any,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run one deterministic replicate and prepare its rendering payload."""

    task_index = task_index_from_parameters(
        grid_values["rain_probability"],
        grid_values["branch_probability"],
        grid_values["thickness_increment"],
        replicate,
    )
    params = sim.parameters_for_task(task_index, 20260617)
    started = time.perf_counter()
    result, store = sim.run_simulation(params, config, return_store=True)
    simulation_wall_time_sec = time.perf_counter() - started
    processing_started = time.perf_counter()
    size = int(store.size)
    coords = np.array(store.position[:size], dtype=np.float64, copy=True)
    parent = np.array(store.parent[:size], dtype=np.int32, copy=True)
    radius = np.array(store.radius[:size], dtype=np.float64, copy=True)
    is_anchor = np.array(store.is_anchor[:size], dtype=np.bool_, copy=True)
    is_axis_continuation = np.array(
        getattr(store, "is_axis_continuation", np.zeros(size, dtype=np.bool_))[:size],
        dtype=np.bool_,
        copy=True,
    )
    strahler_orders = np.array(sim.compute_strahler_orders(store), dtype=np.int32, copy=True)
    branch_generation = compute_branch_generation(parent, is_anchor, is_axis_continuation)
    metadata = getattr(store, "axis_metadata", {}) or {}
    material_arcs = [
        np.asarray(values, dtype=np.float64).copy()
        for values in metadata.get("axis_material_arcs", [])
    ]
    radius_profiles = [
        np.asarray(values, dtype=np.float64).copy()
        for values in metadata.get("axis_radii", [])
    ]
    axis_points = [
        np.asarray(values, dtype=np.float64).copy()
        for values in metadata.get("axis_points", [])
    ]
    axis_parent_ids = np.asarray(
        metadata.get("axis_parent_ids", []), dtype=np.int32
    ).copy()
    axis_parent_arc_lengths = np.asarray(
        metadata.get("axis_parent_arc_lengths", []), dtype=np.float64
    ).copy()
    axis_parent_local_azimuths = np.asarray(
        metadata.get("axis_parent_local_azimuths", []), dtype=np.float64
    ).copy()
    axis_node_ids = [
        np.asarray(values, dtype=np.int32).copy()
        for values in metadata.get("axis_node_ids", [])
    ]
    current_hash = geometry_hash(coords, parent)
    change, max_delta = geometry_comparison(previous, coords, parent, current_hash)
    result_processing_time_sec = time.perf_counter() - processing_started
    return {
        "replicate": replicate,
        "task_index": task_index,
        "seed": int(params.seed),
        "result": result,
        "strahler_rows": sim.strahler_summary_rows(result),
        "coords": coords,
        "parent": parent,
        "radius": radius,
        "is_anchor": is_anchor,
        "is_axis_continuation": is_axis_continuation,
        "strahler_orders": strahler_orders,
        "branch_generation": branch_generation,
        "geometry_hash": current_hash,
        "radius_hash": radius_hash(radius),
        "scientific_radius_hash": scientific_radius_profile_hash(
            material_arcs, radius_profiles
        ),
        "axis_material_arcs": material_arcs,
        "axis_radius_profiles": radius_profiles,
        "axis_points": axis_points,
        "axis_parent_ids": axis_parent_ids,
        "axis_parent_arc_lengths": axis_parent_arc_lengths,
        "axis_parent_local_azimuths": axis_parent_local_azimuths,
        "axis_node_ids": axis_node_ids,
        "strahler_hash": int_array_hash(strahler_orders),
        "branch_generation_hash": int_array_hash(branch_generation),
        "geometry_change": change,
        "max_coordinate_delta": max_delta,
        "simulation_wall_time_sec": simulation_wall_time_sec,
        "result_processing_time_sec": result_processing_time_sec,
        "elapsed": simulation_wall_time_sec + result_processing_time_sec,
    }


def load_massive_replicate_output(
    bundle_path: Path,
    replicate: int,
    lod_level: str,
) -> dict[str, Any]:
    """Lazily map one stored replicate and materialize only selected axes."""

    bundle = load_result_bundle(bundle_path, mmap_mode="r")
    manifest = bundle["manifest"]
    result = bundle["result"]
    axis_count = int(manifest["scientific_axis_count"])
    scientific_axis_points = int(bundle["axis_points"].shape[0])
    if lod_level == "Full" and (
        axis_count > MAX_SAFE_FULL_RENDER_AXES
        or scientific_axis_points > MAX_SAFE_FULL_RENDER_AXIS_POINTS
    ):
        raise ValueError(
            "Full rendering is disabled for this architecture because it "
            f"contains {axis_count:,} axes and {scientific_axis_points:,} "
            "axis points. Use High or a lower rendering-only detail level."
        )
    selected = deterministic_lod_axis_ids(axis_count, lod_level)
    displayed_axis_count = int(selected.size)
    point_offsets = bundle["axis_point_offsets"]
    node_offsets = bundle["axis_node_offsets"]
    axis_points: list[np.ndarray] = []
    axis_arcs: list[np.ndarray] = []
    axis_radii: list[np.ndarray] = []
    axis_nodes: list[np.ndarray] = []
    for axis_id in selected:
        point_start = int(point_offsets[axis_id])
        point_stop = int(point_offsets[axis_id + 1])
        node_start = int(node_offsets[axis_id])
        node_stop = int(node_offsets[axis_id + 1])
        axis_points.append(bundle["axis_points"][point_start:point_stop])
        axis_arcs.append(
            bundle["axis_material_arcs"][point_start:point_stop]
        )
        axis_radii.append(bundle["axis_radii"][point_start:point_stop])
        axis_nodes.append(bundle["axis_node_ids"][node_start:node_stop])
    provenance = manifest.get("provenance", {})
    return {
        "replicate": int(replicate),
        "task_index": int(provenance.get("task_index", -1)),
        "seed": int(provenance.get("seed", 0)),
        "result": result,
        "coords": bundle["position"],
        "parent": bundle["parent"],
        "radius": bundle["radius"],
        "is_anchor": bundle["is_anchor"],
        "is_axis_continuation": bundle["is_axis_continuation"],
        "strahler_orders": bundle["node_strahler_orders"],
        "branch_generation": bundle["node_branch_generation"],
        "geometry_hash": str(provenance.get("geometry_hash", "unknown")),
        "radius_hash": str(provenance.get("radius_hash", "unknown")),
        "scientific_radius_hash": str(
            provenance.get("scientific_radius_profile_hash", "unknown")
        ),
        "strahler_hash": str(provenance.get("strahler_hash", "unknown")),
        "branch_generation_hash": str(
            provenance.get("branch_generation_hash", "unknown")
        ),
        "axis_points": axis_points,
        "axis_material_arcs": axis_arcs,
        "axis_radius_profiles": axis_radii,
        "axis_parent_ids": bundle["axis_parent_ids"][:displayed_axis_count],
        "axis_parent_arc_lengths": (
            bundle["axis_parent_arc_lengths"][:displayed_axis_count]
        ),
        "axis_parent_local_azimuths": (
            bundle["axis_parent_local_azimuths"][:displayed_axis_count]
        ),
        "axis_node_ids": axis_nodes,
        "geometry_change": "stored immutable HPC result",
        "max_coordinate_delta": None,
        "simulation_wall_time_sec": float(
            result.get("execution_time_sec", 0.0)
        ),
        "result_processing_time_sec": 0.0,
        "elapsed": float(result.get("execution_time_sec", 0.0)),
        "scientific_axis_count": axis_count,
        "displayed_axis_count": displayed_axis_count,
        "scientific_point_count": int(manifest["scientific_point_count"]),
        "displayed_point_count": int(sum(
            values.shape[0] for values in axis_points
        )),
        "lod_level": lod_level,
        "_bundle": bundle,
    }


def strahler_bucket_label(order: int) -> str:
    """Return the display bucket for a Horton-Strahler order."""

    if int(order) > MAX_REPORTED_STRAHLER_ORDER:
        return f">{MAX_REPORTED_STRAHLER_ORDER}"
    return str(int(order))


def branch_generation_bucket_label(generation: int) -> str:
    """Return the display bucket for a branch generation."""

    value = int(generation)
    if value <= 0:
        return "Main path"
    if value > MAX_REPORTED_BRANCH_GENERATION:
        return f">{MAX_REPORTED_BRANCH_GENERATION}"
    return f"Generation {value}"


def catmull_rom_chain(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Rendering-only smooth interpolation through an ordered root axis."""

    if points.shape[0] <= 2 or samples_per_segment <= 1:
        return points
    resolution = max(2, int(samples_per_segment))
    padded = np.vstack([points[0], points, points[-1]])
    output: list[np.ndarray] = []
    for index in range(1, padded.shape[0] - 2):
        p0, p1, p2, p3 = padded[index - 1], padded[index], padded[index + 1], padded[index + 2]
        t_values = np.linspace(0.0, 1.0, resolution, endpoint=False)
        t = t_values[:, None]
        curve = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
        )
        output.append(curve)
    output.append(points[-1][None, :])
    return np.vstack(output)


def axis_paths(
    parent: np.ndarray,
    is_anchor: np.ndarray,
    is_axis_continuation: np.ndarray,
) -> list[list[int]]:
    """Return continuous parent→child axes without crossing true branch junctions."""

    children_by_parent: list[list[int]] = [[] for _ in range(parent.shape[0])]
    for child in range(1, parent.shape[0]):
        p = int(parent[child])
        if p >= 0:
            children_by_parent[p].append(child)

    paths: list[list[int]] = []
    for child in range(1, parent.shape[0]):
        p = int(parent[child])
        if p < 0:
            continue
        if bool(is_anchor[child]):
            starts_axis = p == 0 or not bool(is_anchor[p])
        else:
            starts_axis = not bool(is_axis_continuation[child])
        if not starts_axis:
            continue
        path = [p, child]
        current = child
        while True:
            continuation_children = [
                c for c in children_by_parent[current]
                if bool(is_anchor[c]) or bool(is_axis_continuation[c])
            ]
            if not continuation_children:
                break
            next_child = min(continuation_children)
            path.append(next_child)
            current = next_child
        paths.append(path)
    return paths


def add_dynamic_thickness_edges(
    fig: go.Figure,
    coords: np.ndarray,
    parent: np.ndarray,
    radius: np.ndarray,
    node_category_labels: np.ndarray,
    root_color: str,
    min_width: int,
    max_width: int,
    bins: int,
    radius_reference: dict[str, float | str],
    *,
    visible_category_labels: set[str],
    category_colors: dict[str, str],
    category_label_order: list[str],
    category_display_names: dict[str, str],
    show_category_legend: bool,
    smooth_curves: bool,
    curve_resolution: int,
    is_anchor: np.ndarray,
    is_axis_continuation: np.ndarray,
) -> None:
    """Add categorized, radius-scaled centerline traces to a Plotly figure."""

    if coords.shape[0] <= 1 or not visible_category_labels:
        return
    children = np.arange(1, coords.shape[0], dtype=np.int32)
    parents = parent[children]
    valid = parents >= 0
    children, parents = children[valid], parents[valid]
    if not children.size:
        return
    edge_labels = node_category_labels[children]
    visible_mask = np.isin(edge_labels, list(visible_category_labels))
    if not np.any(visible_mask):
        return
    average_radius = 0.5 * (radius[children] + radius[parents])
    widths = radius_to_line_width(
        average_radius, min_width, max_width, radius_reference
    )
    # Fixed bins span the user-selected display range for every replicate.
    # Using shared bin midpoints (rather than a replicate's local bin mean)
    # preserves identical rendered widths for identical physical radii.
    bins = max(1, int(bins))
    edges = np.linspace(float(min_width), float(max_width), bins + 1)
    bin_widths = 0.5 * (edges[:-1] + edges[1:])
    child_width = np.zeros(coords.shape[0], dtype=np.float64)
    child_bin = np.full(coords.shape[0], -1, dtype=np.int32)
    for bin_index in range(bins):
        if bin_index == bins - 1:
            mask = (widths >= edges[bin_index]) & (widths <= edges[bin_index + 1])
        else:
            mask = (widths >= edges[bin_index]) & (widths < edges[bin_index + 1])
        child_width[children[mask]] = float(bin_widths[bin_index])
        child_bin[children[mask]] = bin_index

    legend_seen: set[str] = set()
    if smooth_curves:
        path_runs: dict[tuple[str, int], list[np.ndarray]] = {}
        for path in axis_paths(parent, is_anchor, is_axis_continuation):
            if len(path) < 2:
                continue
            start = 1
            while start < len(path):
                child = path[start]
                label = str(node_category_labels[child])
                bin_index = int(child_bin[child])
                if label not in visible_category_labels or bin_index < 0:
                    start += 1
                    continue
                end = start
                while end + 1 < len(path):
                    next_child = path[end + 1]
                    if (
                        str(node_category_labels[next_child]) != label
                        or int(child_bin[next_child]) != bin_index
                        or label not in visible_category_labels
                    ):
                        break
                    end += 1
                points = coords[path[start - 1:end + 1]]
                path_runs.setdefault((label, bin_index), []).append(
                    catmull_rom_chain(points, curve_resolution)
                )
                start = end + 1
        # Thick geometry is committed to Plotly first so fine laterals remain
        # visible on top. Within a width bin, the main path precedes laterals.
        for bin_index in reversed(range(bins)):
            for label in category_label_order:
                if label not in visible_category_labels:
                    continue
                curves = path_runs.get((label, bin_index), [])
                if not curves:
                    continue
                xs: list[float | None] = []
                ys: list[float | None] = []
                zs: list[float | None] = []
                for curve in curves:
                    xs.extend([float(value) for value in curve[:, 0]])
                    ys.extend([float(value) for value in curve[:, 1]])
                    zs.extend([float(value) for value in curve[:, 2]])
                    xs.append(None); ys.append(None); zs.append(None)
                draw_children = children[
                    (edge_labels == label) & (child_bin[children] == bin_index)
                ]
                width = float(np.mean(child_width[draw_children])) if draw_children.size else 1.0
                legend_name = category_display_names.get(label, label)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line={"color": category_colors.get(label, root_color), "width": width},
                    opacity=0.95, showlegend=False, name=legend_name,
                    legendgroup=legend_name, hoverinfo="skip",
                ))
        return

    # Reverse width-bin order: thick primary edges first, fine laterals last.
    for bin_index in reversed(range(bins)):
        if bin_index == bins - 1:
            mask = (widths >= edges[bin_index]) & (widths <= edges[bin_index + 1])
        else:
            mask = (widths >= edges[bin_index]) & (widths < edges[bin_index + 1])
        mask &= visible_mask
        if not np.any(mask):
            continue
        labels_to_draw = [
            label for label in category_label_order
            if label in visible_category_labels and np.any(mask & (edge_labels == label))
        ]
        for label in labels_to_draw:
            draw_mask = mask & (edge_labels == label)
            line_color = category_colors.get(label, root_color)
            legend_name = category_display_names.get(label, label)
            legend_seen.add(label)
            if not np.any(draw_mask):
                continue
            xs: list[float | None] = []
            ys: list[float | None] = []
            zs: list[float | None] = []
            for p, c in zip(parents[draw_mask], children[draw_mask]):
                xs.extend((float(coords[p, 0]), float(coords[c, 0]), None))
                ys.extend((float(coords[p, 1]), float(coords[c, 1]), None))
                zs.extend((float(coords[p, 2]), float(coords[c, 2]), None))
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line={"color": line_color, "width": float(bin_widths[bin_index])},
                opacity=0.95, showlegend=False, name=legend_name,
                legendgroup=legend_name, hoverinfo="skip",
            ))


def shared_radius_reference(outputs: list[dict[str, Any]]) -> dict[str, float | str]:
    """Build one robust physical-radius scale for a completed five-replicate run.

    The zero-anchored clipped square-root mapping avoids independently stretching
    tiny within-replicate ranges. The 99th percentile limits rare swollen nodes,
    while the base-radius floor keeps nearly uniform runs on a stable scale.
    """

    edge_radii: list[np.ndarray] = []
    for output in outputs:
        radius = np.asarray(output["radius"], dtype=np.float64)
        parent = np.asarray(output["parent"], dtype=np.int32)
        children = np.flatnonzero(parent >= 0)
        if children.size:
            edge_radii.append(0.5 * (radius[children] + radius[parent[children]]))
    combined = np.concatenate(edge_radii) if edge_radii else np.asarray([0.0])
    finite = combined[np.isfinite(combined) & (combined >= 0.0)]
    percentile_99 = float(np.percentile(finite, 99.0)) if finite.size else 0.0
    clip_radius = max(percentile_99, float(sim.SimulationConfig().base_radius), 1e-12)
    return {
        "mapping": "zero-anchored clipped square-root",
        "clip_percentile": 99.0,
        "clip_radius": clip_radius,
    }


def radius_to_line_width(
    radii: np.ndarray,
    min_width: int,
    max_width: int,
    radius_reference: dict[str, float | str],
) -> np.ndarray:
    """Map physical radii to widths identically across completed replicates."""

    clip_radius = max(float(radius_reference["clip_radius"]), 1e-12)
    normalized = np.sqrt(np.clip(np.asarray(radii, dtype=np.float64) / clip_radius, 0.0, 1.0))
    return float(min_width) + normalized * float(max_width - min_width)


def parallel_transport_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build stable ring frames along one sampled continuous axis."""

    count = int(points.shape[0])
    tangents = np.empty((count, 3), dtype=np.float64)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if count > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(norms[:, None], 1e-12)
    normals = np.empty_like(tangents)
    binormals = np.empty_like(tangents)
    reference = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(float(tangents[0, 0])) < 0.80
        else np.asarray([0.0, 1.0, 0.0])
    )
    normals[0] = np.cross(tangents[0], reference)
    normals[0] /= max(float(np.linalg.norm(normals[0])), 1e-12)
    binormals[0] = np.cross(tangents[0], normals[0])
    for index in range(1, count):
        transported = normals[index - 1] - (
            float(np.dot(normals[index - 1], tangents[index])) * tangents[index]
        )
        if float(np.linalg.norm(transported)) <= 1e-10:
            transported = np.cross(binormals[index - 1], tangents[index])
        normals[index] = transported / max(float(np.linalg.norm(transported)), 1e-12)
        binormals[index] = np.cross(tangents[index], normals[index])
        binormals[index] /= max(float(np.linalg.norm(binormals[index])), 1e-12)
    return normals, binormals


def interpolate_render_axis(
    points: np.ndarray,
    arcs: np.ndarray,
    values: np.ndarray,
    target_arc: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a rendered centerline, tangent, and scalar at material arc."""

    point_array = np.asarray(points, dtype=np.float64)
    arc_array = np.asarray(arcs, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if point_array.shape[0] == 1:
        return point_array[0].copy(), np.asarray([0.0, 0.0, -1.0]), float(value_array[0])
    clipped = float(np.clip(target_arc, arc_array[0], arc_array[-1]))
    upper = int(np.searchsorted(arc_array, clipped, side="right"))
    upper = min(max(upper, 1), arc_array.size - 1)
    lower = upper - 1
    span = max(float(arc_array[upper] - arc_array[lower]), 1e-12)
    fraction = float(np.clip((clipped - arc_array[lower]) / span, 0.0, 1.0))
    center = (1.0 - fraction) * point_array[lower] + fraction * point_array[upper]
    tangent = point_array[upper] - point_array[lower]
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    value = (1.0 - fraction) * value_array[lower] + fraction * value_array[upper]
    return center, tangent, float(value)


def rendering_local_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the simulator's deterministic parent-relative azimuth frame."""

    direction = np.asarray(tangent, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    reference = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(float(direction[0])) < 0.80
        else np.asarray([0.0, 1.0, 0.0])
    )
    normal = np.cross(direction, reference)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    binormal = np.cross(direction, normal)
    binormal /= max(float(np.linalg.norm(binormal)), 1e-12)
    return normal, binormal


def point_polyline_distance(point: np.ndarray, points: np.ndarray) -> float:
    """Minimum distance from one point to a sampled centerline."""

    target = np.asarray(point, dtype=np.float64)
    line = np.asarray(points, dtype=np.float64)
    if line.shape[0] <= 1:
        return float(np.linalg.norm(target - line[0]))
    starts = line[:-1]
    segments = line[1:] - starts
    squared = np.einsum("ij,ij->i", segments, segments)
    fractions = np.divide(
        np.einsum("ij,ij->i", target[None, :] - starts, segments),
        squared,
        out=np.zeros_like(squared),
        where=squared > 1e-18,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    closest = starts + fractions[:, None] * segments
    return float(np.min(np.linalg.norm(closest - target[None, :], axis=1)))


def surface_attached_render_axes(
    output: dict[str, Any],
    radius_display_multiplier: float,
) -> tuple[list[dict[str, Any]], dict[str, int | float | str]]:
    """Create rendering-only axes whose laterals begin on displayed parent surfaces.

    Every child centerline is translated as a rigid body. Its scientific shape,
    inter-point lengths, topology, radii, and all simulator/hash inputs remain
    unchanged. Descendants attach to their already rendered parent recursively.
    """

    axis_points = output.get("axis_points", [])
    axis_arcs = output.get("axis_material_arcs", [])
    axis_radii = output.get("axis_radius_profiles", [])
    parent_ids = np.asarray(output.get("axis_parent_ids", []), dtype=np.int32)
    parent_arcs = np.asarray(
        output.get("axis_parent_arc_lengths", []), dtype=np.float64
    )
    azimuths = np.asarray(
        output.get("axis_parent_local_azimuths", []), dtype=np.float64
    )
    node_ids = output.get("axis_node_ids", [])
    axis_count = min(
        len(axis_points), len(axis_arcs), len(axis_radii),
        int(parent_ids.size), int(parent_arcs.size), int(azimuths.size),
    )
    multiplier = float(radius_display_multiplier)
    rendered: list[dict[str, Any]] = []
    visible_laterals = 0
    fully_occluded = 0
    tolerance = 1e-10
    for axis_id in range(axis_count):
        original_points = np.asarray(axis_points[axis_id], dtype=np.float64)
        material_arcs = np.asarray(axis_arcs[axis_id], dtype=np.float64)
        displayed_radii = np.maximum(
            np.asarray(axis_radii[axis_id], dtype=np.float64) * multiplier,
            1e-8,
        )
        if original_points.shape[0] != material_arcs.size or material_arcs.size != displayed_radii.size:
            raise ValueError(f"axis {axis_id} rendering arrays have inconsistent lengths")
        points = original_points.copy()
        parent_axis_id = int(parent_ids[axis_id])
        attachment_center = None
        attachment_radius = 0.0
        if parent_axis_id >= 0:
            if parent_axis_id >= len(rendered):
                raise ValueError("rendering axes must be ordered parent before child")
            parent = rendered[parent_axis_id]
            attachment_center, tangent, attachment_radius = interpolate_render_axis(
                parent["points"], parent["material_arcs"], parent["radii"],
                float(parent_arcs[axis_id]),
            )
            normal, binormal = rendering_local_frame(tangent)
            azimuth = float(azimuths[axis_id])
            radial = np.cos(azimuth) * normal + np.sin(azimuth) * binormal
            displayed_attachment = attachment_center + attachment_radius * radial
            points += displayed_attachment - original_points[0]
            protrudes = any(
                point_polyline_distance(point, parent["points"]) + float(child_radius)
                > attachment_radius + tolerance
                for point, child_radius in zip(points, displayed_radii)
            )
            visible_laterals += int(protrudes)
            fully_occluded += int(not protrudes)
        rendered.append({
            "axis_id": axis_id,
            "parent_axis_id": parent_axis_id,
            "points": points,
            "material_arcs": material_arcs.copy(),
            "radii": displayed_radii,
            "node_ids": (
                np.asarray(node_ids[axis_id], dtype=np.int32).copy()
                if axis_id < len(node_ids) else np.asarray([], dtype=np.int32)
            ),
            "attachment_center": attachment_center,
            "attachment_radius": float(attachment_radius),
        })
    diagnostics: dict[str, int | float | str] = {
        "rendering_model_version": RENDERING_MODEL_VERSION,
        "rendered_axis_count": axis_count,
        "rendered_lateral_count": max(axis_count - 1, 0),
        "rendered_visible_lateral_count": visible_laterals,
        "rendered_laterals_fully_occluded_by_parent": fully_occluded,
        "radius_display_multiplier": multiplier,
    }
    return rendered, diagnostics


def add_tapered_tube_meshes(
    fig: go.Figure,
    output: dict[str, Any],
    node_category_labels: np.ndarray,
    *,
    visible_category_labels: set[str],
    category_colors: dict[str, str],
    category_label_order: list[str],
    radial_resolution: int,
    radius_display_multiplier: float,
) -> dict[str, int | float | str]:
    """Render joined physical tubes from scientific per-point radii.

    The multiplier is global and rendering-only. No replicate is normalized.
    Rings share vertices across ordinary continuation segments; parallel-
    transported frames avoid twist discontinuities.
    """

    resolution = max(5, int(radial_resolution))
    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    accumulators: dict[str, dict[str, list[Any]]] = {
        label: {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}
        for label in category_label_order if label in visible_category_labels
    }
    render_axes, diagnostics = surface_attached_render_axes(
        output, radius_display_multiplier
    )
    for axis in render_axes:
        points = np.asarray(axis["points"], dtype=np.float64)
        if points.shape[0] < 2:
            continue
        path_radii = np.asarray(axis["radii"], dtype=np.float64)
        normals, binormals = parallel_transport_frames(points)
        rings = (
            points[:, None, :]
            + path_radii[:, None, None]
            * (
                cosines[None, :, None] * normals[:, None, :]
                + sines[None, :, None] * binormals[:, None, :]
            )
        )
        path_node_ids = np.asarray(axis["node_ids"], dtype=np.int32)
        if int(axis["axis_id"]) == 0:
            segment_node_ids = path_node_ids[1:]
        else:
            segment_node_ids = path_node_ids
        if segment_node_ids.size != points.shape[0] - 1:
            fallback_label = str(node_category_labels[0])
            segment_labels = [fallback_label] * (points.shape[0] - 1)
        else:
            segment_labels = [
                str(node_category_labels[int(node)]) for node in segment_node_ids
            ]
        for label in set(segment_labels) & visible_category_labels:
            bucket = accumulators.get(label)
            if bucket is None:
                continue
            base = len(bucket["x"])
            flat = rings.reshape(-1, 3)
            bucket["x"].extend(flat[:, 0].tolist())
            bucket["y"].extend(flat[:, 1].tolist())
            bucket["z"].extend(flat[:, 2].tolist())
            for segment, segment_label in enumerate(segment_labels):
                if segment_label != label:
                    continue
                low = base + segment * resolution
                high = low + resolution
                for ring_index in range(resolution):
                    next_index = (ring_index + 1) % resolution
                    bucket["i"].extend((low + ring_index, low + ring_index))
                    bucket["j"].extend((high + ring_index, high + next_index))
                    bucket["k"].extend((high + next_index, low + next_index))
            if segment_labels[0] == label:
                center_index = len(bucket["x"])
                bucket["x"].append(float(points[0, 0]))
                bucket["y"].append(float(points[0, 1]))
                bucket["z"].append(float(points[0, 2]))
                for ring_index in range(resolution):
                    bucket["i"].append(center_index)
                    bucket["j"].append(base + (ring_index + 1) % resolution)
                    bucket["k"].append(base + ring_index)
            if segment_labels[-1] == label:
                center_index = len(bucket["x"])
                bucket["x"].append(float(points[-1, 0]))
                bucket["y"].append(float(points[-1, 1]))
                bucket["z"].append(float(points[-1, 2]))
                final_ring = base + (points.shape[0] - 1) * resolution
                for ring_index in range(resolution):
                    bucket["i"].append(center_index)
                    bucket["j"].append(final_ring + ring_index)
                    bucket["k"].append(final_ring + (ring_index + 1) % resolution)
    for label in category_label_order:
        bucket = accumulators.get(label)
        if bucket is None or not bucket["i"]:
            continue
        fig.add_trace(go.Mesh3d(
            x=bucket["x"], y=bucket["y"], z=bucket["z"],
            i=bucket["i"], j=bucket["j"], k=bucket["k"],
            color=category_colors.get(label, "#003366"),
            opacity=1.0, flatshading=False, hoverinfo="skip",
            lighting={
                "ambient": 0.45, "diffuse": 0.75, "specular": 0.18,
                "roughness": 0.72, "fresnel": 0.08,
            },
            lightposition={"x": 100, "y": 150, "z": 80},
            showlegend=False,
            name=label,
        ))
    return diagnostics


def make_figure(
    output: dict[str, Any],
    display: dict[str, Any],
    radius_reference: dict[str, float | str],
) -> go.Figure:
    """Build an interactive 3D figure for one simulation replicate."""

    coords, parent, radius = output["coords"], output["parent"], output["radius"]
    mode = display["visualization_mode"]
    if mode == "Horton-Strahler order":
        node_category_labels = np.array(
            [strahler_bucket_label(order) for order in output["strahler_orders"]],
            dtype=object,
        )
        category_label_order = STRAHLER_ORDER_LABELS
        visible_category_labels = set(display["visible_strahler_labels"])
        category_colors = display["strahler_colors"]
        category_display_names = {
            label: f"Strahler {label}" for label in STRAHLER_ORDER_LABELS
        }
        show_category_legend = True
    elif mode == "Branch generation":
        node_category_labels = np.array(
            [
                branch_generation_bucket_label(generation)
                for generation in output["branch_generation"]
            ],
            dtype=object,
        )
        category_label_order = BRANCH_GENERATION_LABELS
        visible_category_labels = set(display["visible_generation_labels"])
        category_colors = display["branch_generation_colors"]
        category_display_names = {label: label for label in BRANCH_GENERATION_LABELS}
        show_category_legend = True
    else:
        node_category_labels = np.full(coords.shape[0], "Root edges", dtype=object)
        category_label_order = ["Root edges"]
        visible_category_labels = {"Root edges"}
        category_colors = {"Root edges": display["root_color"]}
        category_display_names = {"Root edges": "Root edges"}
        show_category_legend = False
    fig = go.Figure()
    rendering_started = time.perf_counter()
    rendering_diagnostics: dict[str, int | float | str] = {
        "rendering_model_version": RENDERING_MODEL_VERSION,
        "rendered_axis_count": len(output.get("axis_points", [])),
        "rendered_lateral_count": max(len(output.get("axis_points", [])) - 1, 0),
        "rendered_visible_lateral_count": 0,
        "rendered_laterals_fully_occluded_by_parent": 0,
        "radius_display_multiplier": float(display["radius_display_multiplier"]),
    }
    if display["rendering_geometry"] == "Physical tapered tubes":
        rendering_diagnostics = add_tapered_tube_meshes(
            fig, output, node_category_labels,
            visible_category_labels=visible_category_labels,
            category_colors=category_colors,
            category_label_order=category_label_order,
            radial_resolution=display["tube_radial_resolution"],
            radius_display_multiplier=display["radius_display_multiplier"],
        )
    else:
        add_dynamic_thickness_edges(
            fig, coords, parent, radius, node_category_labels, display["root_color"],
            display["min_line_width"], display["max_line_width"], display["thickness_bins"],
            radius_reference,
            visible_category_labels=visible_category_labels,
            category_colors=category_colors,
            category_label_order=category_label_order,
            category_display_names=category_display_names,
            show_category_legend=show_category_legend,
            smooth_curves=display["smooth_curves"],
            curve_resolution=display["curve_resolution"],
            is_anchor=output["is_anchor"],
            is_axis_continuation=output["is_axis_continuation"],
        )
    rendering_time_sec = time.perf_counter() - rendering_started
    if display["show_nodes"]:
        ids = (
            np.unique(np.linspace(0, coords.shape[0] - 1, display["max_points"]).astype(int))
            if coords.shape[0] > display["max_points"] else np.arange(coords.shape[0])
        )
        fig.add_trace(go.Scatter3d(
            x=coords[ids, 0], y=coords[ids, 1], z=coords[ids, 2], mode="markers",
            marker={"size": 2, "color": display["node_color"], "opacity": 0.5},
            showlegend=False,
        ))
    if display["lateral_detail_view"]:
        lateral_ids = np.flatnonzero(~output["is_anchor"])
        if lateral_ids.size > display["max_points"]:
            lateral_ids = lateral_ids[np.unique(np.linspace(
                0, lateral_ids.size - 1, display["max_points"]
            ).astype(int))]
        if lateral_ids.size:
            fig.add_trace(go.Scatter3d(
                x=coords[lateral_ids, 0],
                y=coords[lateral_ids, 1],
                z=coords[lateral_ids, 2],
                mode="markers",
                marker={"size": 2, "color": display["root_color"], "opacity": 0.65},
                showlegend=False,
                hoverinfo="skip",
                name="Exact lateral support points",
            ))
    fig.add_trace(go.Scatter3d(
        x=[coords[0, 0]], y=[coords[0, 1]], z=[coords[0, 2]], mode="markers",
        marker={"size": 8, "color": display["seed_color"], "symbol": "diamond"},
        showlegend=False,
    ))
    axis = {
        "showbackground": True, "backgroundcolor": display["background_color"],
        "gridcolor": "#E2E2E2", "zerolinecolor": "#D0D0D0", "showgrid": True,
    }
    fig.update_layout(
        height=display["plot_height"], margin={"l": 0, "r": 0, "t": 15, "b": 0},
        scene={
            "xaxis": {**axis, "title": "X"}, "yaxis": {**axis, "title": "Y"},
            "zaxis": {**axis, "title": "Z"}, "bgcolor": display["background_color"],
            "aspectmode": "data", "camera": {"eye": {"x": 1.5, "y": 1.6, "z": 1.15}},
            "uirevision": output["geometry_hash"],
        },
        paper_bgcolor=display["background_color"],
        plot_bgcolor=display["background_color"],
        showlegend=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    serialization_started = time.perf_counter()
    serialized_length = len(fig.to_json(validate=False))
    plotly_serialization_time_sec = time.perf_counter() - serialization_started
    rendering_diagnostics.update({
        "tube_mesh_construction_time_sec": rendering_time_sec,
        "plotly_serialization_time_sec": plotly_serialization_time_sec,
        "plotly_serialized_bytes": serialized_length,
    })
    output["last_render_diagnostics"] = rendering_diagnostics
    return fig


st.sidebar.title("🌱 Controls")
st.sidebar.caption("Simulation widgets are transactional: edit freely, then press Run.")
execution_mode = st.sidebar.radio(
    "Execution mode",
    ("Interactive preview", "Massive HPC run"),
    index=0,
    help=(
        "Interactive preview runs five replicates in the foreground. Massive "
        "HPC mode submits one five-task Slurm array and returns immediately."
    ),
)
with st.sidebar.form("simulation_form", clear_on_submit=False):
    st.caption(
        "Root architecture is not selected directly. Taproot dominance, "
        "distributed branching, and local whorls emerge stochastically from "
        "branch probability, growth, thickness, resources, spacing, and "
        "collision constraints. B.P. alone controls the lineage initiation "
        "draw; resources act only after that draw passes."
    )
    st.markdown("### Grid parameters")
    branch_probability = st.slider(
        "B.P. branch probability", 0.01, 0.99, 0.10, 0.01, format="%.2f",
        help=(
            "Exact one-time probability at each primary-axis site in single-trial "
            "mode, or the exact per-step hazard at each open site in retry mode. "
            "Rain, water, P, N, K, demand, focus, starvation, and local stimulus "
            "never alter this probability."
        ),
    )
    rain_probability = st.slider("R.P. rain probability", 0.01, 0.99, 0.50, 0.01, format="%.2f")
    thickness_increment = st.slider("T.I. thickness increment", 0.10, 7.00, 0.10, 0.10, format="%.2f")
    st.markdown("### Simulation settings")
    massive_mode = execution_mode == "Massive HPC run"
    advanced_massive_limits = (
        st.checkbox(
            "Enable advanced massive limits",
            value=False,
            help=(
                "Raises the massive-mode input ceiling from 50,000 to 100,000 "
                "steps and from 10,000,000 to 20,000,000 sampled points."
            ),
        )
        if massive_mode else False
    )
    developmental_steps = st.number_input(
        "Developmental growth steps",
        min_value=1,
        max_value=(
            100_000 if advanced_massive_limits
            else 50_000 if massive_mode
            else 5_000
        ),
        value=5_000 if massive_mode else 500,
        step=100 if massive_mode else 25,
        help=(
            "Primary biological stopping clock. All five replicates are compared "
            "after this same developmental duration."
        ),
    )
    maximum_sampled_points = st.number_input(
        "Maximum sampled points (safety cap)",
        min_value=100,
        max_value=(
            20_000_000 if advanced_massive_limits
            else 10_000_000 if massive_mode
            else 1_000_000
        ),
        value=1_000_000 if massive_mode else 100_000,
        step=100_000 if massive_mode else 5_000,
        help=(
            "Technical memory and geometry limit, not developmental age. A run "
            "that reaches this cap is incomplete and is labeled sample_cap."
        ),
    )
    st.caption(
        "Replicates are compared after the same number of developmental growth "
        "steps. Maximum sampled points and maximum runtime are technical safety "
        "limits. A run that reaches either safety limit is incomplete."
    )
    root_elongation_rate = st.slider(
        "Root elongation rate",
        0.50, 2.00, 1.00, 0.05,
        format="%.2f",
        help="Controls how far active root tips extend during each growth iteration.",
    )
    branch_spacing = st.slider(
        "Mean spacing between new branch sites",
        0.20, 1.20, float(sim.CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS), 0.05,
        format="%.2f",
        help=(
            "Mean exponential gap in the continuous material-arc Poisson site "
            "process. This is not a hard minimum. Schema-v26 canonical value: 0.20."
        ),
    )
    retry_label = st.selectbox(
        "Branch-site retry behavior",
        ("One trial per new mature site", "Retry open mature sites"),
        index=0,
        help=(
            "One trial treats B.P. as a one-time probability at each new site. "
            "Retry mode treats B.P. as a per-step hazard at every open mature site; "
            "fresh azimuths are drawn and thickening can reopen full circumference."
        ),
    )
    if massive_mode:
        wall_time = st.select_slider(
            "Slurm wall time",
            options=(
                "00:10:00", "00:30:00", "01:00:00", "02:00:00",
                "04:00:00", "08:00:00", "12:00:00", "24:00:00",
                "48:00:00", "72:00:00",
            ),
            value="04:00:00",
        )
        max_seconds = parse_slurm_time(wall_time)
        hpc_partition = st.selectbox(
            "Slurm partition", ALLOWED_PARTITIONS, index=0
        )
        hpc_memory_gb = st.number_input(
            "Memory per replicate (GB)", 4, 1024, 64, 4
        )
        hpc_cpus_per_task = st.number_input(
            "CPUs per replicate", 1, 64, 1, 1,
            help=(
                "Ordered events inside a replicate remain serial. Extra CPUs "
                "are reserved for tested numeric libraries and system overhead."
            ),
        )
        checkpoint_interval_steps = st.number_input(
            "Checkpoint interval (completed steps)", 1, 1_000, 5, 1
        )
        requested_lod = st.selectbox(
            "Initial rendering detail",
            tuple(LOD_AXIS_LIMITS),
            index=0,
            help="Rendering-only. Complete scientific output remains on X-disk.",
        )
    else:
        max_seconds = st.slider(
            "Max seconds per replicate", 30, 300, 120, 10
        )
        wall_time = "00:05:00"
        hpc_partition = "standard"
        hpc_memory_gb = 16
        hpc_cpus_per_task = 1
        checkpoint_interval_steps = 5
        requested_lod = "Preview"
    st.markdown("### Resource model")
    soil_water_background = st.slider("Soil water background", 0.0, 1.0, 0.20, 0.05, format="%.2f")
    rain_water_input = st.slider("Rain water input", 0.0, 1.0, 0.80, 0.05, format="%.2f")
    water_infiltration_depth = st.slider("Water infiltration depth", 1.0, 20.0, 6.0, 0.5, format="%.1f")
    phosphorus_concentration = st.slider("Phosphorus concentration", 0.0, 1.0, 0.90, 0.05, format="%.2f")
    nitrogen_concentration = st.slider("Nitrogen concentration", 0.0, 1.0, 0.80, 0.05, format="%.2f")
    potassium_concentration = st.slider("Potassium concentration", 0.0, 1.0, 0.70, 0.05, format="%.2f")
    compute_convex_hull = st.checkbox("Compute convex hull volume", value=False)
    submitted = st.form_submit_button(
        (
            "Submit 5-replicate Slurm array"
            if massive_mode else "Run all 5 replicates"
        ),
        type="primary",
        use_container_width=True,
    )

with st.sidebar.expander("What do these controls mean?", expanded=False):
    st.markdown(
        """
        - **Developmental growth steps:** biological duration shared by all five replicates.
        - **Maximum sampled points:** technical memory/geometry safety cap, not biological age.
        - **Branch probability:** one-time probability per site in single-trial mode, or per-step hazard at each open site in retry mode.
        - **Rain probability:** controls rain/water availability and nitrogen leaching.
        - **Thickness increment:** controls elastic thickening and local branch capacity.
        - **Resource sliders:** shape where and how the root grows.
        - **Root elongation rate:** how far active root tips extend per growth iteration.
        - **Mean site spacing:** expected exponential gap in a continuous material-arc Poisson process, not a hard minimum.

        **Stopping status:** `developmental_steps_complete` is normal completion.
        `sample_cap`, `time_limit`, `collision_limited`, and `no_active_axes`
        mark incomplete development when they occur before the requested duration.
        """
    )

st.sidebar.markdown("### Rendering only")
rendering_geometry = st.sidebar.selectbox(
    "Root geometry rendering",
    ("Physical tapered tubes", "Fast centerlines"),
    index=0,
    help=(
        "Tubes use every scientific per-point radius. Fast centerlines are a "
        "screen-width fallback for very large completed architectures."
    ),
)
tube_radial_resolution = st.sidebar.slider(
    "Tube radial resolution", 5, 16, 8, 1,
    disabled=rendering_geometry != "Physical tapered tubes",
    help="Rendering-only number of vertices around every scientific-radius ring.",
)
radius_display_multiplier = st.sidebar.slider(
    "Global radius display multiplier", 1.0, 12.0, 4.0, 0.5,
    disabled=rendering_geometry != "Physical tapered tubes",
    help=(
        "Rendering-only multiplier applied identically to all five replicates. "
        "It never changes scientific radii, geometry, metrics, or hashes."
    ),
)
visualization_mode = st.sidebar.selectbox(
    "Visualization mode",
    ("Single color", "Horton-Strahler order", "Branch generation"),
    index=0,
    help="Rendering-only choice. Switching modes redraws completed geometry without rerunning simulation.",
)
visible_strahler_labels = STRAHLER_ORDER_LABELS
strahler_colors = DEFAULT_STRAHLER_COLORS.copy()
visible_generation_labels = BRANCH_GENERATION_LABELS
branch_generation_colors = DEFAULT_BRANCH_GENERATION_COLORS.copy()

if visualization_mode == "Horton-Strahler order":
    with st.sidebar.expander("Visualize Horton-Strahler order", expanded=True):
        visible_strahler_labels = st.multiselect(
            "Visible Strahler orders",
            STRAHLER_ORDER_LABELS,
            default=STRAHLER_ORDER_LABELS,
            help="Filtering is applied to edges only; the completed simulation is not rerun.",
        )
        st.caption(
            "Horton-Strahler order groups branches by downstream merging structure. "
            "Order 1 marks terminal/small branches; higher orders represent larger "
            "structural axes. Repeated V-shaped branches can share the same order."
        )
        if st.button("Reset Strahler colors", use_container_width=True):
            for label, color in DEFAULT_STRAHLER_COLORS.items():
                st.session_state[f"strahler_color_{label}"] = color
            st.rerun()
        with st.expander("Strahler order colors", expanded=False):
            strahler_colors = {
                label: st.color_picker(
                    f"Order {label}",
                    DEFAULT_STRAHLER_COLORS.get(label, "#003366"),
                    key=f"strahler_color_{label}",
                )
                for label in STRAHLER_ORDER_LABELS
            }
elif visualization_mode == "Branch generation":
    with st.sidebar.expander("Visualize branch generation", expanded=True):
        visible_generation_labels = st.multiselect(
            "Visible branch generations",
            BRANCH_GENERATION_LABELS,
            default=BRANCH_GENERATION_LABELS,
            help="Filtering is applied to edges only; the completed simulation is not rerun.",
        )
        st.caption(
            "Branch generation follows lineage from the main downward path: main path = 0, "
            "first-order laterals = 1, branches from laterals = 2, and so on. This helps "
            "separate true branch lineage from Horton-Strahler order assignment."
        )
        if st.button("Reset generation colors", use_container_width=True):
            for label, color in DEFAULT_BRANCH_GENERATION_COLORS.items():
                st.session_state[f"branch_generation_color_{label}"] = color
            st.rerun()
        with st.expander("Branch generation colors", expanded=False):
            branch_generation_colors = {
                label: st.color_picker(
                    label,
                    DEFAULT_BRANCH_GENERATION_COLORS.get(label, "#003366"),
                    key=f"branch_generation_color_{label}",
                )
                for label in BRANCH_GENERATION_LABELS
            }
else:
    st.sidebar.caption("Single-color mode draws all visible root edges with the selected root color.")

smooth_curves = st.sidebar.checkbox(
    "Smooth curves",
    value=True,
    help=(
        "Rendering-only interpolation of the simulator's sampled continuous axes. "
        "The simulation itself is already curve-axis based; this only changes display."
    ),
)
curve_resolution = st.sidebar.slider(
    "Curve smoothness", 2, 16, 6, 1,
    help="Number of rendered samples per original segment when Smooth curves is on.",
)

display = {
    "rendering_geometry": rendering_geometry,
    "tube_radial_resolution": int(tube_radial_resolution),
    "radius_display_multiplier": float(radius_display_multiplier),
    "min_line_width": st.sidebar.slider("Minimum edge thickness", 1, 8, 2),
    "max_line_width": st.sidebar.slider("Maximum edge thickness", 2, 30, 14),
    "thickness_bins": st.sidebar.slider("Thickness levels", 2, 14, 8),
    "plot_height": st.sidebar.slider("Plot height", 350, 900, 600, 50),
    "show_nodes": st.sidebar.checkbox("Show sampled support points", value=False),
    "show_radius_profile_plot": st.sidebar.checkbox(
        "Show primary radius profile plot", value=True,
        help="Rendering-only 2D scientific radius versus normalized primary material arc.",
    ),
    "lateral_detail_view": st.sidebar.checkbox(
        "Lateral-detail overlay",
        value=False,
        help=(
            "Rendering-only markers at exact existing lateral support points. "
            "This does not extend, rescale, or manufacture geometry."
        ),
    ),
    "max_points": st.sidebar.slider("Max sampled point markers", 100, 5000, 500, 100),
    "root_color": st.sidebar.color_picker("Root color", "#003366"),
    "node_color": st.sidebar.color_picker("Node color", "#333333"),
    "seed_color": st.sidebar.color_picker("Seed color", "#FFBF00"),
    "background_color": st.sidebar.color_picker("Background color", "#FFFFFF"),
    "smooth_curves": bool(smooth_curves),
    "curve_resolution": int(curve_resolution),
    "visualization_mode": visualization_mode,
    "visible_strahler_labels": list(visible_strahler_labels),
    "strahler_colors": strahler_colors,
    "visible_generation_labels": list(visible_generation_labels),
    "branch_generation_colors": branch_generation_colors,
}
if st.sidebar.button("Clear displayed completed run", use_container_width=True):
    for key in (
        "replicate_outputs", "submitted_config", "grid_values", "run_serial",
        "submitted_sim_hash", "massive_run_dir", "massive_job_id",
        "massive_render_lod",
    ):
        st.session_state.pop(key, None)
    st.rerun()

if submitted:
    sampled_point_cap = int(maximum_sampled_points)
    grid_values = {
        "branch_probability": float(branch_probability),
        "rain_probability": float(rain_probability),
        "thickness_increment": float(thickness_increment),
    }
    config = sim.SimulationConfig(
        steps=int(developmental_steps),
        max_sampled_points=sampled_point_cap,
        target_architecture_size=0,
        max_nodes=sampled_point_cap,
        interactive_safety_cap=sampled_point_cap,
        segment_length=0.50 * float(root_elongation_rate),
        anchor_initial_segment_length=0.30 * float(root_elongation_rate),
        anchor_min_segment_length=0.05 * float(root_elongation_rate),
        branch_min_spacing_along_axis=float(branch_spacing),
        branch_retry_mode=(
            "single_trial"
            if retry_label == "One trial per new mature site"
            else "retry_open_sites"
        ),
        max_seconds_per_simulation=float(max_seconds),
        soil_water_background=float(soil_water_background),
        rain_water_input=float(rain_water_input),
        water_infiltration_depth=float(water_infiltration_depth),
        phosphorus_concentration=float(phosphorus_concentration),
        nitrogen_concentration=float(nitrogen_concentration),
        potassium_concentration=float(potassium_concentration),
        compute_convex_hull=bool(compute_convex_hull),
    )
    if massive_mode:
        try:
            run_dir = create_run_manifest(
                simulator_path=SIM_PATH,
                app_path=Path(__file__).resolve(),
                config=config,
                grid_values=grid_values,
                partition=str(hpc_partition),
                wall_time=str(wall_time),
                memory_gb=int(hpc_memory_gb),
                cpus_per_task=int(hpc_cpus_per_task),
                checkpoint_interval_steps=int(checkpoint_interval_steps),
                rendering_lod=str(requested_lod),
            )
            job_id = submit_run(run_dir)
            st.session_state["massive_run_dir"] = str(run_dir)
            st.session_state["massive_job_id"] = job_id
            st.session_state["massive_render_lod"] = str(requested_lod)
            st.success(
                f"Submitted five-replicate job array {job_id}. "
                "The browser remains responsive while Puma runs the science."
            )
        except Exception as exc:
            st.error(f"Massive-run submission failed: {type(exc).__name__}: {exc}")
    else:
        previous_outputs = st.session_state.get("replicate_outputs", [])
        previous_by_replicate = {x["replicate"]: x for x in previous_outputs}
        # Compute locally first and commit once. Interrupted reruns cannot leave a
        # mixture of old and new replicates in session state.
        completed_outputs: list[dict[str, Any]] = []
        progress = st.progress(0, text="Computing replicate 1 of 5")
        for replicate in range(sim.GRID_REPLICATES):
            completed_outputs.append(run_one_replicate(
                replicate, grid_values, config, previous_by_replicate.get(replicate)
            ))
            progress.progress(
                (replicate + 1) / sim.GRID_REPLICATES,
                text=f"Computed replicate {replicate + 1} of {sim.GRID_REPLICATES}",
            )
        progress.empty()
        st.session_state["replicate_outputs"] = completed_outputs
        st.session_state["submitted_config"] = asdict(config)
        st.session_state["grid_values"] = grid_values
        st.session_state["submitted_sim_hash"] = SIM_SOURCE_HASH
        st.session_state["run_serial"] = int(st.session_state.get("run_serial", 0)) + 1

if execution_mode == "Massive HPC run":
    st.subheader("Massive Schema-v26 HPC dashboard")
    st.caption(
        "Submission is fast; scientific completion remains explicit. Each "
        "replicate runs as one deterministic Slurm array task with atomic "
        "completed-step checkpoints and lossless X-disk output."
    )
    massive_run_value = st.session_state.get("massive_run_dir")
    if not massive_run_value:
        st.info(
            "Configure the massive run in the sidebar and submit the "
            "five-replicate Slurm array."
        )
        st.stop()
    massive_run_dir = Path(str(massive_run_value))

    def render_massive_dashboard() -> None:
        status = poll_run(massive_run_dir)
        submission = status.get("submission") or {}
        job_id = submission.get(
            "job_id", st.session_state.get("massive_job_id", "pending")
        )
        heading_columns = st.columns([2, 2, 1])
        heading_columns[0].metric("Slurm job ID", str(job_id))
        heading_columns[1].metric(
            "Completed replicates",
            f"{sum(bool(row.get('result_available')) for row in status['replicates'])}/5",
        )
        heading_columns[2].metric(
            "Array", "0–4"
        )
        st.code(f"RUN_DIR={massive_run_dir}")
        progress_rows = []
        for row in status["replicates"]:
            requested = max(int(row.get("requested_steps", 0)), 1)
            completed = int(row.get("completed_steps", 0))
            progress_rows.append({
                "replicate": row["replicate"],
                "task_index": row["task_index"],
                "status": row.get("status", "queued"),
                "completed_steps": completed,
                "requested_steps": requested,
                "completion_pct": 100.0 * completed / requested,
                "axes": int(row.get("axes", 0)),
                "branches": int(row.get("branches", 0)),
                "scientific_points": int(row.get("sampled_points", 0)),
                "runtime_sec": float(row.get("runtime_sec", 0.0)),
                "memory_GB": float(
                    row.get("resident_memory_bytes", 0)
                ) / (1024.0 ** 3),
                "checkpoint": bool(row.get("checkpoint_available")),
                "result": bool(row.get("result_available")),
            })
        st.dataframe(
            pd.DataFrame(progress_rows),
            use_container_width=True,
            hide_index=True,
        )
        action_columns = st.columns(4)
        if action_columns[0].button(
            "Refresh progress", use_container_width=True
        ):
            st.rerun()
        if action_columns[1].button(
            "Cancel Slurm array", use_container_width=True
        ):
            if cancel_run(massive_run_dir):
                st.warning(f"Cancellation requested for job {job_id}.")
        if action_columns[2].button(
            "Resume incomplete tasks", use_container_width=True
        ):
            resumed_job_id = resume_incomplete_run(massive_run_dir)
            if resumed_job_id is None:
                st.info("All five replicates already have complete results.")
            else:
                st.session_state["massive_job_id"] = resumed_job_id
                st.success(
                    f"Submitted exact checkpoint resume as job "
                    f"{resumed_job_id}."
                )
        action_columns[3].caption(
            "Progress is read from atomic per-replicate files."
        )

        available = [
            int(row["replicate"])
            for row in status["replicates"]
            if row.get("result_available")
        ]
        if not available:
            st.info(
                "No completed result bundle is available yet. Partial "
                "checkpoints remain on disk and can be resumed exactly."
            )
            return
        st.markdown("### Lazy result viewer")
        viewer_columns = st.columns(2)
        selected_replicate = viewer_columns[0].selectbox(
            "Completed replicate",
            available,
            index=0,
            key="massive_selected_replicate",
        )
        selected_lod = viewer_columns[1].selectbox(
            "Rendering level of detail",
            tuple(LOD_AXIS_LIMITS),
            index=tuple(LOD_AXIS_LIMITS).index(
                st.session_state.get("massive_render_lod", "Preview")
            ),
            key="massive_selected_lod",
        )
        bundle_path = (
            massive_run_dir / "results"
            / f"replicate_{selected_replicate}"
        )
        try:
            output = load_massive_replicate_output(
                bundle_path, selected_replicate, selected_lod
            )
        except ValueError as exc:
            st.warning(str(exc))
            return
        count_columns = st.columns(4)
        count_columns[0].metric(
            "Scientific axes", f"{output['scientific_axis_count']:,}"
        )
        count_columns[1].metric(
            "Displayed axes", f"{output['displayed_axis_count']:,}"
        )
        count_columns[2].metric(
            "Scientific points", f"{output['scientific_point_count']:,}"
        )
        count_columns[3].metric(
            "Displayed axis points", f"{output['displayed_point_count']:,}"
        )
        st.caption(
            "Level-of-detail selection is rendering-only. Stored scientific "
            "coordinates, radii, topology, metrics, and hashes are unchanged."
        )
        st.plotly_chart(
            make_figure(
                output,
                display,
                shared_radius_reference([output]),
            ),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
            key=(
                f"massive-{status['run_id']}-{selected_replicate}-"
                f"{selected_lod}-{output['geometry_hash']}"
            ),
        )

    if hasattr(st, "fragment"):
        st.fragment(run_every=5)(render_massive_dashboard)()
    else:
        render_massive_dashboard()
    st.stop()

outputs = st.session_state.get("replicate_outputs")
if outputs and st.session_state.get("submitted_sim_hash") != SIM_SOURCE_HASH:
    st.warning(
        "single_root_sim.py changed after the displayed run. Stored geometries were "
        "invalidated; press Run all 5 replicates to use the new simulator."
    )
    outputs = None
    st.session_state.pop("replicate_outputs", None)
if not outputs:
    st.info("Choose the simulation values in the sidebar, then press **Run all 5 replicates**.")
    st.caption(f"Simulator: {SIM_PATH} · source SHA-256 {SIM_SOURCE_HASH} · cache_data disabled")
    st.stop()

config_values = st.session_state["submitted_config"]
grid_values = st.session_state["grid_values"]
run_serial = st.session_state["run_serial"]
radius_reference = shared_radius_reference(outputs)

# Present architecture first while retaining detailed scientific and provenance
# output in secondary tabs.
architecture_tab, summary_tab, diagnostics_tab = st.tabs(
    ["Architecture", "Summary", "Diagnostics"]
)

with architecture_tab:
    completed_count = sum(
        int(x["result"].get("normal_developmental_completion", 0))
        for x in outputs
    )
    st.markdown(
        f"**B.P. {grid_values['branch_probability']:.2f} · "
        f"R.P. {grid_values['rain_probability']:.2f} · "
        f"T.I. {grid_values['thickness_increment']:.2f}**"
    )
    if completed_count == sim.GRID_REPLICATES:
        st.caption(
            f"🟢 All 5 replicates completed {config_values['steps']} developmental steps."
        )
    else:
        compact_status = " · ".join(
            f"R{x['replicate']} {x['result'].get('stop_reason', x['result']['status'])} "
            f"{int(x['result'].get('developmental_steps_completed', 0))}/"
            f"{int(x['result'].get('developmental_steps_requested', config_values['steps']))}"
            for x in outputs
        )
        st.caption(f"🟠 Safety-limited run — {compact_status}")

    plot_columns = st.columns(sim.GRID_REPLICATES)
    for output, column in zip(outputs, plot_columns):
        result = output["result"]
        normal = bool(result.get("normal_developmental_completion", 0))
        badge = "🟢 complete" if normal else f"🟠 {result.get('stop_reason', result['status'])}"
        with column:
            st.markdown(f"**Replicate {output['replicate']}**")
            st.caption(badge)
            mode_token = display["visualization_mode"].lower().replace(" ", "-")
            st.plotly_chart(
                make_figure(output, display, radius_reference),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
                key=(
                    f"v26-root-{run_serial}-{output['replicate']}-"
                    f"{output['geometry_hash']}-{mode_token}-"
                    f"{display['rendering_geometry']}-{display['tube_radial_resolution']}-"
                    f"{display['radius_display_multiplier']}-"
                    f"{int(display['smooth_curves'])}-{display['curve_resolution']}"
                ),
            )

with summary_tab:
    resource_columns = st.columns(4)
    resource_specs = (
        ("Mean water captured", "total_water_captured"),
        ("Mean P captured", "total_P_captured"),
        ("Mean N captured", "total_N_captured"),
        ("Mean K captured", "total_K_captured"),
    )
    for column, (label, field) in zip(resource_columns, resource_specs):
        column.metric(
            label,
            f"{np.mean([float(x['result'][field]) for x in outputs]):.4f}",
        )
    summary_fields = (
        "stop_reason", "developmental_steps_completed", "sampled_point_count",
        "axis_count", "first_order_lateral_count", "higher_order_lateral_count",
        "architecture_width", "architecture_depth",
        "architecture_depth_width_ratio", "mean_lateral_emergence_angle",
        "fraction_near_horizontal_lateral_segments",
        "maximum_consecutive_upward_extensions",
    )
    summary_frame = pd.DataFrame([
        {"replicate": x["replicate"], **{
            field: x["result"].get(field) for field in summary_fields
        }} for x in outputs
    ])
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)
    export_frame = pd.DataFrame([
        {field: x["result"].get(field) for field in sim.RESULT_FIELDS}
        for x in outputs
    ])
    st.download_button(
        "Download five-replicate metrics CSV",
        data=export_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"elastic-root-schema-v26-run-{run_serial}.csv",
        mime="text/csv",
    )

with diagnostics_tab:
    advanced_diagnostics = st.toggle("Advanced diagnostics", value=False)
    with st.expander("Submitted inputs and simulator provenance", expanded=False):
        diagnostic_rows = [
            {"input": "branch_probability", "value": grid_values["branch_probability"]},
            {"input": "rain_probability", "value": grid_values["rain_probability"]},
            {"input": "thickness_increment", "value": grid_values["thickness_increment"]},
        ] + [{"input": key, "value": value} for key, value in config_values.items()]
        st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)
        st.code(
            f"SIM_PATH={SIM_PATH}\nSIM_SOURCE_SHA256={SIM_SOURCE_HASH}\n"
            f"RENDERING_MODEL_VERSION={RENDERING_MODEL_VERSION}"
        )

    with st.expander("Geometry and performance", expanded=False):
        st.dataframe(pd.DataFrame([{
            "replicate": x["replicate"],
            "task_index": x["task_index"],
            "seed": x["seed"],
            "execution_time_sec": x["result"].get("execution_time_sec"),
            "simulation_wall_time_sec": x.get("simulation_wall_time_sec"),
            "result_processing_time_sec": x.get("result_processing_time_sec"),
            "tube_mesh_construction_time_sec": x.get(
                "last_render_diagnostics", {}
            ).get("tube_mesh_construction_time_sec"),
            "plotly_serialization_time_sec": x.get(
                "last_render_diagnostics", {}
            ).get("plotly_serialization_time_sec"),
            "fully_occluded_laterals": x.get(
                "last_render_diagnostics", {}
            ).get("rendered_laterals_fully_occluded_by_parent"),
            "status": x["result"].get("stop_reason", x["result"]["status"]),
            "geometry_hash": x["geometry_hash"],
            "scientific_radius_hash": x["scientific_radius_hash"],
            "strahler_hash": x["strahler_hash"],
            "resource_environment_step": x["result"].get("resource_environment_step"),
            "resource_focus_updates": x["result"].get("resource_focus_updates"),
        } for x in outputs]), use_container_width=True, hide_index=True)

    with st.expander("Scientific primary-radius profiles", expanded=False):
        radius_profile_fields = (
            ("basal_radius", "primary_axis_basal_radius"),
            ("radius_10pct", "primary_radius_at_10_percent"),
            ("radius_50pct", "primary_radius_at_50_percent"),
            ("radius_90pct", "primary_radius_at_90_percent"),
            ("tip_radius", "primary_axis_distal_tip_radius"),
            ("basal_tip_ratio", "primary_axis_basal_to_tip_radius_ratio"),
        )
        st.dataframe(pd.DataFrame([{
            "replicate": output["replicate"],
            **{label: float(output["result"][field]) for label, field in radius_profile_fields},
            "scientific_radius_hash": output["scientific_radius_hash"],
        } for output in outputs]), use_container_width=True, hide_index=True)

    if advanced_diagnostics:
        with st.expander("All schema-v26 result fields", expanded=False):
            st.dataframe(export_frame, use_container_width=True, hide_index=True)
        with st.expander("Per-lateral age, growth, and collision audit", expanded=False):
            lateral_frames: list[pd.DataFrame] = []
            for output in outputs:
                payload = output["result"].get("lateral_axis_diagnostics_json", "[]")
                rows = json.loads(payload)
                if rows:
                    frame = pd.DataFrame(rows)
                    frame.insert(0, "replicate", output["replicate"])
                    lateral_frames.append(frame)
            if lateral_frames:
                st.dataframe(
                    pd.concat(lateral_frames, ignore_index=True),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No lateral axes were present in this completed run.")
        with st.expander("Direction-score and resource-time audits", expanded=False):
            st.dataframe(pd.DataFrame([{
                "replicate": x["replicate"],
                "score_component_means": x["result"].get("direction_score_component_means_json"),
                "score_component_maxima": x["result"].get("direction_score_component_maxima_json"),
                "time_series": x["result"].get("resource_time_series_json"),
            } for x in outputs]), use_container_width=True, hide_index=True)

st.stop()

st.markdown(
    f"### Last completed parameter set\n"
    f"**B.P.** `{grid_values['branch_probability']:.2f}` · "
    f"**R.P.** `{grid_values['rain_probability']:.2f}` · "
    f"**T.I.** `{grid_values['thickness_increment']:.2f}` · "
    f"**Developmental steps** `{config_values['steps']}` · "
    f"**Site trials** `{config_values['branch_retry_mode']}` · "
    f"**Sampled-point safety cap** `{config_values['max_sampled_points']}` · "
    f"**Max seconds/replicate** `{config_values['max_seconds_per_simulation']}`"
)
st.caption(
    f"All plots below belong to completed run {run_serial}. Simulator SHA-256: "
    f"{SIM_SOURCE_HASH}. No st.cache_data function is used."
)
st.caption(
    "Visualization-only controls redraw completed geometry without rerunning. "
    "Physical tubes use scientific per-point radii; the global display multiplier "
    "and Fast centerlines fallback leave geometry, topology, radii, and hashes unchanged. "
    "Horton-Strahler order groups branches by hierarchical merging structure, so many "
    "V-shaped branches can share the same order. Branch generation follows lineage from "
    "the main downward path: main path = 0, first laterals = 1, laterals from laterals = 2, "
    f"and >{MAX_REPORTED_BRANCH_GENERATION} groups deeper generations."
)

incomplete_outputs = [
    output for output in outputs
    if (
        not bool(output["result"].get("normal_developmental_completion", 0))
        or str(output["result"].get("stop_reason", output["result"]["status"]))
        in {"sample_cap", "time_limit", "collision_limited", "no_active_axes"}
    )
]
if incomplete_outputs:
    incomplete_labels = ", ".join(
        f"replicate {output['replicate']} "
        f"({output['result'].get('stop_reason', output['result']['status'])})"
        for output in incomplete_outputs
    )
    st.caption(
        "🟠 Safety-limited developmental runs: " + incomplete_labels
        + ". These are not equal-age completed roots."
    )

diagnostic_rows = [
    {"input": "branch_probability", "value": grid_values["branch_probability"]},
    {"input": "rain_probability", "value": grid_values["rain_probability"]},
    {"input": "thickness_increment", "value": grid_values["thickness_increment"]},
] + [
    {"input": key, "value": value}
    for key, value in config_values.items()
    if key not in {"steps", "max_nodes"}
]
with st.expander("Submitted inputs and key SimulationConfig values", expanded=True):
    st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)
    st.code(f"SIM_PATH={SIM_PATH}\nSIM_SOURCE_SHA256={SIM_SOURCE_HASH}")

metric_columns = st.columns(5)
for output, column in zip(outputs, metric_columns):
    result = output["result"]
    with column:
        st.metric(
            f"Rep {output['replicate']}",
            f"{int(result['sampled_node_count']):,} sampled points",
        )
        st.caption(f"Emergent morphology: `{result['emergent_morphology_class']}`")
        primary_lateral_ratio = (
            float(result["primary_axis_length"])
            / max(float(result["total_lateral_length"]), 1e-12)
        )
        st.caption(
            f"stop **{result['stop_reason']}** · steps "
            f"**{int(result['developmental_steps_completed'])}/"
            f"{int(result['developmental_steps_requested'])}** · normal "
            f"**{'yes' if int(result['normal_developmental_completion']) else 'no'}**\n\n"
            f"points **{int(result['sampled_point_count']):,}/"
            f"{int(result['sampled_point_safety_cap']):,}** "
            f"({100.0 * float(result['sampled_point_cap_utilization']):.1f}%)\n\n"
            f"1st/higher axes **{int(result['first_order_lateral_count'])}/"
            f"{int(result['higher_order_lateral_count'])}** · active tips "
            f"**{int(result['final_active_tip_count'])}**\n\n"
            f"opportunities/passes/accepted **{int(result['branch_opportunities'])}/"
            f"{int(result['branch_probability_passes'])}/"
            f"{int(result['accepted_branches'])}**\n\n"
            f"primary length **{float(result['primary_axis_length']):.3f}** · "
            f"1st mean/max **{float(result['mean_first_order_lateral_length']):.3f}/"
            f"{float(result['max_first_order_lateral_length']):.3f}**\n\n"
            f"lateral total **{float(result['total_lateral_length']):.3f}** · "
            f"primary/lateral **{primary_lateral_ratio:.3f}**\n\n"
            f"allocation primary/lateral **"
            f"{float(result['primary_fraction_total_structural_allocation']):.3f}/"
            f"{float(result['lateral_fraction_total_structural_allocation']):.3f}**"
        )
        st.caption(
            f"task {output['task_index']} · {result['status']} · {output['elapsed']:.2f}s\n\n"
            f"geometry `{output['geometry_hash']}`\n\n"
            f"radius `{output['radius_hash']}` · profile `{output['scientific_radius_hash']}` · "
            f"strahler `{output['strahler_hash']}` · "
            f"generation `{output['branch_generation_hash']}`\n\n"
            f"{output['geometry_change']}"
        )
        if output["max_coordinate_delta"] is not None:
            st.caption(f"max |coordinate Δ|: {output['max_coordinate_delta']:.6g}")

resource_columns = st.columns(4)
resource_specs = (
    ("Mean water captured", "total_water_captured"),
    ("Mean P captured", "total_P_captured"),
    ("Mean N captured", "total_N_captured"),
    ("Mean K captured", "total_K_captured"),
)
for column, (label, field) in zip(resource_columns, resource_specs):
    with column:
        st.metric(label, f"{np.mean([float(x['result'][field]) for x in outputs]):.4f}")

plot_columns = st.columns(5)
if display["rendering_geometry"] == "Physical tapered tubes":
    st.caption(
        "Physical tapered tubes use scientific radius at every support point. "
        f"The sole display multiplier ({display['radius_display_multiplier']:.1f}×) "
        "is shared by all five replicates; there is no per-replicate normalization."
    )
else:
    st.caption(
        "Fast centerline widths use one zero-anchored, 99th-percentile-clipped "
        f"mapping shared by all replicates (clip radius "
        f"{float(radius_reference['clip_radius']):.6g})."
    )
for output, column in zip(outputs, plot_columns):
    with column:
        st.markdown(f"**Replicate {output['replicate']}**")
        plot_result = output["result"]
        st.caption(
            f"1st-order **{int(plot_result['first_order_lateral_count'])}** · "
            f"higher-order **{int(plot_result['higher_order_lateral_count'])}**\n\n"
            f"probability passes **{int(plot_result['branch_probability_passes'])}** · "
            f"accepted **{int(plot_result['accepted_branches'])}**\n\n"
            f"max lateral length **{float(plot_result['max_lateral_axis_length']):.4f}** · "
            f"total lateral length **{float(plot_result['total_lateral_length']):.4f}**"
        )
        mode_token = display["visualization_mode"].lower().replace(" ", "-")
        st.plotly_chart(
            make_figure(output, display, radius_reference), use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
            key=(
                f"root-{run_serial}-{output['replicate']}-{output['geometry_hash']}-"
                f"{mode_token}-{display['rendering_geometry']}-"
                f"tube-{display['tube_radial_resolution']}-"
                f"radius-{display['radius_display_multiplier']}-"
                f"smooth-{int(display['smooth_curves'])}-"
                f"{display['curve_resolution']}-detail-"
                f"{int(display['lateral_detail_view'])}"
            ),
        )

radius_profile_fields = (
    ("basal_radius", "primary_axis_basal_radius"),
    ("radius_10pct", "primary_radius_at_10_percent"),
    ("radius_25pct", "primary_radius_at_25_percent"),
    ("radius_50pct", "primary_radius_at_50_percent"),
    ("radius_75pct", "primary_radius_at_75_percent"),
    ("radius_90pct", "primary_radius_at_90_percent"),
    ("tip_radius", "primary_axis_distal_tip_radius"),
    ("basal_tip_ratio", "primary_axis_basal_to_tip_radius_ratio"),
    ("taper_monotonic_fraction", "primary_taper_monotonic_fraction"),
    ("max_off_junction_increase", "primary_max_local_radius_increase_away_from_junction"),
)
with st.expander("Scientific primary-radius profiles", expanded=True):
    st.dataframe(pd.DataFrame([
        {
            "replicate": output["replicate"],
            **{
                label: float(output["result"][field])
                for label, field in radius_profile_fields
            },
            "scientific_radius_hash": output["scientific_radius_hash"],
        }
        for output in outputs
    ]), use_container_width=True, hide_index=True)
    if display["show_radius_profile_plot"]:
        profile_figure = go.Figure()
        for output in outputs:
            if not output["axis_material_arcs"] or not output["axis_radius_profiles"]:
                continue
            arc = np.asarray(output["axis_material_arcs"][0], dtype=np.float64)
            profile = np.asarray(output["axis_radius_profiles"][0], dtype=np.float64)
            normalized_arc = arc / max(float(arc[-1]), 1e-12)
            profile_figure.add_trace(go.Scatter(
                x=normalized_arc,
                y=profile,
                mode="lines",
                name=f"Replicate {output['replicate']}",
            ))
        profile_figure.update_layout(
            xaxis_title="Normalized primary material arc",
            yaxis_title="Scientific radius",
            height=390,
            margin={"l": 25, "r": 15, "t": 25, "b": 35},
            legend={"orientation": "h", "y": 1.05},
        )
        st.plotly_chart(
            profile_figure,
            use_container_width=True,
            config={"displaylogo": False},
            key=f"radius-profile-{run_serial}-{SIM_SOURCE_HASH}",
        )

with st.expander("Geometry change diagnostics", expanded=True):
    st.dataframe(pd.DataFrame([{
        "replicate": x["replicate"], "task_index": x["task_index"], "seed": x["seed"],
        "sampled_points": x["result"]["sampled_node_count"], "status": x["result"]["status"],
        "geometry_hash": x["geometry_hash"], "radius_hash": x["radius_hash"],
        "scientific_radius_hash": x["scientific_radius_hash"],
        "strahler_hash": x["strahler_hash"],
        "branch_generation_hash": x["branch_generation_hash"],
        "versus_previous_completed_run": x["geometry_change"],
        "max_coordinate_delta": x["max_coordinate_delta"],
    } for x in outputs]), use_container_width=True, hide_index=True)

with st.expander("Performance diagnostics", expanded=False):
    st.dataframe(pd.DataFrame([{
        "replicate": x["replicate"],
        "execution_time_sec": x["result"]["execution_time_sec"],
        "growth_iterations": x["result"]["growth_iterations_completed"],
        "attempted_branches": x["result"]["attempted_branches"],
        "probability_failures": x["result"]["probability_failures"],
        "probability_passes": x["result"]["branch_probability_passes"],
        "accepted_branches": x["result"]["accepted_branches"],
        "retry_mode": x["result"]["branch_retry_mode"],
        "active_tip_attempts": x["result"]["tip_extension_attempts"],
        "accepted_tip_extensions": x["result"]["tip_extensions_accepted"],
        "branch_site_trials": x["result"]["branch_site_trials_total"],
        "branch_site_retry_trials": x["result"]["branch_site_retry_trials"],
        "rejected_origin_surface_clearance": x["result"]["rejected_origin_surface_clearance"],
        "rejected_above_soil_surface": x["result"]["rejected_above_soil_surface"],
        "rejected_parent_collision": x["result"]["rejected_parent_collision"],
        "rejected_other_root_collision": x["result"]["rejected_other_root_collision"],
        "rejected_axis_ceiling": x["result"]["rejected_axis_ceiling"],
        "failed_spatial_collision": x["result"]["failed_spatial_collision"],
        "collision_sample_checks": x["result"]["collision_sample_checks"],
        "curve_growth_attempts": x["result"]["curve_growth_attempts"],
        "kd_tree_rebuilds": x["result"]["kd_tree_rebuilds"],
        "branch_origin_candidate_evaluations": x["result"]["branch_origin_candidate_evaluations"],
        "stimulus_evaluated_probability_passes": x["result"]["stimulus_evaluated_probability_passes"],
        "mean_local_primordium_stimulus": x["result"]["mean_local_primordium_stimulus"],
        "initiation_uniform_mean": x["result"]["initiation_uniform_mean"],
    } for x in outputs]), use_container_width=True, hide_index=True)

with st.expander("Compact Strahler summaries", expanded=True):
    tabs = st.tabs([f"Rep {i}" for i in range(sim.GRID_REPLICATES)])
    for output, tab in zip(outputs, tabs):
        with tab:
            frame = pd.DataFrame(output["strahler_rows"])
            frame = frame[(pd.to_numeric(frame["nodes"], errors="coerce") > 0) |
                          (pd.to_numeric(frame["segments"], errors="coerce") > 0)]
            if display["visualization_mode"] == "Horton-Strahler order":
                frame["visible_in_plot"] = frame["order"].astype(str).isin(
                    set(display["visible_strahler_labels"])
                )
            else:
                frame["visible_in_plot"] = False
            st.dataframe(frame, use_container_width=True, hide_index=True)

with st.expander("Selected global metrics", expanded=False):
    fields = [
        "status", "emergent_morphology_class", "curve_model_version",
        "direction_model_version", "resource_model_version",
        "developmental_steps_requested", "developmental_steps_completed",
        "developmental_fraction_completed", "normal_developmental_completion",
        "stop_reason", "sampled_point_safety_cap", "sampled_point_count",
        "sampled_point_cap_utilization", "sample_cap_reached",
        "remaining_sample_capacity", "sample_points_per_developmental_step",
        "maximum_sample_points_in_any_step",
        "target_architecture_size", "target_axis_count",
        "max_growth_iterations", "growth_target_reached",
        "resource_demand_feedback_enabled",
        "global_starvation_signal", "starvation_signal_mean", "starvation_signal_at_branch_origins_mean",
        "resource_support_gate_mean", "width_depth_ratio",
        "low_resource_downward_response_score",
        "primary_axis_length", "total_lateral_length",
        "mean_lateral_axis_length", "max_lateral_axis_length",
        "mean_first_order_lateral_length", "median_first_order_lateral_length",
        "max_first_order_lateral_length",
        "lateral_to_primary_length_ratio", "primary_axis_basal_radius",
        "primary_axis_max_radius", "primary_axis_distal_tip_radius",
        "primary_axis_basal_to_tip_radius_ratio", "primary_axis_mean_radius",
        "primary_axis_radius_integral", "primary_structural_allocation",
        "lateral_structural_allocation",
        "primary_fraction_total_structural_allocation",
        "lateral_fraction_total_structural_allocation",
        "mean_lateral_radius", "lateral_to_primary_radius_ratio",
        "low_bp_taproot_score", "branch_count_at_bp_001",
        "first_order_lateral_count", "higher_order_lateral_count",
        "whorl_event_count", "mean_branches_per_whorl",
        "max_branches_per_whorl", "whorl_depth_spacing_mean",
        "whorl_azimuth_entropy", "fraction_laterals_in_whorls",
        "whorl_score", "branch_origin_candidate_evaluations",
        "stimulus_evaluated_probability_passes",
        "mean_local_primordium_stimulus", "initiation_uniform_mean",
        "probability_failures", "probability_pass_rate",
        "probability_pass_acceptance_rate", "physical_rejection_count",
        "physical_rejection_rate",
        "branch_retry_mode", "rejected_origin_surface_clearance",
        "rejected_above_soil_surface", "rejected_parent_collision",
        "rejected_other_root_collision", "rejected_axis_ceiling",
        "accepted_first_order_laterals", "accepted_higher_order_laterals",
        "opportunity_accounting_error", "probability_pass_accounting_error",
        "rejected_sample_cap", "active_tips_at_step_start_total",
        "active_tip_observations", "tip_extension_attempts",
        "tip_extensions_accepted", "tip_extensions_collision_blocked",
        "tip_extensions_surface_blocked", "tip_extensions_sample_cap_blocked",
        "tip_extensions_other_blocked", "fraction_active_tip_attempts_accepted",
        "primary_tip_extension_attempts", "primary_tip_extensions_accepted",
        "lateral_tip_extension_attempts", "lateral_tip_extensions_accepted",
        "generation_1_extension_attempts", "generation_1_extensions_accepted",
        "generation_2_extension_attempts", "generation_2_extensions_accepted",
        "generation_3plus_extension_attempts", "generation_3plus_extensions_accepted",
        "maximum_active_tip_count", "final_active_tip_count",
        "active_tip_attempt_accounting_error", "branch_sites_created",
        "branch_sites_currently_open", "branch_sites_closed_single_trial",
        "branch_sites_temporarily_surface_full",
        "branch_sites_reopened_after_thickening", "branch_site_trials_total",
        "branch_site_first_trials", "branch_site_retry_trials",
        "branch_site_probability_failures", "branch_site_probability_passes",
        "multi_branch_site_count", "maximum_branches_at_one_site",
        "mean_branches_per_occupied_site",
        "fraction_branches_from_multi_branch_sites",
        "same_site_min_azimuth_separation_deg",
        "same_site_mean_azimuth_separation_deg",
        "minimum_accepted_axial_origin_separation",
        "accepted_origin_surface_clearance_min",
        "accepted_origin_surface_clearance_mean",
        "parent_radius_at_branch_origin_mean",
        "parent_radius_at_multi_branch_site_mean",
        "branch_origin_child_parent_radius_ratio_mean",
        "branch_origin_child_parent_radius_ratio_max",
        "first_order_origin_depth_min", "first_order_origin_depth_p10",
        "first_order_origin_depth_p25", "first_order_origin_depth_median",
        "first_order_origin_depth_p75", "first_order_origin_depth_p90",
        "first_order_origin_depth_max", "first_order_origin_arc_fraction_mean",
        "first_order_origin_arc_fraction_p10",
        "first_order_origin_arc_fraction_p50",
        "first_order_origin_arc_fraction_p90",
        "fraction_first_order_origins_in_proximal_10_percent",
        "fraction_first_order_origins_in_proximal_25_percent",
        "fraction_first_order_origins_in_middle_50_percent",
        "fraction_first_order_origins_in_distal_25_percent",
        "primary_radius_at_10_percent", "primary_radius_at_25_percent",
        "primary_radius_at_50_percent", "primary_radius_at_75_percent",
        "primary_radius_at_90_percent", "primary_taper_monotonic_fraction",
        "primary_max_local_radius_increase_away_from_junction",
        "mean_first_order_basal_tip_radius_ratio",
        "mean_first_order_taper_monotonic_fraction",
        "mean_extension_direction_z", "median_extension_direction_z",
        "fraction_extensions_direction_z_lt_minus_08",
        "final_resource_demand_water", "final_resource_demand_P",
        "final_resource_demand_N", "final_resource_demand_K",
        "final_resource_capture_share_water", "final_resource_capture_share_P",
        "final_resource_capture_share_N", "final_resource_capture_share_K",
        "resource_capture_balance_error",
        "axis_count", "sampled_node_count",
        "mean_axis_arc_length", "max_axis_arc_length",
        "mean_curvature", "p95_curvature", "max_curvature",
        "mean_tip_bend_angle_deg", "p95_tip_bend_angle_deg",
        "anchor_lateral_drift_ratio", "anchor_mean_vertical_component",
        "anchor_tortuosity", "branch_origin_spacing_mean",
        "branch_origin_spacing_min", "branch_emergence_angle_p10_deg",
        "branch_emergence_angle_p50_deg", "branch_emergence_angle_p90_deg",
        "branch_emergence_angle_entropy", "branch_azimuth_entropy",
        "fraction_branches_15_40_deg", "fraction_branches_40_70_deg",
        "fraction_branches_70_95_deg", "fraction_branches_gt_95_deg",
        "hard_v_junction_score", "straight_stick_artifact_score",
        "above_surface_node_count", "fraction_above_surface_nodes",
        "max_above_surface_z", "above_surface_length",
        "fraction_above_surface_length",
        "mean_curve_collision_samples_per_growth",
        "execution_time_sec", "growth_iterations_completed",
        "attempted_branches", "accepted_branches",
        "failed_spatial_collision", "collision_sample_checks",
        "curve_growth_attempts", "kd_tree_rebuilds",
        "total_nodes", "total_root_length", "max_depth", "root_width_x",
        "root_width_y", "bounding_box_volume", "global_strahler_order",
        "number_of_strahler_orders_present", "total_water_captured",
        "total_P_captured", "total_N_captured", "total_K_captured",
        "mean_effective_branch_probability", "acceptance_rate",
        "mean_environmental_resource_signal", "mean_absolute_vertical_component",
        "mean_lateral_component", "fraction_near_vertical_segments",
        "fraction_strongly_lateral_segments", "fraction_upward_segments",
        "fraction_near_horizontal_segments", "mean_branch_vertical_component",
        "mean_continuation_vertical_component", "mean_gravitropism_score",
        "mean_lateral_exploration_score", "mean_lateral_suppression_score",
        "mean_turn_angle_deg", "p95_turn_angle_deg", "fraction_sharp_turns",
        "mean_tip_continuation_angle_deg", "branch_emergence_angle_mean_deg",
        "fraction_hard_forks", "fraction_multi_lateral_branch_nodes",
        "v_shape_score", "mean_branch_outward_alignment",
        "fraction_inward_lateral_branches", "fraction_outward_lateral_branches",
        "same_axis_direction_similarity_mean", "same_axis_direction_similarity_p95",
        "repeated_axis_direction_score",
        "mean_parent_relative_branch_emergence_angle_deg",
        "generation_0_mean_vertical_component",
        "generation_1_mean_vertical_component",
        "generation_2_mean_vertical_component",
        "generation_3_mean_vertical_component",
    ]
    st.dataframe(pd.DataFrame([
        {"replicate": x["replicate"], "geometry_hash": x["geometry_hash"],
         **{field: x["result"].get(field) for field in fields}}
        for x in outputs
    ]), use_container_width=True, hide_index=True)

with st.expander(f"Raw {len(sim.RESULT_FIELDS)}-column metrics for all replicates", expanded=False):
    st.dataframe(pd.DataFrame([
        {"replicate": x["replicate"], "geometry_hash": x["geometry_hash"], **x["result"]}
        for x in outputs
    ]), use_container_width=True, hide_index=True)

st.caption(
    "Simulation outputs are held in session state only after all five replicates finish. "
    "Rendering controls can redraw those completed arrays without rerunning the model."
)
