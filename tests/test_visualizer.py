"""Application, rendering, and simulator-interface regression tests."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app_elastic_geometry.py"
APP_SOURCE = APP_PATH.read_text()


def app_render_namespace() -> dict[str, object]:
    """Load pure rendering helpers without executing the Streamlit interface."""

    import plotly.graph_objects as go

    wanted = {
        "axis_paths",
        "parallel_transport_frames",
        "add_tapered_tube_meshes",
        "geometry_hash",
        "radius_hash",
        "scientific_radius_profile_hash",
        "interpolate_render_axis",
        "rendering_local_frame",
        "point_polyline_distance",
        "surface_attached_render_axes",
    }
    tree = ast.parse(APP_SOURCE)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            )
        ]
        + definitions,
        type_ignores=[],
    )
    namespace: dict[str, object] = {
        "np": np,
        "go": go,
        "Any": Any,
        "hashlib": __import__("hashlib"),
        "RENDERING_MODEL_VERSION": "parent-surface-attached-global-radius-v26",
    }
    exec(
        compile(ast.fix_missing_locations(module), APP_PATH.as_posix(), "exec"),
        namespace,
    )
    return namespace


def deterministic_renderer_fixture(
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Create a small primary axis with three lateral axes."""

    primary = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]
    )
    lateral_points = [
        np.asarray(
            [[0.0, 0.0, -0.5], [0.0, -0.08, -0.56], [0.0, -0.15, -0.64]]
        ),
        np.asarray(
            [[0.0, 0.0, -1.0], [0.07, 0.04, -1.06], [0.13, 0.08, -1.14]]
        ),
        np.asarray(
            [[0.0, 0.0, -1.5], [-0.07, 0.04, -1.56], [-0.13, 0.08, -1.64]]
        ),
    ]
    axis_points = [primary, *lateral_points]
    axis_arcs = [
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 0.10, 0.21]),
        np.asarray([0.0, 0.10, 0.21]),
        np.asarray([0.0, 0.10, 0.21]),
    ]
    axis_radii = [
        np.asarray([0.20, 0.15, 0.10]),
        np.asarray([0.030, 0.025, 0.020]),
        np.asarray([0.028, 0.023, 0.018]),
        np.asarray([0.026, 0.021, 0.016]),
    ]
    coords = np.vstack([primary, *[points[1:] for points in lateral_points]])
    parent = np.asarray([-1, 0, 1, 0, 3, 1, 5, 1, 7], dtype=np.int32)
    radius = np.asarray(
        [0.20, 0.15, 0.10, 0.025, 0.020, 0.023, 0.018, 0.021, 0.016]
    )
    output: dict[str, object] = {
        "axis_points": axis_points,
        "axis_material_arcs": axis_arcs,
        "axis_radius_profiles": axis_radii,
        "axis_parent_ids": np.asarray([-1, 0, 0, 0], dtype=np.int32),
        "axis_parent_arc_lengths": np.asarray([0.0, 0.5, 1.0, 1.5]),
        "axis_parent_local_azimuths": np.asarray(
            [math.nan, 0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        ),
        "axis_node_ids": [
            np.asarray([0, 1, 2], dtype=np.int32),
            np.asarray([3, 4], dtype=np.int32),
            np.asarray([5, 6], dtype=np.int32),
            np.asarray([7, 8], dtype=np.int32),
        ],
    }
    return output, coords, parent, radius


def test_application_contract_excludes_removed_controls() -> None:
    assert "Root morphology" not in APP_SOURCE
    assert "morphology_mode=" not in APP_SOURCE
    assert "Rain-branch coupling" not in APP_SOURCE
    assert "rain_branch_coupling" not in APP_SOURCE


def test_application_exposes_retry_and_spacing_controls() -> None:
    assert "Branch-site retry behavior" in APP_SOURCE
    assert "One trial per new mature site" in APP_SOURCE
    assert "Retry open mature sites" in APP_SOURCE
    assert "Mean spacing between new branch sites" in APP_SOURCE
    assert "Schema-v26 canonical value: 0.20" in APP_SOURCE


def test_app_defaults_to_scientific_tapered_tubes() -> None:
    assert '("Physical tapered tubes", "Fast centerlines")' in APP_SOURCE
    assert "go.Mesh3d(" in APP_SOURCE
    assert "surface_attached_render_axes" in APP_SOURCE
    assert (
        "displayed_attachment = attachment_center + attachment_radius * radial"
        in APP_SOURCE
    )
    assert "radius_display_multiplier" in APP_SOURCE
    assert "there is no per-replicate normalization" in APP_SOURCE.lower()


def test_tube_rendering_does_not_mutate_geometry_or_hashes() -> None:
    import plotly.graph_objects as go

    namespace = app_render_namespace()
    output, coords, parent, radius = deterministic_renderer_fixture()
    original_points = [values.copy() for values in output["axis_points"]]
    geometry_before = namespace["geometry_hash"](coords, parent)
    radius_before = namespace["radius_hash"](radius)
    figure = go.Figure()
    diagnostics = namespace["add_tapered_tube_meshes"](
        figure,
        output,
        np.asarray(["root"] * coords.shape[0], dtype=object),
        visible_category_labels={"root"},
        category_colors={"root": "#003366"},
        category_label_order=["root"],
        radial_resolution=8,
        radius_display_multiplier=4.0,
    )
    assert len(figure.data) == 1
    assert diagnostics["rendered_axis_count"] == 4
    assert diagnostics["rendered_laterals_fully_occluded_by_parent"] == 0
    assert all(
        np.array_equal(before, after)
        for before, after in zip(original_points, output["axis_points"])
    )
    assert namespace["geometry_hash"](coords, parent) == geometry_before
    assert namespace["radius_hash"](radius) == radius_before


def test_surface_attached_laterals_remain_visible_at_all_multipliers() -> None:
    namespace = app_render_namespace()
    output, coords, parent, radius = deterministic_renderer_fixture()
    geometry_before = namespace["geometry_hash"](coords, parent)
    radius_before = namespace["radius_hash"](radius)
    visible_counts: list[int] = []
    for multiplier in (1.0, 2.0, 4.0, 8.0, 12.0):
        axes, diagnostics = namespace["surface_attached_render_axes"](
            output, multiplier
        )
        assert len(axes) == 4
        assert diagnostics["rendered_laterals_fully_occluded_by_parent"] == 0
        visible_counts.append(int(diagnostics["rendered_visible_lateral_count"]))
        for child in axes[1:]:
            primary = axes[int(child["parent_axis_id"])]
            center, _tangent, displayed_parent_radius = namespace[
                "interpolate_render_axis"
            ](
                primary["points"],
                primary["material_arcs"],
                primary["radii"],
                output["axis_parent_arc_lengths"][int(child["axis_id"])],
            )
            attachment = np.asarray(child["points"])[0]
            assert np.linalg.norm(attachment - center) == pytest.approx(
                displayed_parent_radius
            )
            assert np.linalg.norm(attachment - center) > 0.0
            assert any(
                namespace["point_polyline_distance"](
                    ring_center, primary["points"]
                )
                >= displayed_parent_radius - 1e-10
                for ring_center in child["points"]
            )
            assert child["radii"][0] < displayed_parent_radius
        assert namespace["geometry_hash"](coords, parent) == geometry_before
        assert namespace["radius_hash"](radius) == radius_before
    assert visible_counts == sorted(visible_counts) == [3, 3, 3, 3, 3]


def test_shared_five_replicate_radius_scaling_is_identical() -> None:
    import plotly.graph_objects as go

    namespace = app_render_namespace()

    def rendered_radial_extent(radius_value: float) -> float:
        figure = go.Figure()
        output = {
            "axis_points": [
                np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            ],
            "axis_material_arcs": [np.asarray([0.0, 1.0])],
            "axis_radius_profiles": [
                np.asarray([radius_value, radius_value])
            ],
            "axis_parent_ids": np.asarray([-1], dtype=np.int32),
            "axis_parent_arc_lengths": np.asarray([0.0]),
            "axis_parent_local_azimuths": np.asarray([math.nan]),
            "axis_node_ids": [np.asarray([0, 1], dtype=np.int32)],
        }
        namespace["add_tapered_tube_meshes"](
            figure,
            output,
            np.asarray(["root", "root"], dtype=object),
            visible_category_labels={"root"},
            category_colors={"root": "#003366"},
            category_label_order=["root"],
            radial_resolution=8,
            radius_display_multiplier=4.0,
        )
        mesh = figure.data[0]
        return float(
            np.max(np.hypot(np.asarray(mesh.x)[:8], np.asarray(mesh.y)[:8]))
        )

    assert rendered_radial_extent(0.05) == pytest.approx(0.20)
    assert rendered_radial_extent(0.05) == pytest.approx(
        rendered_radial_extent(0.05)
    )


def test_app_presents_architecture_first_and_hides_diagnostics_by_default() -> None:
    tabs_at = APP_SOURCE.index('["Architecture", "Summary", "Diagnostics"]')
    plots_at = APP_SOURCE.index("v26-root-")
    diagnostics_at = APP_SOURCE.index('st.toggle("Advanced diagnostics"')
    assert tabs_at < plots_at < diagnostics_at
    assert (
        'with st.expander("Scientific primary-radius profiles", expanded=False)'
        in APP_SOURCE
    )
    assert 'APP_DIR / "single_root_sim.py"' in APP_SOURCE


def test_app_preserves_interactive_mode_and_exposes_massive_dashboard() -> None:
    assert '"Interactive preview", "Massive HPC run"' in APP_SOURCE
    assert "max_value=(\n            100_000" in APP_SOURCE
    assert "10_000_000" in APP_SOURCE
    assert "Submit 5-replicate Slurm array" in APP_SOURCE
    assert "time_limit" in APP_SOURCE
    assert "Scientific axes" in APP_SOURCE
    assert "Displayed axes" in APP_SOURCE
    assert "Cancel Slurm array" in APP_SOURCE
    assert "Resume incomplete tasks" in APP_SOURCE
    assert "MAX_SAFE_FULL_RENDER_AXES" in APP_SOURCE
    assert "production_grid_sweep" not in APP_SOURCE


def test_visualizer_python_sources_compile() -> None:
    for path in ROOT.glob("*.py"):
        compile(path.read_text(), path.as_posix(), "exec")


def test_streamlit_application_starts_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception
