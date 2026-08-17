#!/usr/bin/env python3
"""Deterministic 3D stochastic root-architecture simulation and analysis.

The module provides the scientific engine, quantitative metrics, checkpointing,
and local or sharded batch execution. See ``docs/model_design.md`` for model
semantics and numerical assumptions.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


GRID_THICKNESS_COUNT = 70
GRID_RAIN_COUNT = 99
GRID_BRANCH_COUNT = 99
GRID_REPLICATES = 5
SPATIAL_REBUILD_THRESHOLD = 256
MAX_REPORTED_STRAHLER_ORDER = 8
RESOURCE_MODEL_VERSION = "dynamic-transport-availability-demand-tip-focus-v26"
DIRECTION_MODEL_VERSION = "resource-conditioned-escape-diverse-bounded-upward-v26"
CURVE_MODEL_VERSION = "material-sites-transport-taper-age-decaying-laterals-v26"
INITIATION_MODEL_VERSION = "lineage-only-counter-based-stream-v25"
INITIATION_RANDOM_STREAM_VERSION = "splitmix64-site-trial-v1"
SCHEMA_VERSION = 26
CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS = 0.20
BRANCH_RETRY_MODES = ("single_trial", "retry_open_sites")
CAPTURE_REPORTING_EPSILON = 1e-12
MAX_REPORTED_BRANCH_GENERATION = 6
# Post-initiation primordium geometry never changes branch probability.
PRIMORDIUM_STIMULUS_AXIAL_SCALE = 0.34
PRIMORDIUM_STIMULUS_LIFETIME_STEPS = 18.0
PHYSICAL_ORIGIN_ATTEMPTS = 4
# Detection-only thresholds. These are read exclusively after growth finishes.
POSTHOC_WHORL_MIN_BRANCHES = 3
POSTHOC_WHORL_AXIAL_WINDOW = 0.45
# Fixed direction weights encode water-led elongation with mineral support.
RESOURCE_AVAILABILITY_SCORE_WEIGHT = 0.65
WATER_AVAILABILITY_DIRECTION_WEIGHT = 1.35
PHOSPHORUS_AVAILABILITY_DIRECTION_WEIGHT = 1.10
NITROGEN_AVAILABILITY_DIRECTION_WEIGHT = 0.90
POTASSIUM_AVAILABILITY_DIRECTION_WEIGHT = 0.70
RESOURCE_NAMES = ("water", "phosphorus", "nitrogen", "potassium")
RESOURCE_FOCI = (*RESOURCE_NAMES, "balanced")
DIRECTION_PROBE_DIRECTIONS = np.asarray([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
], dtype=np.float64)
DIRECTION_FIXED_TARGETS = (
    np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
)
LATERAL_AGE_GROUPS = (
    ("0_2", 0, 2),
    ("3_5", 3, 5),
    ("6_10", 6, 10),
    ("11_25", 11, 25),
    ("gt_25", 26, None),
)
TOTAL_GRID_TASKS = (
    GRID_THICKNESS_COUNT
    * GRID_RAIN_COUNT
    * GRID_BRANCH_COUNT
    * GRID_REPLICATES
)

MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """Return a stable, well-mixed unsigned 64-bit integer."""

    z = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def seed_for_task(master_seed: int, task_index: int) -> int:
    """Derive a scheduling-independent seed for a grid task."""

    return splitmix64((int(master_seed) & MASK64) ^ splitmix64(task_index))


def initiation_probability_uniform(
    simulation_seed: int,
    task_index: int,
    axis_lineage_identifier: int,
    site_id: int,
    trial_number: int,
) -> float:
    """Return one counter-based initiation variate in ``[0, 1)``.

    ``simulation_seed`` is itself derived from the master seed for grid tasks.
    No mutable generator state or environmental value enters this function, so
    matching axis/site/trial identifiers retain the same draw even when resource
    settings change geometry, focus, emergence, collision ordering, or rendering.
    Trial numbers are one-based in the developmental core.
    """

    counters = (
        int(simulation_seed),
        int(task_index),
        int(axis_lineage_identifier),
        int(site_id),
        int(trial_number),
    )
    mixed = 0xA0761D6478BD642F
    salts = (
        0xE7037ED1A0B428DB,
        0x8EBC6AF09C88C6E3,
        0x589965CC75374CC3,
        0x1D8E4E27C47D124F,
        0xEB44ACCAB455D165,
    )
    for value, salt in zip(counters, salts):
        mixed = splitmix64(mixed ^ splitmix64((value & MASK64) ^ salt))
    return float((mixed >> 11) * (1.0 / float(1 << 53)))


def initiation_probability_passes(
    simulation_seed: int,
    task_index: int,
    axis_lineage_identifier: int,
    site_id: int,
    trial_number: int,
    threshold: float,
) -> bool:
    """Evaluate one resource-independent branch-initiation Bernoulli trial."""

    return initiation_probability_uniform(
        simulation_seed,
        task_index,
        axis_lineage_identifier,
        site_id,
        trial_number,
    ) < float(np.clip(threshold, 0.0, 0.99))


@dataclass(frozen=True)
class SimulationConfig:
    steps: int = 500
    # Anchor elongation decays independently of lateral segment length.
    segment_length: float = 0.5
    anchor_initial_segment_length: float = 0.30
    anchor_min_segment_length: float = 0.05
    anchor_decay_timescale: float = 75.0
    anchor_jitter: float = 0.02
    base_radius: float = 0.05
    balloon_scale: float = 0.05
    angular_clearance: float = 0.002
    spatial_clearance: float = 0.002
    min_pitch_degrees: float = 10.0
    max_pitch_degrees: float = 80.0
    angle_candidates: int = 24
    nutrient_sensitivity: float = 5.0
    # Compatibility alias; resource-specific sensing distances are authoritative.
    nutrient_sensing_distance: float = 5.0
    nutrient_capture_per_iteration: float = 0.1
    nutrient_choice_temperature: float = 0.15
    # Dimensionless capture rates apply per exposed cylindrical surface area.
    soil_water_background: float = 0.20
    rain_water_input: float = 0.80
    water_infiltration_depth: float = 6.0
    water_mobility: float = 1.0
    water_sensing_distance: float = 4.0
    water_capture_per_iteration: float = 0.040
    water_direction_weight: float = 0.60
    phosphorus_concentration: float = 0.90
    phosphorus_z_low: float = -3.0
    phosphorus_z_high: float = 0.0
    phosphorus_mobility: float = 0.05
    phosphorus_sensing_distance: float = 1.5
    phosphorus_capture_per_iteration: float = 0.020
    phosphorus_direction_weight: float = 1.20
    nitrogen_concentration: float = 0.80
    nitrogen_z_low: float = -15.0
    nitrogen_z_high: float = -10.0
    nitrogen_mobility: float = 0.80
    nitrogen_sensing_distance: float = 5.0
    nitrogen_capture_per_iteration: float = 0.030
    nitrogen_direction_weight: float = 1.00
    nitrogen_rain_leaching_depth: float = 1.50
    potassium_concentration: float = 0.70
    potassium_z_low: float = -8.0
    potassium_z_high: float = -3.0
    potassium_mobility: float = 0.35
    potassium_sensing_distance: float = 3.0
    potassium_capture_per_iteration: float = 0.025
    potassium_direction_weight: float = 0.80
    potassium_rain_leaching_depth: float = 0.35
    resource_dispersion_scale: float = 2.0
    # Weak resource gradients favor gravitropic over lateral candidates.
    gravitropism_weight: float = 1.15
    plagiotropism_weight: float = 0.18
    upward_growth_penalty: float = 2.75
    baseline_downward_bias: float = 0.35
    low_resource_lateral_suppression: float = 1.05
    resource_lateral_exploration_weight: float = 0.42
    resource_signal_half_saturation: float = 0.25
    # Smooth-axis persistence, bend, emergence, and age-decay parameters.
    lateral_relative_elongation: float = 0.42
    lateral_elongation_decay_timescale: float = 30.0
    lateral_min_segment_length: float = 0.005
    tip_direction_persistence: float = 1.95
    tip_max_bend_degrees: float = 14.0
    tip_elongation_candidates: int = 10
    tip_choice_temperature: float = 0.12
    lateral_branch_initial_scale: float = 0.55
    lateral_branch_min_age: int = 2
    # Emergence follows local sufficiency and stochastic resource focus.
    lateral_emergence_tolerance_degrees: float = 24.0
    lateral_emergence_min_degrees: float = 20.0
    lateral_emergence_max_degrees: float = 105.0
    lateral_emergence_score_weight: float = 0.80
    lateral_radial_balance_weight: float = 0.15
    sharp_turn_threshold_degrees: float = 45.0
    hard_fork_continuation_angle_degrees: float = 25.0
    compute_convex_hull: bool = False
    max_nodes: int = 100_000
    max_sampled_points: int = 100_000
    # Compatibility alias that can only tighten the sampled-point cap.
    target_architecture_size: int = 0
    target_axis_count: int = 0
    max_growth_iterations: int = 0
    interactive_safety_cap: int = 100_000
    max_seconds_per_simulation: float = 300.0
    initial_capacity: int = 1_024
    strict_lineage_collisions: bool = False
    # Growth and branch origins use material arc; samples support collision/rendering.
    curve_samples_per_extension: int = 5
    branch_min_distance_from_tip: float = 0.45
    branch_min_distance_from_base: float = 0.25
    branch_min_spacing_along_axis: float = CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS
    branch_retry_mode: str = "single_trial"
    branch_collar_clearance_factor: float = 0.90
    branch_collar_safety_margin: float = 0.002
    branch_candidate_temperature: float = 0.22
    anchor_curve_max_bend_degrees: float = 8.0
    lateral_curve_max_bend_degrees: float = 18.0
    anchor_downward_bias_weight: float = 2.30
    lateral_downward_bias_weight: float = 0.65
    lateral_plagiotropic_bias_weight: float = 0.45
    stochastic_tangent_noise: float = 0.18
    resource_patch_count: int = 28
    resource_patch_strength: float = 0.45
    water_patch_strength: float = 0.30
    mean_curve_collision_samples_per_growth_target: float = 5.0
    soil_surface_z: float = 0.0
    max_above_surface_tolerance: float = 0.05
    above_surface_penalty: float = 10.0
    reject_above_surface_curves: bool = True
    enable_resource_demand_feedback: bool = True
    resource_demand_feedback_strength: float = 0.75
    resource_demand_half_saturation: float = 0.15
    resource_capture_ema_alpha: float = 0.18
    resource_deficiency_ema_alpha: float = 0.16
    resource_demand_weight_cap: float = 2.25
    resource_focus_persistence_steps: int = 14
    resource_focus_update_probability: float = 0.08
    resource_focus_balanced_floor: float = 0.18
    maximum_consecutive_upward_extensions: int = 2
    upward_resource_gain_threshold: float = 0.018
    upward_component_threshold: float = 0.12
    # Environmental time constants use absolute soil coordinates anchored at z=0.
    rain_ema_alpha: float = 0.12
    wetting_front_rain_scale: float = 25.0
    nitrate_transport_rain_scale: float = 0.18
    potassium_transport_rain_scale: float = 0.06
    phosphorus_retention_rain_scale: float = 0.0005
    # Starvation uses absolute availability and acts only after initiation.
    water_support_half_saturation: float = 0.16
    nutrient_support_half_saturation: float = 0.14
    water_only_branch_support: float = 0.18
    starvation_lateral_suppression: float = 1.75
    starvation_downward_weight: float = 5.00
    starvation_upward_penalty: float = 14.0
    starvation_horizontal_penalty: float = 7.00
    starvation_axis_confinement_weight: float = 2.50
    starvation_lateral_length_floor: float = 0.65
    starvation_stop_threshold: float = 0.92
    starvation_patience_iterations: int = 90
    phosphorus_topsoil_foraging_weight: float = 1.20
    nitrogen_deep_foraging_weight: float = 1.10
    potassium_midsoil_foraging_weight: float = 0.75
    # Active tips attempt growth once per step; scaling and thickening remain positive.
    lateral_generation_radius_decay: float = 0.65
    lateral_generation_length_decay: float = 0.60
    lateral_generation_probability_exponent: float = 0.85
    lateral_transport_base_fraction: float = 0.16
    tip_extension_candidate_attempts: int = 8
    structural_self_area_coefficient: float = 0.0025
    structural_lateral_self_area_fraction: float = 0.40
    structural_ancestor_area_coefficient: float = 0.0012
    structural_ancestor_transport_decay: float = 0.62
    structural_tip_baseline_fraction: float = 0.25
    branch_origin_child_parent_radius_ratio_limit: float = 0.82
    # New laterals receive a short geometry-only escape corridor.
    lateral_escape_accepted_extensions: int = 2
    lateral_escape_min_outward_component: float = 0.06
    lateral_escape_direction_weight: float = 1.25

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.segment_length <= 0.0:
            raise ValueError("segment_length must be positive")
        if not (
            0.0 < self.anchor_min_segment_length
            <= self.anchor_initial_segment_length
        ):
            raise ValueError(
                "anchor lengths must satisfy 0 < minimum <= initial"
            )
        if self.anchor_decay_timescale <= 0.0:
            raise ValueError("anchor_decay_timescale must be positive")
        if self.base_radius <= 0.0 or self.balloon_scale < 0.0:
            raise ValueError("radius parameters must be non-negative")
        if not 0.0 <= self.anchor_jitter < 1.0:
            raise ValueError("anchor_jitter must be in [0, 1)")
        if not 0.0 < self.min_pitch_degrees <= self.max_pitch_degrees < 90.0:
            raise ValueError("pitch bounds must satisfy 0 < min <= max < 90")
        if self.angle_candidates < 4:
            raise ValueError("angle_candidates must be at least 4")
        if self.nutrient_choice_temperature <= 0.0:
            raise ValueError("nutrient_choice_temperature must be positive")
        if self.nutrient_sensing_distance <= 0.0:
            raise ValueError("nutrient_sensing_distance must be positive")
        bounded_resources = (
            self.soil_water_background,
            self.rain_water_input,
            self.phosphorus_concentration,
            self.nitrogen_concentration,
            self.potassium_concentration,
        )
        if any(value < 0.0 or value > 1.0 for value in bounded_resources):
            raise ValueError("resource backgrounds/concentrations must be in [0, 1]")
        positive_resource_values = (
            self.water_infiltration_depth,
            self.water_sensing_distance,
            self.phosphorus_sensing_distance,
            self.nitrogen_sensing_distance,
            self.potassium_sensing_distance,
            self.resource_dispersion_scale,
        )
        if any(value <= 0.0 for value in positive_resource_values):
            raise ValueError("resource depths, sensing distances, and scales must be positive")
        nonnegative_resource_values = (
            self.water_mobility,
            self.phosphorus_mobility,
            self.nitrogen_mobility,
            self.potassium_mobility,
            self.water_capture_per_iteration,
            self.phosphorus_capture_per_iteration,
            self.nitrogen_capture_per_iteration,
            self.potassium_capture_per_iteration,
            self.water_direction_weight,
            self.phosphorus_direction_weight,
            self.nitrogen_direction_weight,
            self.potassium_direction_weight,
            self.nitrogen_rain_leaching_depth,
            self.potassium_rain_leaching_depth,
            self.nutrient_capture_per_iteration,
            self.gravitropism_weight,
            self.plagiotropism_weight,
            self.upward_growth_penalty,
            self.baseline_downward_bias,
            self.low_resource_lateral_suppression,
            self.resource_lateral_exploration_weight,
            self.lateral_relative_elongation,
            self.lateral_elongation_decay_timescale,
            self.lateral_min_segment_length,
            self.tip_direction_persistence,
            self.tip_choice_temperature,
            self.lateral_branch_initial_scale,
            self.lateral_emergence_score_weight,
            self.lateral_radial_balance_weight,
            self.branch_min_distance_from_tip,
            self.branch_min_distance_from_base,
            self.branch_min_spacing_along_axis,
            self.anchor_downward_bias_weight,
            self.lateral_downward_bias_weight,
            self.lateral_plagiotropic_bias_weight,
            self.stochastic_tangent_noise,
            self.resource_patch_strength,
            self.water_patch_strength,
            self.max_above_surface_tolerance,
            self.above_surface_penalty,
            self.resource_demand_feedback_strength,
            self.resource_demand_half_saturation,
            self.resource_capture_ema_alpha,
            self.resource_deficiency_ema_alpha,
            self.resource_demand_weight_cap,
            self.resource_focus_update_probability,
            self.resource_focus_balanced_floor,
            self.upward_resource_gain_threshold,
            self.upward_component_threshold,
            self.rain_ema_alpha,
            self.wetting_front_rain_scale,
            self.nitrate_transport_rain_scale,
            self.potassium_transport_rain_scale,
            self.phosphorus_retention_rain_scale,
            self.water_support_half_saturation,
            self.nutrient_support_half_saturation,
            self.water_only_branch_support,
            self.starvation_lateral_suppression,
            self.starvation_downward_weight,
            self.starvation_upward_penalty,
            self.starvation_horizontal_penalty,
            self.starvation_axis_confinement_weight,
            self.starvation_lateral_length_floor,
            self.starvation_stop_threshold,
            self.phosphorus_topsoil_foraging_weight,
            self.nitrogen_deep_foraging_weight,
            self.potassium_midsoil_foraging_weight,
            self.branch_collar_clearance_factor,
            self.branch_collar_safety_margin,
            self.structural_self_area_coefficient,
            self.structural_lateral_self_area_fraction,
            self.structural_ancestor_area_coefficient,
        )
        if any(value < 0.0 for value in nonnegative_resource_values):
            raise ValueError("resource mobility, capture, and direction values cannot be negative")
        if not 0.0 <= self.water_only_branch_support <= 1.0:
            raise ValueError("water_only_branch_support must be in [0, 1]")
        if not 0.0 <= self.starvation_lateral_length_floor <= 1.0:
            raise ValueError("starvation_lateral_length_floor must be in [0, 1]")
        if not 0.0 <= self.starvation_stop_threshold <= 1.0:
            raise ValueError("starvation_stop_threshold must be in [0, 1]")
        if self.starvation_patience_iterations < 1:
            raise ValueError("starvation_patience_iterations must be at least 1")
        if self.resource_signal_half_saturation <= 0.0:
            raise ValueError("resource_signal_half_saturation must be positive")
        if not 0.0 < self.resource_capture_ema_alpha <= 1.0:
            raise ValueError("resource_capture_ema_alpha must be in (0, 1]")
        if not 0.0 < self.resource_deficiency_ema_alpha <= 1.0:
            raise ValueError("resource_deficiency_ema_alpha must be in (0, 1]")
        if not 0.0 < self.rain_ema_alpha <= 1.0:
            raise ValueError("rain_ema_alpha must be in (0, 1]")
        if self.resource_focus_persistence_steps < 1:
            raise ValueError("resource_focus_persistence_steps must be at least 1")
        if self.maximum_consecutive_upward_extensions < 0:
            raise ValueError("maximum_consecutive_upward_extensions cannot be negative")
        if self.water_support_half_saturation <= 0.0:
            raise ValueError("water_support_half_saturation must be positive")
        if self.nutrient_support_half_saturation <= 0.0:
            raise ValueError("nutrient_support_half_saturation must be positive")
        if not 0.0 < self.lateral_relative_elongation <= 1.0:
            raise ValueError("lateral_relative_elongation must be in (0, 1]")
        if self.lateral_elongation_decay_timescale <= 0.0:
            raise ValueError("lateral_elongation_decay_timescale must be positive")
        if self.lateral_min_segment_length <= 0.0:
            raise ValueError("lateral_min_segment_length must be positive")
        if self.tip_elongation_candidates < 2:
            raise ValueError("tip_elongation_candidates must be at least 2")
        if self.lateral_branch_initial_scale <= 0.0:
            raise ValueError("lateral_branch_initial_scale must be positive")
        if self.lateral_branch_min_age < 0:
            raise ValueError("lateral_branch_min_age cannot be negative")
        if not 0.0 < self.tip_max_bend_degrees < 90.0:
            raise ValueError("tip_max_bend_degrees must be in (0, 90)")
        if self.lateral_emergence_tolerance_degrees <= 0.0:
            raise ValueError("lateral_emergence_tolerance_degrees must be positive")
        if not (
            0.0
            < self.lateral_emergence_min_degrees
            < self.lateral_emergence_max_degrees
            < 150.0
        ):
            raise ValueError(
                "lateral emergence min/max must satisfy 0 < min < max < 150"
            )
        if not 0.0 < self.hard_fork_continuation_angle_degrees < 90.0:
            raise ValueError("hard_fork_continuation_angle_degrees must be in (0, 90)")
        if not 0.0 < self.sharp_turn_threshold_degrees < 180.0:
            raise ValueError("sharp_turn_threshold_degrees must be in (0, 180)")
        for name, low, high in (
            ("phosphorus", self.phosphorus_z_low, self.phosphorus_z_high),
            ("nitrogen", self.nitrogen_z_low, self.nitrogen_z_high),
            ("potassium", self.potassium_z_low, self.potassium_z_high),
        ):
            if low > high:
                raise ValueError(f"{name}_z_low must not exceed {name}_z_high")
        if self.max_nodes < 2:
            raise ValueError("max_nodes must be at least 2")
        if self.max_sampled_points < 2:
            raise ValueError("max_sampled_points must be at least 2")
        if self.target_architecture_size < 0:
            raise ValueError("target_architecture_size cannot be negative")
        if self.target_axis_count < 0:
            raise ValueError("target_axis_count cannot be negative")
        if self.max_growth_iterations < 0:
            raise ValueError("max_growth_iterations cannot be negative")
        if self.interactive_safety_cap < 2:
            raise ValueError("interactive_safety_cap must be at least 2")
        if self.max_seconds_per_simulation < 0.0:
            raise ValueError("max_seconds_per_simulation cannot be negative")
        if self.initial_capacity < 2:
            raise ValueError("initial_capacity must be at least 2")
        if self.curve_samples_per_extension < 2:
            raise ValueError("curve_samples_per_extension must be at least 2")
        if not 0.0 < self.anchor_curve_max_bend_degrees < 90.0:
            raise ValueError("anchor_curve_max_bend_degrees must be in (0, 90)")
        if not 0.0 < self.lateral_curve_max_bend_degrees < 90.0:
            raise ValueError("lateral_curve_max_bend_degrees must be in (0, 90)")
        if self.resource_patch_count < 0:
            raise ValueError("resource_patch_count cannot be negative")
        if self.lateral_escape_accepted_extensions < 0:
            raise ValueError("lateral_escape_accepted_extensions cannot be negative")
        if not 0.0 <= self.lateral_escape_min_outward_component <= 1.0:
            raise ValueError("lateral_escape_min_outward_component must be in [0, 1]")
        if self.lateral_escape_direction_weight < 0.0:
            raise ValueError("lateral_escape_direction_weight cannot be negative")
        if not 0.0 < self.lateral_generation_radius_decay <= 1.0:
            raise ValueError("lateral_generation_radius_decay must be in (0, 1]")
        if not 0.0 < self.lateral_generation_length_decay <= 1.0:
            raise ValueError("lateral_generation_length_decay must be in (0, 1]")
        if self.lateral_generation_probability_exponent < 0.0:
            raise ValueError("lateral_generation_probability_exponent cannot be negative")
        if self.lateral_transport_base_fraction <= 0.0:
            raise ValueError("lateral_transport_base_fraction must be positive")
        if self.tip_extension_candidate_attempts < 2:
            raise ValueError("tip_extension_candidate_attempts must be at least 2")
        if not 0.0 < self.structural_ancestor_transport_decay <= 1.0:
            raise ValueError("structural_ancestor_transport_decay must be in (0, 1]")
        if not 0.0 < self.structural_lateral_self_area_fraction <= 1.0:
            raise ValueError("structural_lateral_self_area_fraction must be in (0, 1]")
        if not 0.0 < self.structural_tip_baseline_fraction <= 1.0:
            raise ValueError("structural_tip_baseline_fraction must be in (0, 1]")
        if not 0.0 < self.branch_origin_child_parent_radius_ratio_limit < 1.0:
            raise ValueError(
                "branch_origin_child_parent_radius_ratio_limit must be in (0, 1)"
            )
        if self.branch_retry_mode not in BRANCH_RETRY_MODES:
            raise ValueError(
                f"branch_retry_mode must be one of {BRANCH_RETRY_MODES}"
            )
        if self.curve_samples_per_extension < 2:
            raise ValueError("curve_samples_per_extension must be at least 2")
        if self.branch_min_distance_from_tip < 0.0:
            raise ValueError("branch_min_distance_from_tip cannot be negative")
        if self.branch_min_distance_from_base < 0.0:
            raise ValueError("branch_min_distance_from_base cannot be negative")
        if self.branch_min_spacing_along_axis < 0.0:
            raise ValueError("branch_min_spacing_along_axis cannot be negative")
        if self.branch_candidate_temperature <= 0.0:
            raise ValueError("branch_candidate_temperature must be positive")
        nonnegative_curve_values = (
            self.anchor_curve_max_bend_degrees,
            self.lateral_curve_max_bend_degrees,
            self.anchor_downward_bias_weight,
            self.lateral_downward_bias_weight,
            self.lateral_plagiotropic_bias_weight,
            self.stochastic_tangent_noise,
            self.resource_patch_strength,
            self.water_patch_strength,
            self.mean_curve_collision_samples_per_growth_target,
        )
        if any(value < 0.0 for value in nonnegative_curve_values):
            raise ValueError("curve-axis values cannot be negative")
        if self.resource_patch_count < 0:
            raise ValueError("resource_patch_count cannot be negative")


@dataclass(frozen=True)
class SimulationParameters:
    rain_probability: float
    branch_probability: float
    thickness_increment: float
    seed: int
    sim_id: str
    task_index: int = -1

    def validate(self) -> None:
        if not 0.0 <= self.rain_probability <= 1.0:
            raise ValueError("rain_probability must be in [0, 1]")
        if not 0.0 <= self.branch_probability <= 1.0:
            raise ValueError("branch_probability must be in [0, 1]")
        if self.thickness_increment < 0.0:
            raise ValueError("thickness_increment must be non-negative")


def parameters_for_task(task_index: int, master_seed: int) -> SimulationParameters:
    """Decode a grid index without allocating the complete parameter grid."""

    if not 0 <= task_index < TOTAL_GRID_TASKS:
        raise IndexError(f"task_index must be in [0, {TOTAL_GRID_TASKS})")

    remainder = int(task_index)
    replicate = remainder % GRID_REPLICATES
    remainder //= GRID_REPLICATES
    branch_index = remainder % GRID_BRANCH_COUNT
    remainder //= GRID_BRANCH_COUNT
    rain_index = remainder % GRID_RAIN_COUNT
    thickness_index = remainder // GRID_RAIN_COUNT

    thickness = round((thickness_index + 1) * 0.1, 10)
    rain = round((rain_index + 1) * 0.01, 10)
    branch = round((branch_index + 1) * 0.01, 10)
    sim_id = (
        f"R{rain:.2f}_B{branch:.2f}_T{thickness:.2f}_Rep{replicate}"
    )
    return SimulationParameters(
        rain_probability=rain,
        branch_probability=branch,
        thickness_increment=thickness,
        seed=seed_for_task(master_seed, task_index),
        sim_id=sim_id,
        task_index=task_index,
    )


class NodeStore:
    """Structure-of-arrays node registry with geometric capacity growth."""

    __slots__ = (
        "capacity",
        "size",
        "position",
        "direction",
        "edge_length",
        "parent",
        "first_child",
        "next_sibling",
        "child_count",
        "thickness",
        "radius",
        "attachment_angle",
        "depth",
        "birth_step",
        "is_anchor",
        "is_axis_continuation",
        "water_captured",
        "phosphorus_captured",
        "nitrogen_captured",
        "potassium_captured",
        "water_availability_sum",
        "phosphorus_availability_sum",
        "nitrogen_availability_sum",
        "potassium_availability_sum",
        "resource_observations",
        "water_capture_depth_sum",
        "phosphorus_capture_depth_sum",
        "nitrogen_capture_depth_sum",
        "potassium_capture_depth_sum",
        "deepest_water_capture",
        "deepest_phosphorus_capture",
        "deepest_nitrogen_capture",
        "deepest_potassium_capture",
        "direction_resource_score",
        "direction_resource_gate",
        "direction_gravitropism_score",
        "direction_lateral_exploration_score",
        "direction_lateral_suppression_score",
        "branch_opportunities",
        "probability_passes",
        "successful_branches",
        "failed_angle",
        "failed_inflation",
        "failed_spatial",
        "axis_metadata",
    )

    def __init__(self, capacity: int, base_radius: float) -> None:
        self.capacity = max(2, int(capacity))
        self.size = 1
        self.position = np.empty((self.capacity, 3), dtype=np.float64)
        self.direction = np.empty((self.capacity, 3), dtype=np.float64)
        self.edge_length = np.zeros(self.capacity, dtype=np.float64)
        self.parent = np.full(self.capacity, -1, dtype=np.int32)
        self.first_child = np.full(self.capacity, -1, dtype=np.int32)
        self.next_sibling = np.full(self.capacity, -1, dtype=np.int32)
        self.child_count = np.zeros(self.capacity, dtype=np.int32)
        self.thickness = np.zeros(self.capacity, dtype=np.float64)
        self.radius = np.full(self.capacity, base_radius, dtype=np.float64)
        self.attachment_angle = np.full(self.capacity, np.nan, dtype=np.float64)
        self.depth = np.zeros(self.capacity, dtype=np.int32)
        self.birth_step = np.zeros(self.capacity, dtype=np.int32)
        self.is_anchor = np.zeros(self.capacity, dtype=np.bool_)
        self.is_axis_continuation = np.zeros(self.capacity, dtype=np.bool_)
        for name in (
            "water_captured",
            "phosphorus_captured",
            "nitrogen_captured",
            "potassium_captured",
            "water_availability_sum",
            "phosphorus_availability_sum",
            "nitrogen_availability_sum",
            "potassium_availability_sum",
            "water_capture_depth_sum",
            "phosphorus_capture_depth_sum",
            "nitrogen_capture_depth_sum",
            "potassium_capture_depth_sum",
            "deepest_water_capture",
            "deepest_phosphorus_capture",
            "deepest_nitrogen_capture",
            "deepest_potassium_capture",
            "direction_resource_score",
            "direction_resource_gate",
            "direction_gravitropism_score",
            "direction_lateral_exploration_score",
            "direction_lateral_suppression_score",
        ):
            setattr(self, name, np.zeros(self.capacity, dtype=np.float64))
        for name in (
            "resource_observations",
            "branch_opportunities",
            "probability_passes",
            "successful_branches",
            "failed_angle",
            "failed_inflation",
            "failed_spatial",
        ):
            setattr(self, name, np.zeros(self.capacity, dtype=np.int32))

        self.position[0] = (0.0, 0.0, 0.0)
        self.direction[0] = (0.0, 0.0, -1.0)
        self.is_anchor[0] = True
        self.axis_metadata: dict[str, object] = {}

    def _grow(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = max(required, self.capacity + self.capacity // 2)

        def resized(array: np.ndarray, fill: float | int | bool) -> np.ndarray:
            shape = (new_capacity,) + array.shape[1:]
            result = np.full(shape, fill, dtype=array.dtype)
            result[: self.size] = array[: self.size]
            return result

        self.position = resized(self.position, 0.0)
        self.direction = resized(self.direction, 0.0)
        self.edge_length = resized(self.edge_length, 0.0)
        self.parent = resized(self.parent, -1)
        self.first_child = resized(self.first_child, -1)
        self.next_sibling = resized(self.next_sibling, -1)
        self.child_count = resized(self.child_count, 0)
        self.thickness = resized(self.thickness, 0.0)
        self.radius = resized(self.radius, 0.0)
        self.attachment_angle = resized(self.attachment_angle, np.nan)
        self.depth = resized(self.depth, 0)
        self.birth_step = resized(self.birth_step, 0)
        self.is_anchor = resized(self.is_anchor, False)
        self.is_axis_continuation = resized(self.is_axis_continuation, False)
        for name in (
            "water_captured",
            "phosphorus_captured",
            "nitrogen_captured",
            "potassium_captured",
            "water_availability_sum",
            "phosphorus_availability_sum",
            "nitrogen_availability_sum",
            "potassium_availability_sum",
            "water_capture_depth_sum",
            "phosphorus_capture_depth_sum",
            "nitrogen_capture_depth_sum",
            "potassium_capture_depth_sum",
            "deepest_water_capture",
            "deepest_phosphorus_capture",
            "deepest_nitrogen_capture",
            "deepest_potassium_capture",
            "direction_resource_score",
            "direction_resource_gate",
            "direction_gravitropism_score",
            "direction_lateral_exploration_score",
            "direction_lateral_suppression_score",
            "resource_observations",
            "branch_opportunities",
            "probability_passes",
            "successful_branches",
            "failed_angle",
            "failed_inflation",
            "failed_spatial",
        ):
            setattr(self, name, resized(getattr(self, name), 0))
        self.capacity = new_capacity

    def append(
        self,
        *,
        parent: int,
        position: np.ndarray,
        direction: np.ndarray,
        edge_length: float,
        radius: float,
        attachment_angle: float,
        birth_step: int,
        is_anchor: bool,
        is_axis_continuation: bool = False,
        direction_resource_score: float = 0.0,
        direction_resource_gate: float = 0.0,
        direction_gravitropism_score: float = 0.0,
        direction_lateral_exploration_score: float = 0.0,
        direction_lateral_suppression_score: float = 0.0,
    ) -> int:
        node_id = self.size
        self._grow(node_id + 1)
        self.position[node_id] = position
        self.direction[node_id] = direction
        self.edge_length[node_id] = edge_length
        self.parent[node_id] = parent
        self.radius[node_id] = radius
        self.attachment_angle[node_id] = attachment_angle
        self.depth[node_id] = self.depth[parent] + (0 if is_anchor else 1)
        self.birth_step[node_id] = birth_step
        self.is_anchor[node_id] = is_anchor
        self.is_axis_continuation[node_id] = is_axis_continuation
        self.direction_resource_score[node_id] = direction_resource_score
        self.direction_resource_gate[node_id] = direction_resource_gate
        self.direction_gravitropism_score[node_id] = direction_gravitropism_score
        self.direction_lateral_exploration_score[node_id] = direction_lateral_exploration_score
        self.direction_lateral_suppression_score[node_id] = direction_lateral_suppression_score

        self.next_sibling[node_id] = self.first_child[parent]
        self.first_child[parent] = node_id
        self.child_count[parent] += 1
        self.size += 1
        return node_id

    def lateral_children(self, node_id: int) -> Iterator[int]:
        child = int(self.first_child[node_id])
        while child >= 0:
            if not self.is_anchor[child] and not self.is_axis_continuation[child]:
                yield child
            child = int(self.next_sibling[child])

    def path_to_root(self, node_id: int) -> list[int]:
        path: list[int] = []
        current = int(node_id)
        while current >= 0:
            path.append(current)
            current = int(self.parent[current])
        return path

    def relayout(self) -> None:
        """Apply one deterministic balloon-drift layout sweep in O(number nodes)."""

        positions = self.position
        directions = self.direction
        parents = self.parent
        radii = self.radius
        edge_lengths = self.edge_length
        anchors = self.is_anchor
        for node_id in range(1, self.size):
            parent = int(parents[node_id])
            if anchors[node_id]:
                edge_length = float(edge_lengths[node_id])
            else:
                edge_length = (
                    float(edge_lengths[node_id])
                    + float(radii[parent])
                    + float(radii[node_id])
                )
            positions[node_id, 0] = (
                positions[parent, 0] + directions[node_id, 0] * edge_length
            )
            positions[node_id, 1] = (
                positions[parent, 1] + directions[node_id, 1] * edge_length
            )
            positions[node_id, 2] = (
                positions[parent, 2] + directions[node_id, 2] * edge_length
            )


class SameStepGrid:
    """Dynamic center-point grid for nodes absent from the iteration KD-tree."""

    __slots__ = ("cell_size", "inverse_cell_size", "cells")

    def __init__(self, cell_size: float) -> None:
        self.cell_size = float(cell_size)
        self.inverse_cell_size = 1.0 / self.cell_size
        self.cells: dict[tuple[int, int, int], list[int]] = {}

    def _cell(self, point: np.ndarray) -> tuple[int, int, int]:
        scale = self.inverse_cell_size
        return (
            math.floor(float(point[0]) * scale),
            math.floor(float(point[1]) * scale),
            math.floor(float(point[2]) * scale),
        )

    def insert(self, node_id: int, point: np.ndarray) -> None:
        self.cells.setdefault(self._cell(point), []).append(int(node_id))

    def cell_lists(self, point: np.ndarray, reach: float) -> Iterator[list[int]]:
        """Yield occupied nearby cells without a generator frame per node ID."""

        center = self._cell(point)
        cells_out = int(math.ceil(reach / self.cell_size))
        cells = self.cells
        cx, cy, cz = center
        for dx, dy, dz in neighbor_offsets(cells_out):
            values = cells.get((cx + dx, cy + dy, cz + dz))
            if values:
                yield values

    def query(self, point: np.ndarray, reach: float) -> Iterator[int]:
        for values in self.cell_lists(point, reach):
            yield from values


_NEIGHBOR_OFFSET_CACHE: dict[int, tuple[tuple[int, int, int], ...]] = {}


def neighbor_offsets(cells_out: int) -> tuple[tuple[int, int, int], ...]:
    """Cache the small Cartesian stencils used by the dynamic spatial grid."""

    cached = _NEIGHBOR_OFFSET_CACHE.get(cells_out)
    if cached is None:
        extent = range(-cells_out, cells_out + 1)
        cached = tuple((dx, dy, dz) for dx in extent for dy in extent for dz in extent)
        _NEIGHBOR_OFFSET_CACHE[cells_out] = cached
    return cached


class IterationSpatialIndex:
    """Static C KD-tree plus an exact dynamic index for same-step additions."""

    __slots__ = (
        "tree",
        "base_count",
        "dynamic",
        "max_radius",
        "cell_size",
        "dynamic_count",
        "rebuild_threshold",
    )

    def __init__(self, store: NodeStore, cell_size: float) -> None:
        self.cell_size = float(cell_size)
        self.base_count = store.size
        self.tree = cKDTree(store.position[: store.size], compact_nodes=True)
        self.dynamic = SameStepGrid(self.cell_size)
        self.max_radius = float(np.max(store.radius[: store.size]))
        self.dynamic_count = 0
        self.rebuild_threshold = SPATIAL_REBUILD_THRESHOLD

    def insert(self, node_id: int, store: NodeStore) -> None:
        self.dynamic.insert(node_id, store.position[node_id])
        self.dynamic_count += 1
        self.max_radius = max(self.max_radius, float(store.radius[node_id]))
        if self.dynamic_count >= self.rebuild_threshold:
            self.rebuild(store)

    def rebuild(self, store: NodeStore) -> None:
        """Move same-step additions into the C KD-tree without changing geometry."""

        self.base_count = store.size
        self.tree = cKDTree(store.position[: store.size], compact_nodes=True)
        self.dynamic = SameStepGrid(self.cell_size)
        self.dynamic_count = 0
        self.rebuild_threshold = SPATIAL_REBUILD_THRESHOLD

    def query(self, point: np.ndarray, reach: float) -> Iterator[int]:
        yield from self.tree.query_ball_point(point, reach)
        yield from self.dynamic.query(point, reach)


def effective_radius(
    thickness: float | np.ndarray, config: SimulationConfig
) -> float | np.ndarray:
    """Map elastic thickness state to a stable, strictly increasing radius."""

    return config.base_radius + config.balloon_scale * np.log1p(thickness)


def anchor_segment_length(step: int, config: SimulationConfig) -> float:
    """Smooth positive anchor elongation with a non-zero long-run growth rate."""

    excess = (
        config.anchor_initial_segment_length - config.anchor_min_segment_length
    )
    return config.anchor_min_segment_length + excess * math.exp(
        -float(step) / config.anchor_decay_timescale
    )


def angle_is_available(
    node_id: int,
    angle: float,
    proposed_parent_radius: float,
    child_radius: float,
    store: NodeStore,
    clearance: float,
) -> bool:
    """Test dynamic circumference packing for one proposed attachment."""

    for child in store.lateral_children(node_id):
        delta = abs(
            (angle - float(store.attachment_angle[child]) + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        chord = 2.0 * proposed_parent_radius * math.sin(0.5 * delta)
        required = child_radius + float(store.radius[child]) + clearance
        if chord < required:
            return False
    return True


def free_angular_intervals(
    node_id: int,
    proposed_parent_radius: float,
    child_radius: float,
    store: NodeStore,
    clearance: float,
) -> list[tuple[float, float]]:
    """Return the exact free azimuth intervals on a circular parent surface."""

    two_pi = 2.0 * math.pi
    forbidden: list[tuple[float, float]] = []
    for child in store.lateral_children(node_id):
        required = child_radius + float(store.radius[child]) + clearance
        ratio = required / (2.0 * proposed_parent_radius)
        if ratio >= 1.0:
            return []
        half_width = 2.0 * math.asin(ratio)
        center = float(store.attachment_angle[child]) % two_pi
        low = center - half_width
        high = center + half_width
        if low < 0.0:
            forbidden.append((0.0, high))
            forbidden.append((low + two_pi, two_pi))
        elif high >= two_pi:
            forbidden.append((low, two_pi))
            forbidden.append((0.0, high - two_pi))
        else:
            forbidden.append((low, high))

    if not forbidden:
        return [(0.0, two_pi)]

    forbidden.sort()
    merged: list[list[float]] = []
    for low, high in forbidden:
        if not merged or low > merged[-1][1]:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)

    free: list[tuple[float, float]] = []
    cursor = 0.0
    for low, high in merged:
        if low > cursor:
            free.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < two_pi:
        free.append((cursor, two_pi))
    return free


def stratified_angles_from_intervals(
    intervals: Sequence[tuple[float, float]],
    count: int,
    random_offset: float,
) -> np.ndarray:
    """Sample all candidates uniformly over exact free angular measure."""

    lengths = np.fromiter(
        (high - low for low, high in intervals), dtype=np.float64, count=len(intervals)
    )
    total = float(np.sum(lengths))
    if total <= 0.0:
        return np.empty(0, dtype=np.float64)
    stride = total / count
    distances = (np.arange(count, dtype=np.float64) + random_offset) * stride
    cumulative = np.cumsum(lengths)
    interval_indices = np.searchsorted(cumulative, distances, side="right")
    previous = np.where(interval_indices == 0, 0.0, cumulative[interval_indices - 1])
    lows = np.fromiter(
        (interval[0] for interval in intervals),
        dtype=np.float64,
        count=len(intervals),
    )
    return lows[interval_indices] + distances - previous


def sphere_is_clear(
    point: np.ndarray,
    radius: float,
    store: NodeStore,
    index: IterationSpatialIndex,
    clearance: float,
    *,
    excluded_node: int = -1,
    radius_overrides: Mapping[int, float] | None = None,
) -> bool:
    """Exact narrow-phase sphere test after broad-phase spatial lookup."""

    radius_overrides = radius_overrides or {}
    override_max = max(radius_overrides.values(), default=0.0)
    reach = radius + max(index.max_radius, override_max) + clearance
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    positions = store.position
    radii = store.radius
    override_get = radius_overrides.get

    for other_raw in index.tree.query_ball_point(point, reach):
        other = int(other_raw)
        if other == excluded_node:
            continue
        other_radius = float(override_get(other, radii[other]))
        limit = radius + other_radius + clearance
        dx = px - float(positions[other, 0])
        dy = py - float(positions[other, 1])
        dz = pz - float(positions[other, 2])
        if dx * dx + dy * dy + dz * dz < limit * limit:
            return False

    if index.dynamic_count:
        for values in index.dynamic.cell_lists(point, reach):
            for other in values:
                if other == excluded_node:
                    continue
                other_radius = float(override_get(other, radii[other]))
                limit = radius + other_radius + clearance
                dx = px - float(positions[other, 0])
                dy = py - float(positions[other, 1])
                dz = pz - float(positions[other, 2])
                if dx * dx + dy * dy + dz * dz < limit * limit:
                    return False
    return True


def inflated_lineage_is_clear(
    path: Sequence[int],
    proposed_radii: Mapping[int, float],
    store: NodeStore,
    index: IterationSpatialIndex,
    clearance: float,
) -> bool:
    """Validate a tentative path-radius transaction without mutating state."""

    override_max = max(proposed_radii.values(), default=0.0)
    positions = store.position
    parents = store.parent
    radii = store.radius
    proposed_get = proposed_radii.get
    for node_id in path:
        radius = float(proposed_radii[node_id])
        reach = radius + max(index.max_radius, override_max) + clearance
        point = store.position[node_id]
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        node_parent = int(parents[node_id])

        for other_raw in index.tree.query_ball_point(point, reach):
            other = int(other_raw)
            if (
                other == node_id
                or node_parent == other
                or int(parents[other]) == node_id
            ):
                continue
            other_radius = float(proposed_get(other, radii[other]))
            limit = radius + other_radius + clearance
            dx = px - float(positions[other, 0])
            dy = py - float(positions[other, 1])
            dz = pz - float(positions[other, 2])
            if dx * dx + dy * dy + dz * dz < limit * limit:
                return False

        if index.dynamic_count:
            for values in index.dynamic.cell_lists(point, reach):
                for other in values:
                    if (
                        other == node_id
                        or node_parent == other
                        or int(parents[other]) == node_id
                    ):
                        continue
                    other_radius = float(proposed_get(other, radii[other]))
                    limit = radius + other_radius + clearance
                    dx = px - float(positions[other, 0])
                    dy = py - float(positions[other, 1])
                    dz = pz - float(positions[other, 2])
                    if dx * dx + dy * dy + dz * dz < limit * limit:
                        return False
    return True


def inflated_origin_is_clear(
    node_id: int,
    proposed_radius: float,
    store: NodeStore,
    index: IterationSpatialIndex,
    clearance: float,
) -> bool:
    """Fast exact specialization of the default one-node inflation test."""

    point = store.position[node_id]
    px, py, pz = float(point[0]), float(point[1]), float(point[2])
    reach = proposed_radius + index.max_radius + clearance
    positions = store.position
    parents = store.parent
    radii = store.radius
    node_parent = int(parents[node_id])

    for other_raw in index.tree.query_ball_point(point, reach):
        other = int(other_raw)
        if (
            other == node_id
            or node_parent == other
            or int(parents[other]) == node_id
        ):
            continue
        limit = proposed_radius + float(radii[other]) + clearance
        dx = px - float(positions[other, 0])
        dy = py - float(positions[other, 1])
        dz = pz - float(positions[other, 2])
        if dx * dx + dy * dy + dz * dz < limit * limit:
            return False

    if index.dynamic_count:
        for values in index.dynamic.cell_lists(point, reach):
            for other in values:
                if (
                    other == node_id
                    or node_parent == other
                    or int(parents[other]) == node_id
                ):
                    continue
                limit = proposed_radius + float(radii[other]) + clearance
                dx = px - float(positions[other, 0])
                dy = py - float(positions[other, 1])
                dz = pz - float(positions[other, 2])
                if dx * dx + dy * dy + dz * dz < limit * limit:
                    return False
    return True


def distance_to_interval_scalar(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _layer_profile(
    z: np.ndarray,
    low: float,
    high: float,
    concentration: float,
    mobility: float,
    config: SimulationConfig,
) -> np.ndarray:
    """Smooth 1-D availability profile around a nominal soil layer.

    Mobility controls the length of the exponential tail outside the source
    layer. It is a normalized transport parameter, not molecular size.
    """

    distance = np.maximum(np.maximum(low - z, z - high), 0.0)
    dispersion = max(0.02, mobility * config.resource_dispersion_scale)
    return np.clip(concentration * np.exp(-distance / dispersion), 0.0, 1.0)


def _water_profile(
    z: np.ndarray | float, raining: bool, config: SimulationConfig
) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    depth = np.maximum(-values, 0.0)
    rain_pulse = (
        float(raining)
        * config.rain_water_input
        * np.exp(
            -depth
            / max(
                config.water_infiltration_depth * max(config.water_mobility, 0.05),
                1e-12,
            )
        )
    )
    return np.clip(config.soil_water_background + rain_pulse, 0.0, 1.0)


def resource_values(
    z: np.ndarray | float,
    raining: bool,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return independent normalized water, P, N, and K availability profiles.

    Rain adds a surface pulse that infiltrates with depth. Nitrogen and, more
    weakly, potassium profiles shift downward during a rainy iteration to model
    leaching. Phosphorus remains localized because sorption strongly limits its
    mobility. This is a fast profile model, not a water-flow or solute PDE.
    """

    values = np.asarray(z, dtype=np.float64)
    water = _water_profile(values, raining, config)

    phosphorus = _layer_profile(
        values,
        config.phosphorus_z_low,
        config.phosphorus_z_high,
        config.phosphorus_concentration,
        config.phosphorus_mobility,
        config,
    )
    n_shift = float(raining) * config.nitrogen_rain_leaching_depth
    nitrogen = _layer_profile(
        values,
        config.nitrogen_z_low - n_shift,
        config.nitrogen_z_high - n_shift,
        config.nitrogen_concentration,
        config.nitrogen_mobility,
        config,
    )
    k_shift = float(raining) * config.potassium_rain_leaching_depth
    potassium = _layer_profile(
        values,
        config.potassium_z_low - k_shift,
        config.potassium_z_high - k_shift,
        config.potassium_concentration,
        config.potassium_mobility,
        config,
    )
    return water, phosphorus, nitrogen, potassium


def nutrient_values(
    z: np.ndarray, config: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated v3 API returning dry-profile P and N availability."""

    _, phosphorus, nitrogen, _ = resource_values(z, False, config)
    return phosphorus, nitrogen


def _layer_direction_score(
    origin_z: float,
    candidate_z: np.ndarray,
    low: float,
    high: float,
    sensing: float,
    concentration: float,
) -> np.ndarray:
    origin_distance = distance_to_interval_scalar(origin_z, low, high)
    if origin_distance > sensing:
        return np.zeros(candidate_z.shape, dtype=np.float64)
    candidate_distance = np.maximum(
        np.maximum(low - candidate_z, candidate_z - high), 0.0
    )
    progress = np.maximum(0.0, origin_distance - candidate_distance) / sensing
    inside = (candidate_distance == 0.0).astype(np.float64)
    return concentration * (progress + inside)


def _resource_availability_signal(
    candidate_z: np.ndarray,
    raining: bool,
    config: SimulationConfig,
) -> np.ndarray:
    """Return normalized candidate resource availability for direction gating.

    This is deliberately different from a pure gradient score. A wet or
    nutrient-rich local environment can make lateral exploration biologically
    plausible even if every candidate is not moving strictly "uphill" in that
    resource. Inactive resources do not dilute the signal.
    """

    water, phosphorus, nitrogen, potassium = resource_values(
        candidate_z, raining, config
    )
    return resource_availability_signal_from_values(
        water, phosphorus, nitrogen, potassium, raining, config
    )


def resource_availability_signal_from_values(
    water: np.ndarray,
    phosphorus: np.ndarray,
    nitrogen: np.ndarray,
    potassium: np.ndarray,
    raining: bool,
    config: SimulationConfig,
) -> np.ndarray:
    """Normalize active water/P/N/K availability into one bounded signal.

    The signal is used only as a biological bias/gate. It does not collapse the
    resource accounting model: water, P, N, and K capture remain separate.
    """

    weights = np.array(
        [
            WATER_AVAILABILITY_DIRECTION_WEIGHT,
            PHOSPHORUS_AVAILABILITY_DIRECTION_WEIGHT,
            NITROGEN_AVAILABILITY_DIRECTION_WEIGHT,
            POTASSIUM_AVAILABILITY_DIRECTION_WEIGHT,
        ],
        dtype=np.float64,
    )
    numerator = (
        weights[0] * water
        + weights[1] * phosphorus
        + weights[2] * nitrogen
        + weights[3] * potassium
    )
    # Preserve absolute scale so trace concentrations remain weak signals.
    denominator = float(np.sum(weights))
    return np.clip(numerator / max(denominator, 1e-12), 0.0, 1.0)


def local_resource_support(
    water: np.ndarray | float,
    phosphorus: np.ndarray | float,
    nitrogen: np.ndarray | float,
    potassium: np.ndarray | float,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return water, nutrient, construction-support, and starvation gates.

    Water supports elongation, but water alone provides only a small fraction
    of the support needed for building many lateral axes. P/N/K are kept
    separate in direction scoring; this gate only answers whether local growth
    has enough combined support to construct a branch.
    """

    w = np.clip(np.asarray(water, dtype=np.float64), 0.0, 1.0)
    p = np.clip(np.asarray(phosphorus, dtype=np.float64), 0.0, 1.0)
    n = np.clip(np.asarray(nitrogen, dtype=np.float64), 0.0, 1.0)
    k = np.clip(np.asarray(potassium, dtype=np.float64), 0.0, 1.0)
    water_gate = w / (w + max(config.water_support_half_saturation, 1e-12))
    nutrient_raw = (1.20 * p + 1.00 * n + 0.80 * k) / 3.0
    nutrient_gate = nutrient_raw / (
        nutrient_raw + max(config.nutrient_support_half_saturation, 1e-12)
    )
    support = water_gate * (
        config.water_only_branch_support
        + (1.0 - config.water_only_branch_support) * nutrient_gate
    )
    support = np.clip(support, 0.0, 1.0)
    starvation = 1.0 - support
    return water_gate, nutrient_gate, support, starvation


def severe_starvation_response(
    starvation: np.ndarray | float,
) -> np.ndarray:
    """Smoothly activate the strong survival regime only near starvation."""

    x = np.clip((np.asarray(starvation, dtype=np.float64) - 0.68) / 0.32, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class ResourceEnvironmentState:
    """Absolute-coordinate, time-dependent water and nutrient transport state."""

    current_step: int = -1
    cumulative_rain_input: float = 0.0
    recent_rain_ema: float = 0.0
    effective_wetting_depth: float = 0.0
    effective_nitrate_depth: float = 0.0
    effective_potassium_depth: float = 0.0
    phosphorus_retention: float = 1.0

    def update(self, step: int, raining: bool, config: SimulationConfig) -> None:
        self.current_step = int(step)
        rain_input = float(config.rain_water_input) if raining else 0.0
        self.cumulative_rain_input += rain_input
        alpha = float(np.clip(config.rain_ema_alpha, 1e-6, 1.0))
        self.recent_rain_ema += alpha * (float(raining) - self.recent_rain_ema)
        rain_root = math.sqrt(max(self.cumulative_rain_input, 0.0))
        self.effective_wetting_depth = float(
            config.water_infiltration_depth
            * math.sqrt(
                max(self.cumulative_rain_input, 0.0)
                / max(config.wetting_front_rain_scale, 1e-12)
            )
        )
        nitrate_center = 0.5 * (abs(config.nitrogen_z_low) + abs(config.nitrogen_z_high))
        potassium_center = 0.5 * (abs(config.potassium_z_low) + abs(config.potassium_z_high))
        self.effective_nitrate_depth = float(
            nitrate_center
            + config.nitrogen_mobility
            * config.nitrate_transport_rain_scale
            * rain_root
        )
        self.effective_potassium_depth = float(
            potassium_center
            + config.potassium_mobility
            * config.potassium_transport_rain_scale
            * rain_root
        )
        self.phosphorus_retention = float(math.exp(
            -config.phosphorus_mobility
            * config.phosphorus_retention_rain_scale
            * self.cumulative_rain_input
        ))

    def supply_gates(self, config: SimulationConfig) -> np.ndarray:
        """Return whole-profile supply gates; an absent resource is exactly zero."""

        water_supply = float(np.clip(
            config.soil_water_background
            + config.rain_water_input * max(self.recent_rain_ema, 0.0),
            0.0,
            1.0,
        ))
        return np.clip(np.asarray([
            water_supply,
            config.phosphorus_concentration * self.phosphorus_retention,
            config.nitrogen_concentration,
            config.potassium_concentration,
        ], dtype=np.float64), 0.0, 1.0)


class ResourceDemandState:
    """Availability-aware whole-plant demand with capture-rate normalization."""

    TARGET_SHARES = np.asarray([0.35, 0.22, 0.25, 0.18], dtype=np.float64)

    def __init__(self, config: SimulationConfig) -> None:
        self.enabled = bool(config.enable_resource_demand_feedback)
        self.capture_totals = np.zeros(4, dtype=np.float64)
        self.recent_capture_ema = np.zeros(4, dtype=np.float64)
        self._step_capture = np.zeros(4, dtype=np.float64)
        self.supply = np.ones(4, dtype=np.float64)
        self.active_target_shares = self.TARGET_SHARES.copy()
        self.normalized_capture_shares = np.zeros(4, dtype=np.float64)
        self.deficiency = self.TARGET_SHARES.copy()
        self.demand_weights = np.ones(4, dtype=np.float64)

    @staticmethod
    def capture_rates(config: SimulationConfig) -> np.ndarray:
        return np.asarray([
            config.water_capture_per_iteration,
            config.phosphorus_capture_per_iteration,
            config.nitrogen_capture_per_iteration,
            config.potassium_capture_per_iteration,
        ], dtype=np.float64)

    def begin_step(
        self, environment: ResourceEnvironmentState, config: SimulationConfig
    ) -> None:
        self._step_capture.fill(0.0)
        self.supply = environment.supply_gates(config)
        active_targets = self.TARGET_SHARES * (self.supply > CAPTURE_REPORTING_EPSILON)
        target_total = float(np.sum(active_targets))
        self.active_target_shares = (
            active_targets / target_total
            if target_total > CAPTURE_REPORTING_EPSILON
            else np.zeros(4, dtype=np.float64)
        )
        rates = np.maximum(self.capture_rates(config), CAPTURE_REPORTING_EPSILON)
        comparable = self.capture_totals / rates
        comparable *= self.supply > CAPTURE_REPORTING_EPSILON
        comparable_total = float(np.sum(comparable))
        self.normalized_capture_shares = (
            comparable / comparable_total
            if comparable_total > CAPTURE_REPORTING_EPSILON
            else np.zeros(4, dtype=np.float64)
        )
        raw_deficiency = np.clip(
            self.active_target_shares - self.normalized_capture_shares,
            0.0,
            1.0,
        ) * (self.supply > CAPTURE_REPORTING_EPSILON)
        alpha = float(np.clip(config.resource_deficiency_ema_alpha, 1e-6, 1.0))
        self.deficiency += alpha * (raw_deficiency - self.deficiency)
        half = max(float(config.resource_demand_half_saturation), 1e-12)
        response = self.deficiency / (half + self.deficiency)
        if self.enabled:
            self.demand_weights = self.supply * (
                1.0 + config.resource_demand_feedback_strength * response
            )
            self.demand_weights = np.minimum(
                self.demand_weights, config.resource_demand_weight_cap
            )
        else:
            self.demand_weights = (self.supply > CAPTURE_REPORTING_EPSILON).astype(float)

    def end_step(self, config: SimulationConfig) -> None:
        alpha = float(np.clip(config.resource_capture_ema_alpha, 1e-6, 1.0))
        rates = np.maximum(self.capture_rates(config), CAPTURE_REPORTING_EPSILON)
        comparable_step = self._step_capture / rates
        self.recent_capture_ema += alpha * (
            comparable_step - self.recent_capture_ema
        )

    def weights(self, config: SimulationConfig) -> np.ndarray:
        del config
        return self.demand_weights.copy()

    def shares(self) -> np.ndarray:
        return self.normalized_capture_shares.copy()

    def balance_error(self) -> float:
        return float(np.sum(np.abs(
            self.normalized_capture_shares - self.active_target_shares
        )))

    def focus_probabilities(
        self, local_values: np.ndarray, config: SimulationConfig
    ) -> np.ndarray:
        active = (self.supply > CAPTURE_REPORTING_EPSILON).astype(np.float64)
        opportunity = np.clip(np.asarray(local_values, dtype=np.float64), 0.0, 1.0)
        scores = active * (
            0.30 * self.active_target_shares
            + 0.50 * self.deficiency
            + 0.20 * opportunity
        )
        balanced = (
            max(config.resource_focus_balanced_floor, 1e-6)
            if float(np.sum(active)) > 0.0 else 1.0
        )
        probabilities = np.append(scores, balanced)
        return probabilities / max(float(np.sum(probabilities)), 1e-12)

    def accumulate_points(
        self,
        points: np.ndarray,
        resource_field: "HeterogeneousResourceField",
        raining: bool,
        config: SimulationConfig,
        *,
        surface_scale: float = 1.0,
    ) -> None:
        if not self.enabled or points.size == 0:
            return
        values = np.vstack(resource_field.values(
            np.asarray(points, dtype=np.float64), raining, config
        ))
        increment = (
            np.sum(values, axis=1)
            * self.capture_rates(config)
            * max(float(surface_scale), 0.0)
        )
        self.capture_totals += increment
        self._step_capture += increment


def demand_weighted_resource_signal(
    water: np.ndarray,
    phosphorus: np.ndarray,
    nitrogen: np.ndarray,
    potassium: np.ndarray,
    raining: bool,
    config: SimulationConfig,
    demand_state: ResourceDemandState | None = None,
) -> np.ndarray:
    base = resource_availability_signal_from_values(
        water, phosphorus, nitrogen, potassium, raining, config
    )
    if demand_state is None or not demand_state.enabled:
        return base
    demand = demand_state.weights(config)
    weighted = (
        demand[0] * WATER_AVAILABILITY_DIRECTION_WEIGHT * water
        + demand[1] * PHOSPHORUS_AVAILABILITY_DIRECTION_WEIGHT * phosphorus
        + demand[2] * NITROGEN_AVAILABILITY_DIRECTION_WEIGHT * nitrogen
        + demand[3] * POTASSIUM_AVAILABILITY_DIRECTION_WEIGHT * potassium
    )
    weights = np.asarray(
        [
            WATER_AVAILABILITY_DIRECTION_WEIGHT,
            PHOSPHORUS_AVAILABILITY_DIRECTION_WEIGHT,
            NITROGEN_AVAILABILITY_DIRECTION_WEIGHT,
            POTASSIUM_AVAILABILITY_DIRECTION_WEIGHT,
        ],
        dtype=np.float64,
    )
    # Under-capture amplifies existing gradients but cannot create supply.
    denominator = float(np.sum(weights))
    blended = np.clip(weighted / max(denominator, 1e-12), 0.0, 1.0)
    return np.clip(0.45 * base + 0.55 * blended, 0.0, 1.0)


def resource_direction_scores(
    origin_z: float,
    candidate_z: np.ndarray,
    raining: bool,
    config: SimulationConfig,
) -> np.ndarray:
    """Combine explicit resource-specific directional signals and weights."""

    origin_water = float(_water_profile(origin_z, raining, config))
    candidate_water = _water_profile(candidate_z, raining, config)
    water_change = np.maximum(0.0, candidate_water - origin_water)
    water_score = water_change / max(config.water_sensing_distance, 1e-12)

    n_shift = float(raining) * config.nitrogen_rain_leaching_depth
    k_shift = float(raining) * config.potassium_rain_leaching_depth
    phosphorus_score = _layer_direction_score(
        origin_z, candidate_z,
        config.phosphorus_z_low, config.phosphorus_z_high,
        config.phosphorus_sensing_distance, config.phosphorus_concentration,
    )
    nitrogen_score = _layer_direction_score(
        origin_z, candidate_z,
        config.nitrogen_z_low - n_shift, config.nitrogen_z_high - n_shift,
        config.nitrogen_sensing_distance, config.nitrogen_concentration,
    )
    potassium_score = _layer_direction_score(
        origin_z, candidate_z,
        config.potassium_z_low - k_shift, config.potassium_z_high - k_shift,
        config.potassium_sensing_distance, config.potassium_concentration,
    )
    gradient_score = (
        config.water_direction_weight * water_score
        + config.phosphorus_direction_weight * phosphorus_score
        + config.nitrogen_direction_weight * nitrogen_score
        + config.potassium_direction_weight * potassium_score
    )
    availability_score = _resource_availability_signal(candidate_z, raining, config)
    return gradient_score + RESOURCE_AVAILABILITY_SCORE_WEIGHT * availability_score


def derived_direction_modifiers(
    resource_gate: np.ndarray | float,
    depth: np.ndarray | float,
    config: SimulationConfig,
) -> dict[str, np.ndarray]:
    """Derive internal direction weights from local biology/environment.

    These are intentionally not app-facing knobs. Dry or resource-poor local
    conditions strengthen persistence/downward search and suppress strong
    lateral wandering. Wet/resource-rich conditions relax that suppression and
    permit more plagiotropic/lateral exploration. Depth adds a mild stabilizing
    effect so deep axes do not whip sideways too easily.
    """

    gate = np.clip(np.asarray(resource_gate, dtype=np.float64), 0.0, 1.0)
    depth_values = np.maximum(np.asarray(depth, dtype=np.float64), 0.0)
    dryness = 1.0 - gate
    depth_signal = depth_values / (depth_values + 6.0)
    return {
        "gravitropism_weight": config.gravitropism_weight
        * (1.0 + 0.35 * dryness + 0.12 * depth_signal),
        "plagiotropism_weight": config.plagiotropism_weight
        * (0.65 + 0.85 * gate),
        "upward_growth_penalty": config.upward_growth_penalty
        * (1.0 + 0.40 * dryness + 0.15 * depth_signal),
        "lateral_suppression_weight": config.low_resource_lateral_suppression
        * (0.35 + 0.90 * dryness)
        * (1.0 + 0.12 * depth_signal),
        "lateral_exploration_weight": config.resource_lateral_exploration_weight
        * (0.35 + 1.45 * gate),
        "downward_bias": config.baseline_downward_bias
        * (0.75 + 0.65 * dryness + 0.10 * depth_signal),
    }


def direction_component_scores(
    node_position: np.ndarray,
    directions: np.ndarray,
    edge_length: float,
    raining: bool,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score arbitrary directions with the shared resource/gravity model."""

    candidate_z = node_position[2] + directions[:, 2] * edge_length
    resource_score = resource_direction_scores(
        float(node_position[2]), candidate_z, raining, config
    )
    resource_gate = resource_score / (
        resource_score + config.resource_signal_half_saturation
    )
    resource_gate = np.clip(resource_gate, 0.0, 1.0)
    modifiers = derived_direction_modifiers(
        resource_gate, np.maximum(-candidate_z, 0.0), config
    )
    downward_component = np.clip(-directions[:, 2], 0.0, 1.0)
    upward_component = np.clip(directions[:, 2], 0.0, 1.0)
    lateral_component = np.linalg.norm(directions[:, :2], axis=1)
    low_resource_gate = 1.0 - resource_gate

    gravitropism_score = (
        modifiers["gravitropism_weight"]
        * (modifiers["downward_bias"] + 0.45 * low_resource_gate)
        * downward_component
    )
    lateral_exploration_score = (
        modifiers["lateral_exploration_weight"]
        * resource_gate
        * lateral_component
    )
    plagiotropism_score = modifiers["plagiotropism_weight"] * lateral_component
    upward_penalty = (
        modifiers["upward_growth_penalty"] * upward_component * upward_component
    )
    lateral_suppression_score = (
        modifiers["gravitropism_weight"]
        * modifiers["lateral_suppression_weight"]
        * low_resource_gate
        * lateral_component
    )
    base_scores = (
        config.nutrient_sensitivity * resource_score
        + gravitropism_score
        + lateral_exploration_score
        + plagiotropism_score
        - lateral_suppression_score
        - upward_penalty
    )
    return (
        base_scores,
        resource_score,
        resource_gate,
        gravitropism_score,
        lateral_exploration_score,
        lateral_suppression_score,
    )


def candidate_directions(
    node_position: np.ndarray,
    incoming_direction: np.ndarray,
    edge_length: float,
    angles: np.ndarray,
    pitches: np.ndarray,
    uniform: np.ndarray,
    raining: bool,
    config: SimulationConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate parent-relative lateral branch candidates and score them.

    Lateral roots are born in the local coordinate frame of the parent root
    axis, not in the global downward hemisphere. Gravity, plagiotropism, radial
    exploration, and resource gradients are soft ranking terms applied after
    those parent-relative directions exist.
    """

    incoming = normalized_direction(incoming_direction)
    u, v = local_orthonormal_frame(incoming)
    emergence = np.asarray(pitches, dtype=np.float64)
    lateral_ring = (
        np.cos(angles)[:, None] * u[None, :]
        + np.sin(angles)[:, None] * v[None, :]
    )
    directions = (
        np.cos(emergence)[:, None] * incoming[None, :]
        + np.sin(emergence)[:, None] * lateral_ring
    )
    norms = np.linalg.norm(directions, axis=1)
    np.divide(directions, norms[:, None], out=directions, where=norms[:, None] > 1e-12)
    (
        base_scores,
        resource_score,
        resource_gate,
        gravitropism_score,
        lateral_exploration_score,
        lateral_suppression_score,
    ) = direction_component_scores(
        node_position, directions, edge_length, raining, config
    )

    divergence = np.arccos(np.clip(directions @ incoming, -1.0, 1.0))
    preferred = 0.5 * (
        math.radians(config.lateral_emergence_min_degrees)
        + math.radians(config.lateral_emergence_max_degrees)
    )
    tolerance = math.radians(config.lateral_emergence_tolerance_degrees)
    lateral_emergence_score = config.lateral_emergence_score_weight * np.exp(
        -0.5 * ((divergence - preferred) / tolerance) ** 2
    )
    radial = horizontal_radial_direction(node_position)
    if float(np.linalg.norm(radial)) > 0.0:
        horizontal_norm = np.linalg.norm(directions[:, :2], axis=1)
        horizontal_unit = np.zeros_like(directions[:, :2])
        np.divide(
            directions[:, :2],
            horizontal_norm[:, None],
            out=horizontal_unit,
            where=horizontal_norm[:, None] > 1e-12,
        )
        radial_alignment = horizontal_unit @ radial
        radial_balance_score = (
            config.lateral_radial_balance_weight
            * radial_alignment
            * horizontal_norm
        )
    else:
        radial_balance_score = np.zeros(directions.shape[0], dtype=np.float64)

    # Gumbel-max sampling retains stochastic preference under gravity.
    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    gumbel = -np.log(-np.log(uniform))
    scores = (
        base_scores
        + lateral_emergence_score
        + radial_balance_score
        + config.nutrient_choice_temperature * gumbel
    )
    return (
        angles,
        directions,
        scores,
        resource_score,
        resource_gate,
        gravitropism_score,
        lateral_exploration_score,
        lateral_suppression_score,
    )


def commit_lineage_inflation(
    path: Sequence[int],
    thickness_increment: float,
    proposed_radii: Mapping[int, float],
    store: NodeStore,
    index: IterationSpatialIndex,
) -> None:
    for node_id in path:
        store.thickness[node_id] += thickness_increment
        store.radius[node_id] = proposed_radii[node_id]
        index.max_radius = max(index.max_radius, float(proposed_radii[node_id]))


def force_lineage_inflation(
    node_id: int,
    thickness_increment: float,
    store: NodeStore,
    config: SimulationConfig,
) -> None:
    """Mandatory anchor-growth inflation; anchor growth is never rejected."""

    current = int(node_id)
    while current >= 0:
        store.thickness[current] += thickness_increment
        store.radius[current] = effective_radius(store.thickness[current], config)
        current = int(store.parent[current])


def direction_diagnostics(
    node_position: np.ndarray,
    direction: np.ndarray,
    edge_length: float,
    raining: bool,
    config: SimulationConfig,
) -> tuple[float, float, float, float, float]:
    """Return the same direction diagnostic components used for branch candidates."""

    candidate_z = np.asarray(
        [float(node_position[2]) + float(direction[2]) * edge_length],
        dtype=np.float64,
    )
    resource_score = float(
        resource_direction_scores(float(node_position[2]), candidate_z, raining, config)[0]
    )
    resource_gate = resource_score / (
        resource_score + config.resource_signal_half_saturation
    )
    resource_gate = float(np.clip(resource_gate, 0.0, 1.0))
    modifiers = derived_direction_modifiers(
        resource_gate,
        max(0.0, -float(candidate_z[0])),
        config,
    )
    downward_component = float(np.clip(-float(direction[2]), 0.0, 1.0))
    lateral_component = float(np.linalg.norm(direction[:2]))
    low_resource_gate = 1.0 - resource_gate
    gravitropism_score = (
        float(modifiers["gravitropism_weight"])
        * (float(modifiers["downward_bias"]) + 0.45 * low_resource_gate)
        * downward_component
    )
    lateral_exploration_score = (
        float(modifiers["lateral_exploration_weight"])
        * resource_gate
        * lateral_component
    )
    lateral_suppression_score = (
        float(modifiers["gravitropism_weight"])
        * float(modifiers["lateral_suppression_weight"])
        * low_resource_gate
        * lateral_component
    )
    return (
        resource_score,
        resource_gate,
        float(gravitropism_score),
        float(lateral_exploration_score),
        float(lateral_suppression_score),
    )


def true_lateral_child_count(store: NodeStore, node_id: int) -> int:
    """Count true lateral branch children, excluding anchor and axis continuations."""

    count = 0
    child = int(store.first_child[node_id])
    while child >= 0:
        if not bool(store.is_anchor[child]) and not bool(store.is_axis_continuation[child]):
            count += 1
        child = int(store.next_sibling[child])
    return count


def normalized_direction(direction: np.ndarray) -> np.ndarray:
    """Return a finite unit vector, falling back to downward growth."""

    base = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(base))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return base / norm


def local_orthonormal_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two stable unit vectors perpendicular to a local root tangent."""

    t = normalized_direction(tangent)
    helper = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(t[2])) > 0.9
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    u = np.cross(t, helper)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 1e-12:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = u / u_norm
    v = np.cross(t, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        v = v / v_norm
    return u, v


def vector_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return the unsigned angle between two direction vectors in degrees."""

    a = normalized_direction(left)
    b = normalized_direction(right)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def horizontal_radial_direction(position: np.ndarray) -> np.ndarray:
    """Return horizontal direction away from the approximate anchor/root center."""

    radial = np.asarray([position[0], position[1]], dtype=np.float64)
    norm = float(np.linalg.norm(radial))
    if norm <= 1e-9:
        return np.zeros(2, dtype=np.float64)
    return radial / norm


def cone_directions_around(
    base_direction: np.ndarray,
    rng: np.random.Generator,
    count: int,
    max_bend_degrees: float,
) -> np.ndarray:
    """Sample candidate directions inside a cone around the incoming axis."""

    base = normalized_direction(base_direction)

    u, v = local_orthonormal_frame(base)

    cone_angle = math.radians(max_bend_degrees)
    directions = np.empty((max(1, int(count)), 3), dtype=np.float64)
    directions[0] = base
    if directions.shape[0] > 1:
        bends = rng.uniform(0.0, cone_angle, size=directions.shape[0] - 1)
        azimuths = rng.uniform(0.0, 2.0 * math.pi, size=directions.shape[0] - 1)
        for offset, (bend, azimuth) in enumerate(zip(bends, azimuths), start=1):
            candidate = (
                math.cos(float(bend)) * base
                + math.sin(float(bend))
                * (math.cos(float(azimuth)) * u + math.sin(float(azimuth)) * v)
            )
            directions[offset] = candidate / np.linalg.norm(candidate)
    return directions


def cone_directions_from_frame(
    base: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
    count: int,
    max_bend_degrees: float,
) -> np.ndarray:
    """Exact cone sampler using a precomputed unchanged local frame."""

    cone_angle = math.radians(max_bend_degrees)
    directions = np.empty((max(1, int(count)), 3), dtype=np.float64)
    directions[0] = base
    if directions.shape[0] > 1:
        bends = rng.uniform(0.0, cone_angle, size=directions.shape[0] - 1)
        azimuths = rng.uniform(
            0.0, 2.0 * math.pi, size=directions.shape[0] - 1
        )
        for offset, (bend, azimuth) in enumerate(
            zip(bends, azimuths), start=1
        ):
            candidate = (
                math.cos(float(bend)) * base
                + math.sin(float(bend))
                * (
                    math.cos(float(azimuth)) * u
                    + math.sin(float(azimuth)) * v
                )
            )
            directions[offset] = candidate / np.linalg.norm(candidate)
    return directions


def lateral_branch_depth_bucket(origin_z: float, config: SimulationConfig) -> int:
    if origin_z >= config.phosphorus_z_low:
        return 0
    if origin_z > config.nitrogen_z_high:
        return 1
    if origin_z >= config.nitrogen_z_low:
        return 2
    return 3


def _sample_without_replacement(
    values: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if values.size <= size:
        return values
    return rng.choice(values, size=size, replace=False)


def _softmax_select_index(scores: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    scaled = scores / max(temperature, 1e-12)
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(np.clip(scaled, -60.0, 60.0))
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return int(rng.integers(0, scores.size))
    threshold = float(rng.random()) * total
    cumulative = np.cumsum(weights)
    return int(np.searchsorted(cumulative, threshold, side="right"))


def _entropy_from_angles(values: np.ndarray, bins: int = 12) -> float:
    """Return normalized circular entropy in [0, 1] for angle-like samples."""

    if values.size <= 1:
        return 0.0
    hist, _ = np.histogram(
        np.mod(values, 2.0 * math.pi),
        bins=bins,
        range=(0.0, 2.0 * math.pi),
    )
    probabilities = hist.astype(np.float64)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return 0.0
    probabilities = probabilities[probabilities > 0.0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(bins))


def _linear_entropy(values: np.ndarray, low: float, high: float, bins: int = 10) -> float:
    if values.size <= 1 or high <= low:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(low, high))
    probabilities = hist.astype(np.float64)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return 0.0
    probabilities = probabilities[probabilities > 0.0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(bins))


@functools.lru_cache(maxsize=16)
def hermite_curve_basis(
    sample_count: int,
) -> tuple[np.ndarray, ...]:
    """Cache immutable scalar Hermite basis arrays for a support-point count."""

    n = max(2, int(sample_count))
    u = np.linspace(1.0 / n, 1.0, n, dtype=np.float64)
    u2 = u * u
    u3 = u2 * u
    return (
        2.0 * u3 - 3.0 * u2 + 1.0,
        u3 - 2.0 * u2 + u,
        -2.0 * u3 + 3.0 * u2,
        u3 - u2,
        6.0 * u2 - 6.0 * u,
        3.0 * u2 - 4.0 * u + 1.0,
        -6.0 * u2 + 6.0 * u,
        3.0 * u2 - 2.0 * u,
    )


def hermite_curve_samples(
    origin: np.ndarray,
    origin_tangent: np.ndarray,
    end_tangent: np.ndarray,
    length: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a short cubic Hermite centerline extension.

    The simulator stores axes as smooth curve samples.  The returned first point
    is not the origin; it contains only newly grown support points.
    """

    n = max(2, int(sample_count))
    p0 = np.asarray(origin, dtype=np.float64)
    t0 = normalized_direction(origin_tangent)
    t1 = normalized_direction(end_tangent)
    p1 = p0 + t1 * float(length)
    m0 = t0 * float(length) * 0.65
    m1 = t1 * float(length) * 0.65
    h00, h10, h01, h11, dh00, dh10, dh01, dh11 = (
        hermite_curve_basis(n)
    )
    samples = (
        h00[:, None] * p0
        + h10[:, None] * m0
        + h01[:, None] * p1
        + h11[:, None] * m1
    )
    derivatives = (
        dh00[:, None] * p0
        + dh10[:, None] * m0
        + dh01[:, None] * p1
        + dh11[:, None] * m1
    )
    norms = np.linalg.norm(derivatives, axis=1)
    tangents = np.divide(
        derivatives,
        norms[:, None],
        out=np.tile(t1, (n, 1)),
        where=norms[:, None] > 1e-12,
    )
    return samples, tangents


class HeterogeneousResourceField:
    """Seeded 3-D patches over schema-v26 time-dependent vertical profiles."""

    __slots__ = (
        "water_centers", "water_widths", "water_amplitudes",
        "phosphorus_centers", "phosphorus_widths", "phosphorus_amplitudes",
        "nitrogen_centers", "nitrogen_widths", "nitrogen_amplitudes",
        "potassium_centers", "potassium_widths", "potassium_amplitudes",
        "environment", "_shifted_centers_step",
        "_shifted_nitrogen_centers", "_shifted_potassium_centers",
    )

    def __init__(self, seed: int, config: SimulationConfig) -> None:
        rng = np.random.default_rng(splitmix64(seed ^ 0xA53A_11C3_42E1_900D))
        self.environment = ResourceEnvironmentState()
        self.environment.update(-1, False, config)
        count = int(config.resource_patch_count)
        self.water_centers, self.water_widths, self.water_amplitudes = (
            self._make_patches(rng, max(6, count // 2), -18.0, 0.0, 7.0, 9.0)
        )
        self.phosphorus_centers, self.phosphorus_widths, self.phosphorus_amplitudes = (
            self._make_patches(rng, count, config.phosphorus_z_low, config.phosphorus_z_high, 3.0, 2.0)
        )
        self.nitrogen_centers, self.nitrogen_widths, self.nitrogen_amplitudes = (
            self._make_patches(rng, count, config.nitrogen_z_low - 4.0, config.nitrogen_z_high + 2.0, 5.0, 5.0)
        )
        self.potassium_centers, self.potassium_widths, self.potassium_amplitudes = (
            self._make_patches(rng, count, config.potassium_z_low - 2.0, config.potassium_z_high + 1.0, 4.0, 3.0)
        )
        self._shifted_centers_step = -2
        self._shifted_nitrogen_centers = self.nitrogen_centers.copy()
        self._shifted_potassium_centers = self.potassium_centers.copy()

    def shifted_mobile_centers(
        self,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact rain-shifted N/K centers, cached for one environment step."""

        env = self.environment
        if self._shifted_centers_step != env.current_step:
            n_centers = self.nitrogen_centers.copy()
            k_centers = self.potassium_centers.copy()
            if n_centers.size:
                initial_n = 0.5 * (
                    abs(config.nitrogen_z_low) + abs(config.nitrogen_z_high)
                )
                n_centers[:, 2] -= max(
                    0.0, env.effective_nitrate_depth - initial_n
                )
            if k_centers.size:
                initial_k = 0.5 * (
                    abs(config.potassium_z_low) + abs(config.potassium_z_high)
                )
                k_centers[:, 2] -= max(
                    0.0, env.effective_potassium_depth - initial_k
                )
            self._shifted_nitrogen_centers = n_centers
            self._shifted_potassium_centers = k_centers
            self._shifted_centers_step = int(env.current_step)
        return (
            self._shifted_nitrogen_centers,
            self._shifted_potassium_centers,
        )

    @staticmethod
    def _make_patches(
        rng: np.random.Generator,
        count: int,
        z_low: float,
        z_high: float,
        xy_scale: float,
        z_width: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if count <= 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        centers = np.empty((count, 3), dtype=np.float64)
        centers[:, 0] = rng.normal(0.0, xy_scale, size=count)
        centers[:, 1] = rng.normal(0.0, xy_scale, size=count)
        centers[:, 2] = rng.uniform(min(z_low, z_high), max(z_low, z_high), size=count)
        widths = np.empty((count, 3), dtype=np.float64)
        widths[:, 0] = rng.uniform(1.2, max(1.3, xy_scale * 0.75), size=count)
        widths[:, 1] = rng.uniform(1.2, max(1.3, xy_scale * 0.75), size=count)
        widths[:, 2] = rng.uniform(0.7, max(0.8, z_width), size=count)
        amplitudes = rng.uniform(0.35, 1.0, size=count)
        return centers, widths, amplitudes

    @staticmethod
    def _patch_signal(
        points: np.ndarray,
        centers: np.ndarray,
        widths: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        if centers.shape[0] == 0 or points.shape[0] == 0:
            return np.zeros(points.shape[0], dtype=np.float64)
        diff = points[:, None, :] - centers[None, :, :]
        scaled = np.sum((diff / np.maximum(widths[None, :, :], 1e-9)) ** 2, axis=2)
        signal = np.exp(-0.5 * scaled) @ amplitudes
        return signal / (1.0 + signal)

    def values(
        self,
        points: np.ndarray,
        raining: bool | float,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        env = self.environment
        depth = np.maximum(config.soil_surface_z - pts[:, 2], 0.0)
        # A smooth wetting front retains a nonuniform deep tail.
        wet_depth = max(env.effective_wetting_depth, 0.25)
        wet_front = np.exp(-0.5 * ((depth - wet_depth) / max(0.35 * wet_depth, 0.8)) ** 2)
        wet_tail = np.exp(-depth / max(1.35 * wet_depth, 1.0))
        base_water = np.clip(
            config.soil_water_background * np.exp(-depth / 18.0)
            + config.rain_water_input
            * (0.55 * env.recent_rain_ema * wet_tail + 0.45 * wet_front),
            0.0,
            1.0,
        )
        # P remains shallow; nitrate and potassium move downward at distinct rates.
        base_p = np.clip(
            config.phosphorus_concentration
            * env.phosphorus_retention
            * np.exp(-depth / max(config.phosphorus_sensing_distance, 0.5)),
            0.0,
            1.0,
        )
        nitrate_width = max(
            2.5, 0.5 * abs(config.nitrogen_z_high - config.nitrogen_z_low) + 2.0
        )
        base_n = np.clip(
            config.nitrogen_concentration
            * np.exp(-0.5 * ((depth - env.effective_nitrate_depth) / nitrate_width) ** 2),
            0.0,
            1.0,
        )
        potassium_width = max(
            1.8, 0.5 * abs(config.potassium_z_high - config.potassium_z_low) + 1.2
        )
        base_k = np.clip(
            config.potassium_concentration
            * np.exp(-0.5 * ((depth - env.effective_potassium_depth) / potassium_width) ** 2),
            0.0,
            1.0,
        )
        water_patch = self._patch_signal(
            pts, self.water_centers, self.water_widths, self.water_amplitudes
        )
        p_patch = self._patch_signal(
            pts, self.phosphorus_centers, self.phosphorus_widths, self.phosphorus_amplitudes
        )
        n_centers, k_centers = self.shifted_mobile_centers(config)
        n_patch = self._patch_signal(
            pts, n_centers, self.nitrogen_widths, self.nitrogen_amplitudes
        )
        k_patch = self._patch_signal(
            pts, k_centers, self.potassium_widths, self.potassium_amplitudes
        )
        water = np.clip(base_water * (0.75 + config.water_patch_strength * water_patch), 0.0, 1.0)
        phosphorus = np.clip(base_p * (0.65 + config.resource_patch_strength * p_patch), 0.0, 1.0)
        nitrogen = np.clip(base_n * (0.65 + config.resource_patch_strength * n_patch), 0.0, 1.0)
        potassium = np.clip(base_k * (0.65 + config.resource_patch_strength * k_patch), 0.0, 1.0)
        return water, phosphorus, nitrogen, potassium

    def combined_signal(
        self,
        points: np.ndarray,
        raining: bool | float,
        config: SimulationConfig,
    ) -> np.ndarray:
        water, phosphorus, nitrogen, potassium = self.values(points, raining, config)
        return resource_availability_signal_from_values(
            water, phosphorus, nitrogen, potassium, bool(float(raining) > 0.0), config
        )


@dataclass
class BranchSite:
    """One biological initiation site fixed to continuous material arc."""

    site_id: int
    axis_id: int
    material_arc: float
    birth_step: int
    first_eligible_step: int
    trial_count: int = 0
    failure_count: int = 0
    probability_pass_count: int = 0
    accepted_branch_count: int = 0
    occupied_azimuths: list[float] = field(default_factory=list)
    accepted_branch_base_radii: list[float] = field(default_factory=list)
    last_trial_step: int = -1
    last_initiation_uniform: float = math.nan
    last_initiation_threshold: float = math.nan
    last_initiation_passed: bool = False
    closed_in_single_trial_mode: bool = False
    physically_open: bool = True
    temporarily_surface_full: bool = False
    last_evaluated_parent_radius: float = 0.0


@dataclass
class RootAxis:
    axis_id: int
    parent_axis_id: int
    parent_arc_length: float
    parent_local_azimuth: float
    birth_step: int
    is_anchor_axis: bool
    branch_generation: int
    points: list[np.ndarray]
    tangents: list[np.ndarray]
    point_birth_steps: list[int]
    material_arcs: list[float] = field(default_factory=lambda: [0.0])
    origin_site_id: int = -1
    structural_area_total: float = 0.0
    active: bool = True
    branch_origins: list[float] = field(default_factory=list)
    branch_azimuths: list[float] = field(default_factory=list)
    branch_origin_steps: list[int] = field(default_factory=list)
    branch_origin_base_radii: list[float] = field(default_factory=list)
    next_branch_site_arc: float | None = None
    branch_site_ids: list[int] = field(default_factory=list)
    # Each range-add event applies area from the axis base through its end arc.
    structural_area_events: list[tuple[float, float]] = field(default_factory=list)
    structural_area_version: int = 0
    _area_cache_version: int = -1
    _area_cache_ends: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    _area_cache_suffix: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    _material_arc_cache_count: int = field(default=-1, repr=False)
    _material_arc_cache: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    radius_scale: float = 1.0
    birth_resource_support: float = 1.0
    parent_transport_capacity: float = 1.0
    collision_failure_streak: int = 0
    extension_events: int = 0
    last_growth_step: int = -1
    initial_emergence_shoulder_length: float = 0.0
    initial_shoulder_point_count: int = 1
    escape_extensions_remaining: int = 0
    post_emergence_extension_attempts: int = 0
    post_emergence_extensions_accepted: int = 0
    parent_collision_blocked_extensions: int = 0
    other_root_collision_blocked_extensions: int = 0
    surface_blocked_extensions: int = 0
    other_blocked_extensions: int = 0
    accepted_extension_length_sum: float = 0.0
    local_resource_sufficiency_sum: float = 0.0
    local_resource_sufficiency_count: int = 0
    last_local_resource_sufficiency: float = 0.0
    post_emergence_direction_z_sum: float = 0.0
    post_emergence_direction_count: int = 0
    resource_focus: str = "balanced"
    resource_focus_birth_step: int = 0
    resource_focus_last_update_step: int = 0
    resource_focus_updates: int = 0
    resource_focus_score_at_assignment: float = 0.0
    resource_focus_seed: int = 0
    consecutive_upward_extensions: int = 0
    maximum_consecutive_upward_extensions: int = 0
    last_direction_score_components: dict[str, float] = field(default_factory=dict)

    def arc_lengths(self) -> np.ndarray:
        count = len(self.material_arcs)
        if self._material_arc_cache_count != count:
            self._material_arc_cache = np.asarray(
                self.material_arcs, dtype=np.float64
            )
            self._material_arc_cache_count = count
        return self._material_arc_cache

    def total_length(self) -> float:
        return float(self.material_arcs[-1]) if self.material_arcs else 0.0

    def add_structural_area_event(self, end_arc: float, increment: float) -> None:
        if increment <= 0.0:
            return
        self.structural_area_events.append((max(0.0, float(end_arc)), float(increment)))
        self.structural_area_version += 1
        self.structural_area_total += float(increment)

    def area_event_cache(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted event ends and reverse cumulative increments."""

        if self._area_cache_version != self.structural_area_version:
            if self.structural_area_events:
                events = np.asarray(self.structural_area_events, dtype=np.float64)
                order = np.argsort(events[:, 0], kind="mergesort")
                ends = events[order, 0]
                increments = events[order, 1]
                suffix = np.cumsum(increments[::-1])[::-1]
            else:
                ends = np.empty(0, dtype=np.float64)
                suffix = np.empty(0, dtype=np.float64)
            self._area_cache_ends = ends
            self._area_cache_suffix = suffix
            self._area_cache_version = self.structural_area_version
        return self._area_cache_ends, self._area_cache_suffix


class RootAxisStore:
    """Continuous-axis root architecture that exports sampled graph support."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.axes: list[RootAxis] = [
            RootAxis(
                axis_id=0,
                parent_axis_id=-1,
                parent_arc_length=0.0,
                parent_local_azimuth=math.nan,
                birth_step=0,
                is_anchor_axis=True,
                branch_generation=0,
                points=[np.array([0.0, 0.0, 0.0], dtype=np.float64)],
                tangents=[np.array([0.0, 0.0, -1.0], dtype=np.float64)],
                point_birth_steps=[0],
                structural_area_total=0.0,
                resource_focus="balanced",
                resource_focus_birth_step=0,
                resource_focus_last_update_step=0,
            )
        ]
        initial_spatial_capacity = max(
            256,
            min(int(config.initial_capacity), self.sample_cap if hasattr(self, "sample_cap") else int(config.max_nodes)),
        )
        self.spatial_points = np.empty(
            (initial_spatial_capacity, 3), dtype=np.float64
        )
        self.spatial_axis_ids = np.empty(
            initial_spatial_capacity, dtype=np.int32
        )
        self.spatial_points[0] = self.axes[0].points[0]
        self.spatial_axis_ids[0] = 0
        self.spatial_size = 1
        self.tree: cKDTree | None = None
        self.tree_size = 0
        self.dynamic_since_rebuild = 0
        self.sample_count = 1
        self.tree_rebuild_count = 0
        self.curvature_values: list[float] = []
        self.tip_bend_angles_deg: list[float] = []
        self.branch_emergence_angles_deg: list[float] = []
        self.branch_azimuth_angles: list[float] = []
        self.branch_origin_spacings: list[float] = []
        self.collision_sample_checks = 0
        self.growth_attempts = 0
        self.starvation_signal_sum = 0.0
        self.starvation_signal_count = 0
        self.branch_origin_starvation_sum = 0.0
        self.branch_origin_starvation_count = 0
        self.resource_support_gate_sum = 0.0
        self.resource_support_gate_count = 0
        self.extension_starvation_sum = 0.0
        self.extension_starvation_count = 0
        self.branch_origin_candidate_evaluations = 0
        self.sample_cap = effective_sampled_point_cap(config)
        self.branch_sites: list[BranchSite] = []
        self.active_tips_at_step_start_total = 0
        self.active_tip_count_observations = 0
        self.active_tip_count_max = 1
        self.tip_extension_attempts = 0
        self.tip_extensions_accepted = 0
        self.tip_extensions_collision_blocked = 0
        self.tip_extensions_surface_blocked = 0
        self.tip_extensions_sample_cap_blocked = 0
        self.tip_extensions_other_blocked = 0
        self.primary_tip_extension_attempts = 0
        self.primary_tip_extensions_accepted = 0
        self.lateral_tip_extension_attempts = 0
        self.lateral_tip_extensions_accepted = 0
        self.generation_1_extension_attempts = 0
        self.generation_1_extensions_accepted = 0
        self.generation_2_extension_attempts = 0
        self.generation_2_extensions_accepted = 0
        self.generation_3plus_extension_attempts = 0
        self.generation_3plus_extensions_accepted = 0
        self.branch_sites_closed_single_trial = 0
        self.branch_sites_reopened_after_thickening = 0
        self.accepted_origin_surface_clearances: list[float] = []
        self.parent_radii_at_branch_origins: list[float] = []
        self.maximum_sample_points_in_any_step = 0
        self.accepted_extension_directions: list[np.ndarray] = []
        self.accepted_extension_generations: list[int] = []
        self.accepted_extension_foci: list[str] = []
        self.accepted_extensions_by_focus = {name: 0 for name in RESOURCE_FOCI}
        self.direction_score_component_sum: dict[str, float] = {}
        self.direction_score_component_max: dict[str, float] = {}
        self.direction_score_evaluations = 0
        self.resource_focus_assignments = {name: 0 for name in RESOURCE_FOCI}
        self.resource_focus_assignments["balanced"] = 1
        self.time_series_snapshots: list[dict[str, float | int]] = []
        self.profile_retry_site_traversal_sec = 0.0
        self.profile_branch_probability_trials_sec = 0.0
        self.profile_physical_origin_search_sec = 0.0
        self.profile_active_tip_extensions_sec = 0.0
        self.profile_collision_queries_sec = 0.0
        self.profile_resource_direction_candidates_sec = 0.0

    def estimated_sample_count(self) -> int:
        """Return exported support-point count in O(1)."""

        return int(self.sample_count)

    def active_axis_ids(self) -> list[int]:
        return [axis.axis_id for axis in self.axes if axis.active]

    def complete_insertion_fits(self, sample_count: int) -> bool:
        return self.sample_count + max(0, int(sample_count)) <= self.sample_cap

    def _maybe_rebuild_tree(self) -> None:
        if self.tree is None or self.dynamic_since_rebuild >= 128:
            pts = self.spatial_points[:self.spatial_size]
            self.tree = cKDTree(pts, compact_nodes=True) if pts.shape[0] else None
            self.tree_size = self.spatial_size
            self.dynamic_since_rebuild = 0
            self.tree_rebuild_count += 1

    def _ensure_spatial_capacity(self, required: int) -> None:
        if required <= self.spatial_points.shape[0]:
            return
        capacity = self.spatial_points.shape[0]
        while capacity < required:
            capacity = min(
                max(capacity * 2, required),
                max(self.sample_cap, required),
            )
        points = np.empty((capacity, 3), dtype=np.float64)
        axis_ids = np.empty(capacity, dtype=np.int32)
        points[:self.spatial_size] = self.spatial_points[:self.spatial_size]
        axis_ids[:self.spatial_size] = self.spatial_axis_ids[:self.spatial_size]
        self.spatial_points = points
        self.spatial_axis_ids = axis_ids

    def _append_spatial_points(self, axis_id: int, points: np.ndarray) -> None:
        count = int(points.shape[0])
        if count <= 0:
            return
        start = self.spatial_size
        stop = start + count
        self._ensure_spatial_capacity(stop)
        self.spatial_points[start:stop] = np.asarray(
            points, dtype=np.float64
        )
        self.spatial_axis_ids[start:stop] = int(axis_id)
        self.spatial_size = stop
        self.dynamic_since_rebuild += count

    def samples_are_clear(
        self,
        samples: np.ndarray,
        *,
        own_axis_id: int,
        parent_axis_id: int = -1,
        branch_origin: np.ndarray | None = None,
    ) -> bool:
        return self.samples_clearance_rejection(
            samples,
            own_axis_id=own_axis_id,
            parent_axis_id=parent_axis_id,
            branch_origin=branch_origin,
        ) is None

    def samples_clearance_rejection(
        self,
        samples: np.ndarray,
        *,
        own_axis_id: int,
        parent_axis_id: int = -1,
        branch_origin: np.ndarray | None = None,
        allow_shared_emergence_shoulders: bool = True,
    ) -> str | None:
        """Return an explicit physical rejection reason, or ``None``.

        Extension callers use the boolean wrapper above. Primordium callers
        retain this reason so every passed probability trial is auditable.
        """

        self.growth_attempts += 1
        self.collision_sample_checks += int(samples.shape[0])
        if (
            self.config.reject_above_surface_curves
            and samples.shape[0]
            and float(np.max(samples[:, 2]))
            > self.config.soil_surface_z + self.config.max_above_surface_tolerance
        ):
            return "above_soil_surface"
        self._maybe_rebuild_tree()
        if self.tree is None:
            return None
        # The anchor is exempt from root-root collision rejection.
        if own_axis_id == 0:
            return None
        radius = 2.4 * self.config.base_radius + self.config.spatial_clearance
        origin = branch_origin if branch_origin is not None else samples[0]
        spatial_axis_ids = self.spatial_axis_ids
        spatial_points = self.spatial_points

        def is_shared_emergence_shoulder(
            other_axis_id: int,
            other_point: np.ndarray,
            candidate_point: np.ndarray,
        ) -> bool:
            """Allow physically separated sibling collars to leave one surface site.

            Cylindrical collar clearance is evaluated before this method. During
            creation only, centerlines from same/nearby parent-surface origins
            necessarily share a small 3D neighborhood before their different
            azimuths diverge. Later tip extensions disable this sibling-collar
            exemption even when a newborn parent-corridor origin is supplied.
            """

            if (
                not allow_shared_emergence_shoulders
                or branch_origin is None
                or parent_axis_id < 0
                or other_axis_id < 0
                or other_axis_id >= len(self.axes)
            ):
                return False
            other_axis = self.axes[other_axis_id]
            if other_axis.parent_axis_id != parent_axis_id:
                return False
            emergence_zone = 3.0 * radius
            return bool(
                np.linalg.norm(other_axis.points[0] - origin) < emergence_zone
                and np.linalg.norm(other_point - origin) < emergence_zone
                and np.linalg.norm(candidate_point - origin) < emergence_zone
            )

        broad_phase_neighbors = self.tree.query_ball_point(
            samples,
            radius,
            return_sorted=False,
        )
        for point, raw_indices in zip(samples, broad_phase_neighbors):
            for raw_index in raw_indices:
                other_index = int(raw_index)
                other_axis = int(spatial_axis_ids[other_index])
                if other_axis == own_axis_id:
                    continue
                # Laterals cannot permanently block the anchor's downward search.
                if own_axis_id == 0:
                    continue
                if other_axis == parent_axis_id:
                    if float(np.linalg.norm(spatial_points[other_index] - origin)) < 3.0 * radius:
                        continue
                if (
                    allow_shared_emergence_shoulders
                    and is_shared_emergence_shoulder(
                        other_axis, spatial_points[other_index], point
                    )
                ):
                    continue
                return (
                    "parent_collision"
                    if other_axis == parent_axis_id
                    else "other_root_collision"
                )
        # Dynamic points added after the last rebuild are checked exactly.
        if self.tree_size < self.spatial_size:
            for point in samples:
                for other_index in range(self.tree_size, self.spatial_size):
                    other_axis = int(spatial_axis_ids[other_index])
                    if other_axis == own_axis_id:
                        continue
                    if own_axis_id == 0:
                        continue
                    if other_axis == parent_axis_id and float(
                        np.linalg.norm(spatial_points[other_index] - origin)
                    ) < 3.0 * radius:
                        continue
                    if (
                        allow_shared_emergence_shoulders
                        and is_shared_emergence_shoulder(
                            other_axis, spatial_points[other_index], point
                        )
                    ):
                        continue
                    if float(np.linalg.norm(spatial_points[other_index] - point)) < radius:
                        return (
                            "parent_collision"
                            if other_axis == parent_axis_id
                            else "other_root_collision"
                        )
        return None

    def append_curve_extension(
        self,
        axis: RootAxis,
        samples: np.ndarray,
        tangents: np.ndarray,
        step: int,
    ) -> None:
        previous_tangent = axis.tangents[-1]
        previous_point = axis.points[-1]
        material_arc = float(axis.material_arcs[-1])
        added = 0
        for sample, tangent in zip(samples, tangents):
            segment = np.asarray(sample, dtype=np.float64) - previous_point
            segment_length = float(np.linalg.norm(segment))
            if segment_length <= 1e-12:
                continue
            unit_segment = segment / segment_length
            bend = vector_angle_degrees(previous_tangent, unit_segment)
            self.tip_bend_angles_deg.append(bend)
            self.curvature_values.append(math.radians(bend) / max(segment_length, 1e-12))
            axis.points.append(np.asarray(sample, dtype=np.float64).copy())
            axis.tangents.append(normalized_direction(tangent))
            axis.point_birth_steps.append(step)
            material_arc += segment_length
            axis.material_arcs.append(material_arc)
            added += 1
            previous_tangent = axis.tangents[-1]
            previous_point = axis.points[-1]
        if samples.shape[0]:
            self._append_spatial_points(axis.axis_id, samples)
        self.sample_count += added

    def create_branch_axis(
        self,
        parent_axis: RootAxis,
        origin_arc: float,
        azimuth: float,
        emergence_angle_deg: float,
        step: int,
        thickness_increment: float,
        rng: np.random.Generator,
        resource_field: HeterogeneousResourceField,
        raining: bool,
        demand_state: ResourceDemandState | None = None,
        starvation_signal: float = 0.0,
        site: BranchSite | None = None,
        resource_focus: str = "balanced",
        resource_focus_score: float = 0.0,
        resource_focus_seed: int = 0,
    ) -> tuple[RootAxis | None, str | None]:
        origin, parent_tangent = interpolate_axis(parent_axis, origin_arc)
        u, v = local_orthonormal_frame(parent_tangent)
        radial = math.cos(azimuth) * u + math.sin(azimuth) * v
        origin_radius = float(np.linalg.norm(origin[:2]))
        starvation_signal = float(np.clip(starvation_signal, 0.0, 1.0))
        starvation_response = float(severe_starvation_response(starvation_signal))
        if starvation_response > 0.0 and origin_radius > 0.10:
            # Project scarcity confinement into the parent's normal plane.
            inward = np.array([-origin[0], -origin[1], 0.0], dtype=np.float64)
            inward -= float(np.dot(inward, parent_tangent)) * parent_tangent
            inward_norm = float(np.linalg.norm(inward))
            if inward_norm > 1e-12:
                inward /= inward_norm
                inward_mix = 0.72 * starvation_response
                radial = normalized_direction(
                    (1.0 - inward_mix) * radial + inward_mix * inward
                )
        angle = math.radians(emergence_angle_deg)
        emergence_tangent = normalized_direction(
            math.cos(angle) * parent_tangent + math.sin(angle) * radial
        )
        if starvation_response > 0.0:
            downward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            downward_pull = 0.08 * starvation_response
            emergence_tangent = normalized_direction(
                (1.0 - downward_pull) * emergence_tangent
                + downward_pull * downward
            )
        branch_id = len(self.axes)
        generation = parent_axis.branch_generation + 1
        generation_index = max(generation - 1, 0)
        resource_support = 1.0 - starvation_signal
        parent_radius = sampled_axis_radius(parent_axis, origin_arc, self.config)
        parent_transport = float(np.clip(
            parent_radius / max(self.config.base_radius, 1e-12), 0.40, 4.0
        ))
        variation_phase = (
            12.9898 * (branch_id + 1)
            + 0.1733 * float(step)
            + 1.6180 * float(origin_arc)
            + 0.7110 * float(azimuth)
        )
        axis_variation = float(np.clip(
            math.exp(0.16 * math.sin(variation_phase)), 0.72, 1.32
        ))
        radius_scale = float(np.clip(
            self.config.lateral_transport_base_fraction
            * math.sqrt(parent_transport)
            * self.config.lateral_generation_radius_decay ** generation_index
            * (0.72 + 0.28 * resource_support)
            * axis_variation,
            0.045,
            0.55,
        ))
        branch = RootAxis(
            axis_id=branch_id,
            parent_axis_id=parent_axis.axis_id,
            parent_arc_length=float(origin_arc),
            parent_local_azimuth=float(azimuth),
            birth_step=step,
            is_anchor_axis=False,
            branch_generation=generation,
            points=[origin.copy()],
            tangents=[emergence_tangent],
            point_birth_steps=[step],
            material_arcs=[0.0],
            origin_site_id=site.site_id if site is not None else -1,
            structural_area_total=0.0,
            radius_scale=float(radius_scale),
            birth_resource_support=float(resource_support),
            parent_transport_capacity=parent_transport,
            resource_focus=resource_focus,
            resource_focus_birth_step=step,
            resource_focus_last_update_step=step,
            resource_focus_score_at_assignment=float(resource_focus_score),
            resource_focus_seed=int(resource_focus_seed),
            escape_extensions_remaining=int(
                self.config.lateral_escape_accepted_extensions
            ),
        )
        if starvation_response > 0.0:
            # A short shoulder prevents starved branches from overlapping the anchor.
            downward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            end_direction = normalized_direction(
                (1.0 - 0.24 * starvation_response) * emergence_tangent
                + 0.24 * starvation_response * downward
            )
        else:
            direction_started = time.perf_counter()
            end_direction = biased_axis_direction(
                branch,
                rng,
                resource_field,
                raining,
                self.config,
                demand_state,
            )
            self.profile_resource_direction_candidates_sec += (
                time.perf_counter() - direction_started
            )
        # Emergence length depends on post-initiation state, never B.P.
        initial_scale = self.config.lateral_branch_initial_scale * (
            0.70 + 0.30 * (1.0 - starvation_response)
        )
        length = max(
            self.config.segment_length * 0.28,
            self.config.segment_length
            * initial_scale
            * self.config.lateral_generation_length_decay ** generation_index
            * (0.72 + 0.28 * resource_support)
            * (0.78 + 0.22 * min(parent_transport, 2.0) / 2.0)
            * axis_variation,
        )
        samples, tangents = hermite_curve_samples(
            origin,
            emergence_tangent,
            end_direction,
            length,
            self.config.curve_samples_per_extension,
        )
        if not self.complete_insertion_fits(samples.shape[0]):
            return None, "sample_cap"
        collision_started = time.perf_counter()
        rejection_reason = self.samples_clearance_rejection(
            samples,
            own_axis_id=branch_id,
            parent_axis_id=parent_axis.axis_id,
            branch_origin=origin,
        )
        self.profile_collision_queries_sec += time.perf_counter() - collision_started
        if rejection_reason is not None:
            return None, rejection_reason
        branch.extension_events += 1
        branch.last_growth_step = step
        branch.initial_emergence_shoulder_length = float(length)
        self.axes.append(branch)
        self.resource_focus_assignments[resource_focus] = (
            self.resource_focus_assignments.get(resource_focus, 0) + 1
        )
        self.append_curve_extension(branch, samples, tangents, step)
        branch.initial_shoulder_point_count = len(branch.points)
        accepted_direction = normalized_direction(end_direction)
        self.accepted_extension_directions.append(accepted_direction.copy())
        self.accepted_extension_generations.append(branch.branch_generation)
        self.accepted_extension_foci.append(branch.resource_focus)
        self.accepted_extensions_by_focus[branch.resource_focus] = (
            self.accepted_extensions_by_focus.get(branch.resource_focus, 0) + 1
        )
        if accepted_direction[2] > self.config.upward_component_threshold:
            branch.consecutive_upward_extensions = 1
            branch.maximum_consecutive_upward_extensions = 1
        record_transport_path_growth(
            self,
            branch,
            thickness_increment=thickness_increment,
            grown_length=length,
        )
        if demand_state is not None:
            demand_state.accumulate_points(samples, resource_field, raining, self.config)
        parent_axis.branch_origins.append(float(origin_arc))
        parent_axis.branch_azimuths.append(float(azimuth))
        parent_axis.branch_origin_steps.append(int(step))
        branch_base_radius = sampled_axis_radius(branch, 0.0, self.config)
        parent_axis.branch_origin_base_radii.append(branch_base_radius)
        if site is not None:
            site.accepted_branch_count += 1
            site.occupied_azimuths.append(float(azimuth))
            site.accepted_branch_base_radii.append(branch_base_radius)
            site.last_evaluated_parent_radius = sampled_axis_radius(
                parent_axis, origin_arc, self.config
            )
        self.branch_emergence_angles_deg.append(float(emergence_angle_deg))
        self.branch_azimuth_angles.append(float(azimuth))
        spacings = [
            abs(float(origin_arc) - previous)
            for previous in parent_axis.branch_origins[:-1]
        ]
        self.branch_origin_spacings.append(min(spacings) if spacings else math.nan)
        return branch, None


def interpolate_axis(axis: RootAxis, arc_length: float) -> tuple[np.ndarray, np.ndarray]:
    arcs = axis.arc_lengths()
    return interpolate_axis_with_arcs(axis, arcs, arc_length)


def interpolate_axis_with_arcs(
    axis: RootAxis,
    arcs: np.ndarray,
    arc_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate an axis while reusing a caller-computed arc-length array."""

    if arcs.size <= 1:
        return axis.points[0].copy(), axis.tangents[0].copy()
    target = float(np.clip(arc_length, 0.0, arcs[-1]))
    index = int(np.searchsorted(arcs, target, side="right"))
    index = int(np.clip(index, 1, arcs.size - 1))
    low = float(arcs[index - 1])
    high = float(arcs[index])
    fraction = 0.0 if high <= low else (target - low) / (high - low)
    point = (1.0 - fraction) * axis.points[index - 1] + fraction * axis.points[index]
    tangent = normalized_direction(
        (1.0 - fraction) * axis.tangents[index - 1] + fraction * axis.tangents[index]
    )
    return point, tangent


def axis_radii_at_arcs(
    axis: RootAxis,
    arcs: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Evaluate scientific pipe-model radii from material range additions.

    Each area event applies from material arc zero through its event end. Event
    ends are sorted once per axis version and reverse-cumulated; vector radius
    queries then use binary searches without scanning prior growth history.
    """

    query = np.asarray(arcs, dtype=np.float64)
    base_scale = 1.0 if axis.is_anchor_axis else max(0.08, float(axis.radius_scale))
    tip_radius = (
        config.base_radius
        * config.structural_tip_baseline_fraction
        * base_scale
    )
    area = np.full(query.shape, math.pi * tip_radius * tip_radius, dtype=np.float64)
    ends, suffix = axis.area_event_cache()
    if ends.size:
        indices = np.searchsorted(ends, query, side="left")
        valid = indices < suffix.size
        area[valid] += suffix[indices[valid]]

    return np.sqrt(np.maximum(area, 1e-18) / math.pi)


def sampled_axis_radius(axis: RootAxis, arc: float, config: SimulationConfig) -> float:
    return float(axis_radii_at_arcs(
        axis, np.asarray([float(arc)], dtype=np.float64), config
    )[0])


def record_transport_path_growth(
    axis_store: RootAxisStore,
    axis: RootAxis,
    *,
    thickness_increment: float,
    grown_length: float,
) -> None:
    """Add self and recursively attenuated ancestor cross-sectional area.

    Self: dA = T.I. * grown_length * structural_self_area_coefficient * F,
    where F=1 for the primary and structural_lateral_self_area_fraction for
    lateral axes. The lateral fraction keeps child collars bounded by their
    parent transport path.
    Ancestor depth d (direct parent d=1):
      dA = T.I. * grown_length * structural_ancestor_area_coefficient
            * structural_ancestor_transport_decay ** (d - 1).
    Ancestor events end at the descendant attachment, so distal parent tissue
    receives no contribution. Radius is later computed as sqrt(area / pi).
    """

    config = axis_store.config
    ti = max(0.0, float(thickness_increment))
    length = max(0.0, float(grown_length))
    self_fraction = (
        1.0 if axis.is_anchor_axis
        else config.structural_lateral_self_area_fraction
    )
    axis.add_structural_area_event(
        axis.total_length(),
        ti * length * config.structural_self_area_coefficient * self_fraction,
    )
    descendant = axis
    depth = 1
    while descendant.parent_axis_id >= 0:
        parent = axis_store.axes[descendant.parent_axis_id]
        increment = (
            ti
            * length
            * config.structural_ancestor_area_coefficient
            * config.structural_ancestor_transport_decay ** (depth - 1)
        )
        parent.add_structural_area_event(descendant.parent_arc_length, increment)
        descendant = parent
        depth += 1

    # Apply child-to-root area corrections to enforce the pipe/collar bound.
    descendant = axis
    ratio_limit = config.branch_origin_child_parent_radius_ratio_limit
    while descendant.parent_axis_id >= 0:
        parent = axis_store.axes[descendant.parent_axis_id]
        child_area = math.pi * sampled_axis_radius(
            descendant, 0.0, config
        ) ** 2
        parent_radius = sampled_axis_radius(
            parent, descendant.parent_arc_length, config
        )
        parent_area = math.pi * parent_radius * parent_radius
        required_parent_area = child_area / (ratio_limit * ratio_limit)
        if required_parent_area > parent_area:
            parent.add_structural_area_event(
                descendant.parent_arc_length,
                required_parent_area - parent_area,
            )
        descendant = parent


def axis_curve_resource_signal(
    point: np.ndarray,
    raining: bool,
    field_model: HeterogeneousResourceField,
    config: SimulationConfig,
) -> float:
    return float(field_model.combined_signal(np.asarray(point, dtype=np.float64), raining, config)[0])


def curvature_limited_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_bend_degrees: float,
) -> np.ndarray:
    """Spherically rotate current toward target without exceeding max bend."""

    start = normalized_direction(current)
    finish = normalized_direction(target)
    cosine = float(np.clip(np.dot(start, finish), -1.0, 1.0))
    angle = math.acos(cosine)
    limit = math.radians(max_bend_degrees)
    if angle <= limit or angle <= 1e-12:
        return finish
    fraction = limit / angle
    sine = math.sin(angle)
    if abs(sine) <= 1e-12:
        return normalized_direction((1.0 - fraction) * start + fraction * finish)
    return normalized_direction(
        math.sin((1.0 - fraction) * angle) / sine * start
        + math.sin(fraction * angle) / sine * finish
    )


def draw_resource_focus(
    rng: np.random.Generator,
    local_values: np.ndarray,
    demand_state: ResourceDemandState,
    config: SimulationConfig,
) -> tuple[str, float]:
    """Draw one reproducible focus while excluding resources with zero supply."""

    probabilities = demand_state.focus_probabilities(local_values, config)
    selected = int(rng.choice(len(RESOURCE_FOCI), p=probabilities))
    return RESOURCE_FOCI[selected], float(probabilities[selected])


def maybe_update_resource_focus(
    axis: RootAxis,
    step: int,
    rng: np.random.Generator,
    local_values: np.ndarray,
    demand_state: ResourceDemandState,
    config: SimulationConfig,
) -> bool:
    """Persist a focus, then update it stochastically or when supply disappears."""

    age_since_update = int(step) - int(axis.resource_focus_last_update_step)
    unavailable = (
        axis.resource_focus in RESOURCE_NAMES
        and demand_state.supply[RESOURCE_NAMES.index(axis.resource_focus)]
        <= CAPTURE_REPORTING_EPSILON
    )
    if (
        age_since_update < config.resource_focus_persistence_steps
        and not unavailable
    ):
        return False
    if not unavailable and float(rng.random()) >= config.resource_focus_update_probability:
        return False
    focus, score = draw_resource_focus(rng, local_values, demand_state, config)
    changed = focus != axis.resource_focus
    axis.resource_focus = focus
    axis.resource_focus_last_update_step = int(step)
    axis.resource_focus_score_at_assignment = float(score)
    if changed:
        axis.resource_focus_updates += 1
    return changed


@dataclass(frozen=True)
class DirectionResourceContext:
    """Reusable exact resource terms for repeated candidates at one fixed tip."""

    point: np.ndarray
    previous: np.ndarray
    max_bend: float
    cone_base: np.ndarray
    cone_u: np.ndarray
    cone_v: np.ndarray
    deterministic_candidates: tuple[np.ndarray, ...]
    current_values: np.ndarray
    supply: np.ndarray
    active: np.ndarray
    sufficiency: float
    focus_weights: np.ndarray


def prepare_direction_resource_context(
    axis: RootAxis,
    resource_field: HeterogeneousResourceField,
    raining: bool,
    config: SimulationConfig,
    demand_state: ResourceDemandState | None,
    current_resource_values: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
) -> DirectionResourceContext:
    """Evaluate tip-invariant direction terms once per extension attempt.

    The values are reused for every candidate at the same unchanged tip without
    changing candidate order, random draws, float64 equations, or scientific
    events.
    """

    point = np.asarray(axis.points[-1], dtype=np.float64)
    previous = normalized_direction(axis.tangents[-1])
    max_bend = (
        config.anchor_curve_max_bend_degrees
        if axis.is_anchor_axis else config.lateral_curve_max_bend_degrees
    )
    probe = max(config.segment_length, 0.25)
    probe_values = np.vstack(resource_field.values(
        point[None, :] + DIRECTION_PROBE_DIRECTIONS * probe,
        raining,
        config,
    )).T
    supply = (
        demand_state.supply if demand_state is not None
        else resource_field.environment.supply_gates(config)
    )
    cone_base = normalized_direction(previous)
    cone_u, cone_v = local_orthonormal_frame(cone_base)
    deterministic_candidates: list[np.ndarray] = [
        curvature_limited_toward(
            previous,
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            max_bend,
        )
    ]
    for target in DIRECTION_FIXED_TARGETS:
        deterministic_candidates.append(
            curvature_limited_toward(previous, target, max_bend)
        )
    for resource_index in range(4):
        if supply[resource_index] <= CAPTURE_REPORTING_EPSILON:
            continue
        best_probe = DIRECTION_PROBE_DIRECTIONS[
            int(np.argmax(probe_values[:, resource_index]))
        ]
        deterministic_candidates.append(
            curvature_limited_toward(previous, best_probe, max_bend)
        )
    if current_resource_values is None:
        current_resource_values = resource_field.values(
            point, raining, config
        )
    current_values = np.asarray(
        [value[0] for value in current_resource_values],
        dtype=np.float64,
    )
    demand = (
        demand_state.weights(config) if demand_state is not None
        else np.ones(4, dtype=np.float64)
    )
    active = (supply > CAPTURE_REPORTING_EPSILON).astype(np.float64)
    active_count = max(float(np.sum(active)), 1.0)
    environmental_sufficiency = float(np.sum(supply) / active_count)
    local_sufficiency = float(
        np.sum(current_values * active) / active_count
    )
    sufficiency = float(np.clip(
        0.45 * environmental_sufficiency + 0.55 * local_sufficiency,
        0.0,
        1.0,
    ))
    focus_index = (
        RESOURCE_NAMES.index(axis.resource_focus)
        if axis.resource_focus in RESOURCE_NAMES else -1
    )
    if (
        focus_index >= 0
        and supply[focus_index] > CAPTURE_REPORTING_EPSILON
    ):
        focus_weights = np.zeros(4, dtype=np.float64)
        focus_weights[focus_index] = max(demand[focus_index], 0.1)
    else:
        focus_weights = active * np.maximum(demand, 0.05)
    focus_weights /= max(float(np.sum(focus_weights)), 1e-12)
    return DirectionResourceContext(
        point=point,
        previous=previous,
        max_bend=float(max_bend),
        cone_base=cone_base,
        cone_u=cone_u,
        cone_v=cone_v,
        deterministic_candidates=tuple(deterministic_candidates),
        current_values=current_values,
        supply=supply,
        active=active,
        sufficiency=sufficiency,
        focus_weights=focus_weights,
    )


def biased_axis_direction(
    axis: RootAxis,
    rng: np.random.Generator,
    resource_field: HeterogeneousResourceField,
    raining: bool,
    config: SimulationConfig,
    demand_state: ResourceDemandState | None = None,
    escape_outward_direction: np.ndarray | None = None,
    resource_context: DirectionResourceContext | None = None,
) -> np.ndarray:
    """Choose from a diverse curvature-limited, focus-specific candidate set."""

    if resource_context is None:
        resource_context = prepare_direction_resource_context(
            axis,
            resource_field,
            raining,
            config,
            demand_state,
        )
    point = resource_context.point
    previous = resource_context.previous
    max_bend = resource_context.max_bend
    count = max(6, int(config.tip_elongation_candidates))
    candidates = [*cone_directions_from_frame(
        resource_context.cone_base,
        resource_context.cone_u,
        resource_context.cone_v,
        rng,
        count,
        max_bend,
    )]
    # Candidate references are fixed for each tip evaluation.
    candidates.extend(resource_context.deterministic_candidates)
    candidates_array = np.asarray(candidates, dtype=np.float64)
    candidate_points = point[None, :] + candidates_array * config.segment_length
    values = np.vstack(resource_field.values(candidate_points, raining, config)).T
    current_values = resource_context.current_values
    sufficiency = resource_context.sufficiency
    focus_weights = resource_context.focus_weights
    positive_gain = np.maximum(values - current_values[None, :], 0.0)
    absolute_value = values @ focus_weights
    directional_gain = positive_gain @ focus_weights

    downward = np.clip(-candidates_array[:, 2], 0.0, 1.0)
    upward = np.clip(candidates_array[:, 2], 0.0, 1.0)
    horizontal = np.linalg.norm(candidates_array[:, :2], axis=1)
    persistence_component = 1.35 * (candidates_array @ previous)
    if axis.is_anchor_axis:
        gravity_component = (2.25 + 0.55 * (1.0 - sufficiency)) * downward
        plagiotropic_component = 0.10 * sufficiency * horizontal
    else:
        gravity_component = (0.20 + 3.20 * (1.0 - sufficiency) ** 1.4) * downward
        plagiotropic_component = 3.40 * sufficiency * horizontal
    absolute_component = 1.20 * absolute_value
    gradient_component = 5.00 * directional_gain
    surface_excess = np.clip(
        candidate_points[:, 2]
        - (config.soil_surface_z + config.max_above_surface_tolerance),
        0.0, None,
    )
    surface_component = -config.above_surface_penalty * surface_excess
    upward_component = -config.upward_growth_penalty * upward * (1.0 - 0.55 * sufficiency)
    meaningful_gain = directional_gain >= config.upward_resource_gain_threshold
    upward_forbidden = (
        (candidates_array[:, 2] > config.upward_component_threshold)
        & (
            (axis.consecutive_upward_extensions >= config.maximum_consecutive_upward_extensions)
            | (~meaningful_gain)
            | (candidate_points[:, 2] > config.soil_surface_z)
            | (candidates_array[:, 2] > 0.30)
        )
    )
    upward_component[upward_forbidden] -= 100.0
    scores = (
        persistence_component + gravity_component + plagiotropic_component
        + absolute_component + gradient_component + surface_component
        + upward_component
    )
    escape_component = np.zeros(scores.shape, dtype=np.float64)
    if escape_outward_direction is not None and not axis.is_anchor_axis:
        outward = normalized_direction(escape_outward_direction)
        outward_projection = candidates_array @ outward
        escape_component = (
            config.lateral_escape_direction_weight * outward_projection
        )
        escape_forbidden = (
            outward_projection < config.lateral_escape_min_outward_component
        )
        escape_component[escape_forbidden] -= 100.0
        scores += escape_component
    uniform = np.clip(rng.random(scores.size), 1e-12, 1.0 - 1e-12)
    gumbel = -np.log(-np.log(uniform))
    chosen = int(np.argmax(scores + config.tip_choice_temperature * gumbel))
    axis.last_direction_score_components = {
        "persistence": float(persistence_component[chosen]),
        "gravity": float(gravity_component[chosen]),
        "plagiotropism": float(plagiotropic_component[chosen]),
        "absolute_resource": float(absolute_component[chosen]),
        "positive_resource_gain": float(gradient_component[chosen]),
        "surface": float(surface_component[chosen]),
        "upward": float(upward_component[chosen]),
        "escape_outward": float(escape_component[chosen]),
        "total": float(scores[chosen]),
        "sufficiency": sufficiency,
    }
    axis.resource_focus_score_at_assignment = float(absolute_value[chosen])
    return normalized_direction(candidates_array[chosen])


def sample_branch_emergence_angle(
    rng: np.random.Generator,
    starvation: float = 0.0,
    phosphorus: float = 0.0,
    nitrogen: float = 0.0,
    potassium: float = 0.0,
    resource_focus: str = "balanced",
    sufficiency: float | None = None,
) -> float:
    """Sample a dynamic parent-relative angle from sufficiency and tip focus."""

    starvation = float(np.clip(starvation, 0.0, 1.0))
    support = float(np.clip(1.0 - starvation if sufficiency is None else sufficiency, 0.0, 1.0))
    # Scarcity favors 10--35° shoulders; rich support permits plagiotropism.
    mean = 20.0 + 70.0 * support
    focus_shift = {
        "phosphorus": 9.0,
        "potassium": 5.0,
        "nitrogen": -18.0,
        "water": -10.0,
        "balanced": 0.0,
    }.get(resource_focus, 0.0)
    # Local concentrations modulate, but do not deterministically select, focus.
    shallow_bias = 4.0 * max(phosphorus, potassium)
    deep_bias = 3.0 * nitrogen
    mean += focus_shift + support * (shallow_bias - deep_bias)
    spread = 5.0 + 10.0 * support
    upper = 35.0 + 77.0 * support
    return float(np.clip(rng.normal(mean, spread), 10.0, upper))


def candidate_branch_collar_radius(
    parent_radius: float,
    generation: int,
    config: SimulationConfig,
) -> float:
    """Return a conservative physical collar radius for a proposed lateral."""

    generation_scale = config.lateral_generation_radius_decay ** max(generation - 1, 0)
    biological = config.base_radius * 0.18 * generation_scale
    return float(max(
        config.base_radius * 0.025,
        min(0.42 * max(parent_radius, 1e-12), biological),
    ))


def cylindrical_surface_clearance(
    axis: RootAxis,
    origin_arc: float,
    azimuth: float,
    candidate_collar_radius: float,
    config: SimulationConfig,
) -> tuple[bool, float]:
    """Test branch collars on the current cylindrical parent surface.

    For each accepted origin, distance is
      sqrt(delta_arc**2 + (local_parent_radius * delta_theta)**2).
    Clearance is that distance minus the collar sum times the configured factor
    and a small numerical margin. Same-arc origins are therefore allowed when
    their circumferential separation physically fits.
    """

    if not axis.branch_origins:
        return True, math.inf
    candidate_parent_radius = sampled_axis_radius(axis, origin_arc, config)
    minimum_clearance = math.inf
    radii = axis.branch_origin_base_radii
    for index, (existing_arc, existing_azimuth) in enumerate(zip(
        axis.branch_origins, axis.branch_azimuths
    )):
        existing_parent_radius = sampled_axis_radius(axis, existing_arc, config)
        local_parent_radius = 0.5 * (
            candidate_parent_radius + existing_parent_radius
        )
        delta_arc = abs(float(origin_arc) - float(existing_arc))
        delta_theta = abs(
            (float(azimuth) - float(existing_azimuth) + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        circumferential_distance = local_parent_radius * delta_theta
        surface_distance = math.hypot(delta_arc, circumferential_distance)
        existing_collar = (
            float(radii[index]) if index < len(radii) else candidate_collar_radius
        )
        required = (
            config.branch_collar_clearance_factor
            * (candidate_collar_radius + existing_collar)
            + config.branch_collar_safety_margin
        )
        minimum_clearance = min(minimum_clearance, surface_distance - required)
    return minimum_clearance >= 0.0, float(minimum_clearance)


def origin_surface_clearance_rejection(
    axis: RootAxis,
    origin_arc: float,
    azimuth: float,
    candidate_collar_radius: float,
    config: SimulationConfig,
) -> str | None:
    available, _clearance = cylindrical_surface_clearance(
        axis, origin_arc, azimuth, candidate_collar_radius, config
    )
    return None if available else "surface_clearance"


def depth_resource_development_gate(
    point_z: float,
    nitrogen: float,
    potassium: float,
    config: SimulationConfig,
) -> float:
    """Reserve some branch construction for configured deeper resources.

    When N or K is part of the environment, a P-rich surface should not consume
    the entire size target before the anchor has had time to reach those zones.
    """

    depth = max(-float(point_z), 0.0)
    gate = 1.0
    if config.nitrogen_concentration > 0.10:
        n_local = float(nitrogen) / max(config.nitrogen_concentration, 1e-12)
        n_progress = depth / max(abs(config.nitrogen_z_high), 1.0)
        gate *= 0.25 + 0.75 * float(np.clip(max(n_local, n_progress), 0.0, 1.0))
    if config.potassium_concentration > 0.10:
        k_local = float(potassium) / max(config.potassium_concentration, 1e-12)
        k_progress = depth / max(abs(config.potassium_z_high), 1.0)
        gate *= 0.50 + 0.50 * float(np.clip(max(k_local, k_progress), 0.0, 1.0))
    return float(np.clip(gate, 0.08, 1.0))


def local_primordium_stimulus(
    axis: RootAxis,
    candidate_arc: float,
    step: int,
) -> float:
    """Return a bounded post-pass geometric signal near prior branch origins.

    Prior accepted origins contribute smooth axial and temporal kernels. Schema
    v26 evaluates this signal only after a probability pass; it may inform
    physical azimuth ranking and diagnostics but never the probability threshold.
    """

    if not axis.branch_origins or not axis.branch_origin_steps:
        return 0.0
    signal = 0.0
    axial_scale = max(PRIMORDIUM_STIMULUS_AXIAL_SCALE, 1e-12)
    lifetime = max(PRIMORDIUM_STIMULUS_LIFETIME_STEPS, 1e-12)
    # Contributions beyond six stimulus lifetimes are below 0.25%.
    maximum_age = int(math.ceil(6.0 * lifetime))
    for origin_arc, origin_step in zip(
        reversed(axis.branch_origins),
        reversed(axis.branch_origin_steps),
    ):
        age = max(0, int(step) - int(origin_step))
        if age > maximum_age:
            break
        axial_distance = (float(candidate_arc) - float(origin_arc)) / axial_scale
        signal += math.exp(-0.5 * axial_distance * axial_distance) * math.exp(
            -float(age) / lifetime
        )
    return float(np.clip(signal, 0.0, 1.0))


def advance_continuous_branch_sites(
    axis_store: RootAxisStore,
    axis: RootAxis,
    step: int,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> list[BranchSite]:
    """Create a Poisson point process on newly mature material arc."""

    if step - axis.birth_step < config.lateral_branch_min_age:
        return []
    mature_frontier = axis.total_length() - config.branch_min_distance_from_tip
    protected_base = config.branch_min_distance_from_base
    if mature_frontier < protected_base:
        return []
    mean_spacing = max(config.branch_min_spacing_along_axis, 1e-9)
    if axis.next_branch_site_arc is None:
        axis.next_branch_site_arc = (
            protected_base + max(float(rng.exponential(mean_spacing)), 1e-9)
        )
    created: list[BranchSite] = []
    while axis.next_branch_site_arc <= mature_frontier:
        site = BranchSite(
            site_id=len(axis_store.branch_sites),
            axis_id=axis.axis_id,
            material_arc=float(axis.next_branch_site_arc),
            birth_step=step,
            first_eligible_step=step,
            last_evaluated_parent_radius=sampled_axis_radius(
                axis, axis.next_branch_site_arc, config
            ),
        )
        axis_store.branch_sites.append(site)
        axis.branch_site_ids.append(site.site_id)
        created.append(site)
        gap = max(float(rng.exponential(mean_spacing)), 1e-9)
        axis.next_branch_site_arc += gap
    return created


def branch_site_has_surface_capacity(
    axis: RootAxis,
    site: BranchSite,
    config: SimulationConfig,
) -> bool:
    """Deterministically probe the current circumference for one open collar."""

    parent_radius = sampled_axis_radius(axis, site.material_arc, config)
    collar = candidate_branch_collar_radius(
        parent_radius, axis.branch_generation + 1, config
    )
    for azimuth in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        available, _ = cylindrical_surface_clearance(
            axis, site.material_arc, float(azimuth), collar, config
        )
        if available:
            return True
    return False


def eligible_branch_sites(
    axis_store: RootAxisStore,
    axis: RootAxis,
    step: int,
    config: SimulationConfig,
) -> list[BranchSite]:
    """Return sites receiving at most one initiation trial this step."""

    eligible: list[BranchSite] = []
    for site_id in axis.branch_site_ids:
        site = axis_store.branch_sites[site_id]
        if site.first_eligible_step > step or site.last_trial_step == step:
            continue
        if config.branch_retry_mode == "single_trial":
            if site.trial_count == 0 and not site.closed_in_single_trial_mode:
                eligible.append(site)
            continue
        if not site.physically_open:
            current_radius = sampled_axis_radius(axis, site.material_arc, config)
            if (
                current_radius > site.last_evaluated_parent_radius + 1e-12
                and branch_site_has_surface_capacity(axis, site, config)
            ):
                site.physically_open = True
                site.temporarily_surface_full = False
                site.last_evaluated_parent_radius = current_radius
                axis_store.branch_sites_reopened_after_thickening += 1
        if site.physically_open:
            eligible.append(site)
    return eligible


def physical_site_azimuth_candidates(
    primary_azimuth: float,
    rng: np.random.Generator,
) -> list[float]:
    """Return bounded alternative azimuths for one probability-passed site."""

    return [float(primary_azimuth)] + [
        float(rng.uniform(0.0, 2.0 * math.pi))
        for _ in range(max(0, PHYSICAL_ORIGIN_ATTEMPTS - 1))
    ]


def final_physical_rejection(reasons: Sequence[str]) -> str:
    """Choose one exhaustive final reason for a lost probability pass."""

    priority = {
        "surface_clearance": 2,
        "above_soil_surface": 3,
        "parent_collision": 4,
        "other_root_collision": 5,
    }
    if not reasons:
        return "other_root_collision"
    return max(reasons, key=lambda reason: priority.get(reason, 0))


def tip_extension_length(axis: RootAxis, step: int, config: SimulationConfig) -> float:
    """Return the strictly positive schema-v23 extension length.

    Primary tips use the decaying anchor law. Every lateral starts
    with a visible generation-scaled meristem rate, then receives the same
    exponential age slowdown and positive floor.  B.P. does not appear: an
    early lateral therefore cannot keep a constant 0.21-unit rate for all
    remaining steps, and the identical post-initiation law applies at every
    branch probability.
    """

    if axis.is_anchor_axis:
        return max(config.anchor_min_segment_length, anchor_segment_length(step, config))
    generation_index = max(axis.branch_generation - 1, 0)
    axis_age = max(0, int(step) - int(axis.birth_step))
    age_decay = math.exp(
        -float(axis_age) / config.lateral_elongation_decay_timescale
    )
    return max(
        config.lateral_min_segment_length,
        config.segment_length
        * config.lateral_relative_elongation
        * config.lateral_generation_length_decay ** generation_index
        * age_decay,
    )


def branch_escape_radial_direction(
    axis_store: RootAxisStore,
    axis: RootAxis,
) -> np.ndarray | None:
    """Return the fixed parent-relative outward direction for a newborn lateral."""

    if axis.is_anchor_axis or axis.parent_axis_id < 0:
        return None
    parent = axis_store.axes[axis.parent_axis_id]
    _center, parent_tangent = interpolate_axis(parent, axis.parent_arc_length)
    normal, binormal = local_orthonormal_frame(parent_tangent)
    azimuth = float(axis.parent_local_azimuth)
    return normalized_direction(
        math.cos(azimuth) * normal + math.sin(azimuth) * binormal
    )


def try_extend_axis(
    axis_store: RootAxisStore,
    axis: RootAxis,
    thickness_increment: float,
    step: int,
    rng: np.random.Generator,
    resource_field: HeterogeneousResourceField,
    raining: bool,
    config: SimulationConfig,
    demand_state: ResourceDemandState | None = None,
) -> str:
    """Give one active terminal tip one bounded physical extension attempt."""

    if not axis.active:
        return "inactive"
    axis.post_emergence_extension_attempts += 1
    local_values = resource_field.values(axis.points[-1], raining, config)
    if demand_state is not None:
        focus_age = int(step) - int(axis.resource_focus_last_update_step)
        focus_unavailable = (
            axis.resource_focus in RESOURCE_NAMES
            and demand_state.supply[
                RESOURCE_NAMES.index(axis.resource_focus)
            ] <= CAPTURE_REPORTING_EPSILON
        )
        if (
            focus_age >= config.resource_focus_persistence_steps
            or focus_unavailable
        ):
            focus_rng = np.random.default_rng(splitmix64(
                axis.resource_focus_seed
                ^ ((step + 1) * 0x94D049BB133111EB)
            ))
            maybe_update_resource_focus(
                axis,
                step,
                focus_rng,
                np.asarray(
                    [float(x[0]) for x in local_values],
                    dtype=np.float64,
                ),
                demand_state,
                config,
            )
    _, _, local_support, local_starvation = local_resource_support(
        *local_values, config
    )
    starvation_value = float(local_starvation[0])
    sufficiency_value = float(local_support[0])
    axis.local_resource_sufficiency_sum += sufficiency_value
    axis.local_resource_sufficiency_count += 1
    axis.last_local_resource_sufficiency = sufficiency_value
    axis_store.starvation_signal_sum += starvation_value
    axis_store.starvation_signal_count += 1
    axis_store.resource_support_gate_sum += sufficiency_value
    axis_store.resource_support_gate_count += 1
    axis_store.extension_starvation_sum += starvation_value
    axis_store.extension_starvation_count += 1
    length = tip_extension_length(axis, step, config)
    physical_rejections: list[str] = []
    in_escape = bool(
        not axis.is_anchor_axis and axis.escape_extensions_remaining > 0
    )
    escape_outward = (
        branch_escape_radial_direction(axis_store, axis) if in_escape else None
    )
    branch_origin = None
    if in_escape and axis.parent_axis_id >= 0:
        branch_origin, _parent_tangent = interpolate_axis(
            axis_store.axes[axis.parent_axis_id], axis.parent_arc_length
        )
    resource_context = prepare_direction_resource_context(
        axis,
        resource_field,
        raining,
        config,
        demand_state,
        current_resource_values=local_values,
    )
    for _ in range(config.tip_extension_candidate_attempts):
        direction_started = time.perf_counter()
        end_direction = biased_axis_direction(
            axis,
            rng,
            resource_field,
            raining,
            config,
            demand_state,
            escape_outward_direction=escape_outward,
            resource_context=resource_context,
        )
        axis_store.profile_resource_direction_candidates_sec += (
            time.perf_counter() - direction_started
        )
        if (
            end_direction[2] > 0.30
            or (
                axis.consecutive_upward_extensions
                >= config.maximum_consecutive_upward_extensions
                and end_direction[2] > config.upward_component_threshold
            )
        ):
            physical_rejections.append("bounded_upward")
            continue
        samples, tangents = hermite_curve_samples(
            axis.points[-1],
            axis.tangents[-1],
            end_direction,
            length,
            config.curve_samples_per_extension,
        )
        if not axis_store.complete_insertion_fits(samples.shape[0]):
            return "sample_cap"
        collision_started = time.perf_counter()
        rejection = axis_store.samples_clearance_rejection(
            samples,
            own_axis_id=axis.axis_id,
            parent_axis_id=axis.parent_axis_id if in_escape else -1,
            branch_origin=branch_origin,
            allow_shared_emergence_shoulders=False,
        )
        axis_store.profile_collision_queries_sec += (
            time.perf_counter() - collision_started
        )
        if rejection is None:
            axis_store.append_curve_extension(axis, samples, tangents, step)
            accepted_direction = normalized_direction(end_direction)
            axis_store.accepted_extension_directions.append(accepted_direction.copy())
            axis_store.accepted_extension_generations.append(axis.branch_generation)
            axis_store.accepted_extension_foci.append(axis.resource_focus)
            axis_store.accepted_extensions_by_focus[axis.resource_focus] = (
                axis_store.accepted_extensions_by_focus.get(axis.resource_focus, 0) + 1
            )
            if accepted_direction[2] > config.upward_component_threshold:
                axis.consecutive_upward_extensions += 1
            else:
                axis.consecutive_upward_extensions = 0
            axis.maximum_consecutive_upward_extensions = max(
                axis.maximum_consecutive_upward_extensions,
                axis.consecutive_upward_extensions,
            )
            for name, value in axis.last_direction_score_components.items():
                axis_store.direction_score_component_sum[name] = (
                    axis_store.direction_score_component_sum.get(name, 0.0) + float(value)
                )
                axis_store.direction_score_component_max[name] = max(
                    axis_store.direction_score_component_max.get(name, -math.inf),
                    float(value),
                )
            axis_store.direction_score_evaluations += 1
            record_transport_path_growth(
                axis_store,
                axis,
                thickness_increment=thickness_increment,
                grown_length=length,
            )
            axis.extension_events += 1
            axis.post_emergence_extensions_accepted += 1
            axis.accepted_extension_length_sum += float(length)
            axis.post_emergence_direction_z_sum += float(accepted_direction[2])
            axis.post_emergence_direction_count += 1
            if in_escape:
                axis.escape_extensions_remaining = max(
                    0, axis.escape_extensions_remaining - 1
                )
            axis.last_growth_step = step
            axis.collision_failure_streak = 0
            if demand_state is not None:
                demand_state.accumulate_points(samples, resource_field, raining, config)
            return "accepted"
        physical_rejections.append(rejection)
        axis.collision_failure_streak += 1
    if physical_rejections and all(
        reason == "above_soil_surface" for reason in physical_rejections
    ):
        axis.surface_blocked_extensions += 1
        return "surface"
    if physical_rejections and any(
        reason in {"parent_collision", "other_root_collision"}
        for reason in physical_rejections
    ):
        if "other_root_collision" in physical_rejections:
            axis.other_root_collision_blocked_extensions += 1
        else:
            axis.parent_collision_blocked_extensions += 1
        return "collision"
    axis.other_blocked_extensions += 1
    return "other"


def axis_branch_threshold(
    base_probability: float,
    config: SimulationConfig,
) -> float:
    """Return the lineage-only probability for one mature branch site.

    The config argument makes the fixed model boundary explicit, but no config,
    resource, demand, starvation, focus, stimulus, or geometry value is read.
    Physical feasibility is evaluated only after this probability passes.
    """

    del config
    return float(np.clip(base_probability, 0.0, 0.99))


def lineage_branch_probability(
    base_probability: float,
    parent_generation: int,
    config: SimulationConfig,
) -> float:
    """Reduce descendant initiation without changing first-order B.P.

    B.P. remains the probability for a mature site on the primary axis.
    Higher-order sites pay a generation-dependent exponent so a rare primary
    initiation cannot recursively explode into a dicot-like crown.
    """

    base = float(np.clip(base_probability, 0.0, 0.99))
    generation = max(0, int(parent_generation))
    if generation == 0:
        return base
    exponent = 1.0 + config.lateral_generation_probability_exponent * generation
    return float(base ** exponent)


def site_initiation_probability(
    configured_branch_probability: float,
    parent_generation: int,
    config: SimulationConfig,
) -> float:
    """Return the complete schema-v26 branch-site probability.

    Its deliberately narrow signature is the scientific boundary: only the
    configured B.P., parent generation, and fixed lineage exponent are available.
    """

    lineage_probability = lineage_branch_probability(
        configured_branch_probability, parent_generation, config
    )
    return axis_branch_threshold(lineage_probability, config)


def detect_local_whorls(
    axis_store: RootAxisStore,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Classify local primary-axis branch groups after growth has finished.

    This diagnostic is never called by or read from any growth decision.
    """

    labels = np.full(len(axis_store.axes), -1, dtype=np.int32)
    first_order = sorted(
        (
            axis for axis in axis_store.axes
            if axis.parent_axis_id == 0 and axis.branch_generation == 1
        ),
        key=lambda axis: axis.parent_arc_length,
    )
    events: list[dict[str, object]] = []
    index = 0
    window = POSTHOC_WHORL_AXIAL_WINDOW
    while index < len(first_order):
        start = index
        end = start + 1
        while (
            end < len(first_order)
            and first_order[end].parent_arc_length
            - first_order[start].parent_arc_length <= window
        ):
            end += 1
        group = first_order[start:end]
        if len(group) >= POSTHOC_WHORL_MIN_BRANCHES:
            event_id = len(events)
            center_arc = float(np.mean([
                axis.parent_arc_length for axis in group
            ]))
            origin, _ = interpolate_axis(axis_store.axes[0], center_arc)
            for axis in group:
                labels[axis.axis_id] = event_id
            events.append({
                "whorl_id": event_id,
                "parent_axis_id": 0,
                "origin_arc": center_arc,
                "depth": float(max(-origin[2], 0.0)),
                "branch_count": len(group),
                "azimuths": [axis.parent_local_azimuth for axis in group],
                "axial_offsets": [
                    axis.parent_arc_length - center_arc for axis in group
                ],
                "emergence_angles": [
                    vector_angle_degrees(
                        interpolate_axis(
                            axis_store.axes[0], axis.parent_arc_length
                        )[1],
                        axis.tangents[0],
                    )
                    for axis in group
                ],
            })
            index = end
        else:
            index += 1
    return events, labels


def effective_sampled_point_cap(config: SimulationConfig) -> int:
    """Resolve technical sampled-point limits without defining biological age."""

    limits = [
        int(config.max_nodes),
        int(config.max_sampled_points),
        int(config.interactive_safety_cap),
    ]
    if config.target_architecture_size > 0:
        # Deprecated alias: it can only tighten the technical safety cap.
        limits.append(int(config.target_architecture_size))
    return int(max(2, min(limits)))


def effective_target_architecture_size(config: SimulationConfig) -> int:
    """Deprecated pre-v22 name for the sampled-point safety cap."""

    return effective_sampled_point_cap(config)


def effective_target_axis_count(
    config: SimulationConfig,
    sampled_point_cap: int,
) -> int:
    """Return a non-generative physical/memory safety ceiling.

    B.P. never enters this limit. The sampled-point cap supplies only the
    absolute number of axes that can fit in allocated memory; it does not
    request, schedule, or otherwise create axes.
    """

    support_per_axis = max(3.0, float(config.curve_samples_per_extension) + 1.0)
    physical_capacity = max(1, int(sampled_point_cap / support_per_axis))
    if config.target_axis_count > 0:
        # Retained as an explicit backwards-compatible safety ceiling only.
        physical_capacity = min(physical_capacity, int(config.target_axis_count))
    return int(max(1, physical_capacity))


def effective_developmental_steps(config: SimulationConfig) -> int:
    """Return fixed biological age, with one deprecated explicit override."""

    if config.max_growth_iterations > 0:
        return int(config.max_growth_iterations)
    return int(config.steps)


def effective_max_growth_iterations(config: SimulationConfig) -> int:
    """Deprecated compatibility name for effective_developmental_steps."""

    return effective_developmental_steps(config)


def _resource_capture_for_sampled_store(
    store: NodeStore,
    parameters: SimulationParameters,
    config: SimulationConfig,
    resource_field: HeterogeneousResourceField,
    rain_fraction: float,
    steps_completed: int,
) -> None:
    """Populate resource capture arrays after curve-axis sampling."""

    size = store.size
    if size <= 0:
        return
    points = store.position[:size]
    water, phosphorus, nitrogen, potassium = resource_field.values(
        points, rain_fraction, config
    )
    observations = np.maximum(
        1,
        int(steps_completed) - store.birth_step[:size] + 1,
    ).astype(np.int32)
    store.resource_observations[:size] = observations
    for values, target in (
        (water, store.water_availability_sum),
        (phosphorus, store.phosphorus_availability_sum),
        (nitrogen, store.nitrogen_availability_sum),
        (potassium, store.potassium_availability_sum),
    ):
        target[:size] = values * observations
    if size <= 1:
        return
    child = np.arange(1, size)
    parent = store.parent[1:size]
    segment_length = np.linalg.norm(points[1:size] - points[parent], axis=1)
    exposed_area = 2.0 * math.pi * store.radius[1:size] * segment_length
    obs = observations[1:size].astype(np.float64)
    depth = np.maximum(-points[1:size, 2], 0.0)
    resource_specs = (
        (water[1:size], config.water_capture_per_iteration, store.water_captured,
         store.water_capture_depth_sum, store.deepest_water_capture),
        (phosphorus[1:size], config.phosphorus_capture_per_iteration, store.phosphorus_captured,
         store.phosphorus_capture_depth_sum, store.deepest_phosphorus_capture),
        (nitrogen[1:size], config.nitrogen_capture_per_iteration, store.nitrogen_captured,
         store.nitrogen_capture_depth_sum, store.deepest_nitrogen_capture),
        (potassium[1:size], config.potassium_capture_per_iteration, store.potassium_captured,
         store.potassium_capture_depth_sum, store.deepest_potassium_capture),
    )
    for availability, rate, captured_total, depth_total, deepest in resource_specs:
        captured = availability * rate * exposed_area * obs
        captured_total[1:size] = captured
        depth_total[1:size] = captured * depth
        positive = captured > CAPTURE_REPORTING_EPSILON
        if np.any(positive):
            selected = child[positive]
            deepest[selected] = np.maximum(deepest[selected], depth[positive])


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """Return the minimum Euclidean distance from a point to a sampled polyline."""

    target = np.asarray(point, dtype=np.float64)
    line = np.asarray(polyline, dtype=np.float64)
    if line.shape[0] <= 1:
        return float(np.linalg.norm(target - line[0])) if line.size else math.inf
    starts = line[:-1]
    segments = line[1:] - starts
    squared = np.einsum("ij,ij->i", segments, segments)
    projection = np.divide(
        np.einsum("ij,ij->i", target[None, :] - starts, segments),
        squared,
        out=np.zeros_like(squared),
        where=squared > 1e-18,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, None] * segments
    return float(np.min(np.linalg.norm(closest - target[None, :], axis=1)))


def lateral_axis_diagnostics(
    axis_store: RootAxisStore,
    resource_field: HeterogeneousResourceField,
    raining: bool,
    completed_steps: int,
    config: SimulationConfig,
) -> list[dict[str, int | float | bool]]:
    """Export per-lateral age, growth, collision, and shoulder diagnostics."""

    rows: list[dict[str, int | float | bool]] = []
    completed_step = max(-1, int(completed_steps) - 1)
    for axis in axis_store.axes[1:]:
        parent = axis_store.axes[axis.parent_axis_id]
        parent_points = np.asarray(parent.points, dtype=np.float64)
        parent_center, parent_tangent = interpolate_axis(
            parent, axis.parent_arc_length
        )
        shoulder_index = min(
            max(int(axis.initial_shoulder_point_count) - 1, 0),
            len(axis.points) - 1,
        )
        shoulder_end = np.asarray(axis.points[shoulder_index], dtype=np.float64)
        shoulder_delta = shoulder_end - parent_center
        shoulder_radial = shoulder_delta - (
            float(np.dot(shoulder_delta, parent_tangent)) * parent_tangent
        )
        initial_radial_displacement = float(np.linalg.norm(shoulder_radial))
        distance_after_shoulder = point_to_polyline_distance(
            shoulder_end, parent_points
        )
        parent_radius = sampled_axis_radius(
            parent, axis.parent_arc_length, config
        )
        post_shoulder_points = np.asarray(
            axis.points[axis.initial_shoulder_point_count:], dtype=np.float64
        )
        curves_back_inside = bool(
            post_shoulder_points.size
            and any(
                point_to_polyline_distance(point, parent_points)
                < parent_radius - 1e-12
                for point in post_shoulder_points
            )
        )
        local_values = resource_field.values(axis.points[-1], raining, config)
        _water, _nutrients, sufficiency, _starvation = local_resource_support(
            *local_values, config
        )
        accepted = int(axis.post_emergence_extensions_accepted)
        attempts = int(axis.post_emergence_extension_attempts)
        collision_blocked = int(
            axis.parent_collision_blocked_extensions
            + axis.other_root_collision_blocked_extensions
        )
        rows.append({
            "axis_id": int(axis.axis_id),
            "parent_axis_id": int(axis.parent_axis_id),
            "generation": int(axis.branch_generation),
            "birth_step": int(axis.birth_step),
            "completed_simulation_step": int(completed_steps),
            "biological_age_steps": max(0, completed_step - int(axis.birth_step)),
            "extension_attempts": attempts,
            "accepted_extensions": accepted,
            "collision_blocked_extensions": collision_blocked,
            "parent_collision_blocked_extensions": int(
                axis.parent_collision_blocked_extensions
            ),
            "other_root_collision_blocked_extensions": int(
                axis.other_root_collision_blocked_extensions
            ),
            "surface_blocked_extensions": int(axis.surface_blocked_extensions),
            "other_blocked_extensions": int(axis.other_blocked_extensions),
            "current_arc_length": float(axis.total_length()),
            "initial_emergence_shoulder_length": float(
                axis.initial_emergence_shoulder_length
            ),
            "mean_accepted_extension_length": _safe_ratio(
                axis.accepted_extension_length_sum, accepted
            ),
            "last_accepted_growth_step": int(axis.last_growth_step),
            "current_active_state": bool(axis.active),
            "birth_resource_sufficiency": float(axis.birth_resource_support),
            "mean_local_resource_sufficiency": _safe_ratio(
                axis.local_resource_sufficiency_sum,
                axis.local_resource_sufficiency_count,
            ),
            "current_local_resource_sufficiency": float(sufficiency[0]),
            "mean_direction_z_after_emergence": _safe_ratio(
                axis.post_emergence_direction_z_sum,
                axis.post_emergence_direction_count,
            ),
            "initial_radial_displacement": initial_radial_displacement,
            "distance_from_parent_after_shoulder": distance_after_shoulder,
            "curves_back_inside_parent_radius": curves_back_inside,
            "only_initial_shoulder": bool(accepted == 0),
            "only_one_support_curve": bool(axis.extension_events == 1),
            "collision_rate": _safe_ratio(collision_blocked, attempts),
            "escape_extensions_remaining": int(axis.escape_extensions_remaining),
        })
    return rows


def axis_store_to_node_store(
    axis_store: RootAxisStore,
    parameters: SimulationParameters,
    config: SimulationConfig,
    resource_field: HeterogeneousResourceField,
    rain_fraction: float,
    steps_completed: int,
    branch_parent_records: Sequence[tuple[int, int, str]],
) -> NodeStore:
    """Export continuous axes as sampled graph support for metrics/app/CSV."""

    capacity = min(max(config.initial_capacity, axis_store.estimated_sample_count() + 8), config.max_nodes)
    store = NodeStore(capacity, config.base_radius)
    store.radius[0] = sampled_axis_radius(axis_store.axes[0], 0.0, config)
    store.thickness[0] = float(axis_store.axes[0].structural_area_total)
    axis_node_ids: dict[int, list[int]] = {0: [0]}
    axis_arc_values: dict[int, np.ndarray] = {0: np.array([0.0], dtype=np.float64)}
    branch_node_by_axis: dict[int, int] = {}

    for axis in axis_store.axes:
        arcs = axis.arc_lengths()
        if axis.axis_id == 0:
            previous_node = 0
            node_ids = [0]
            for index in range(1, len(axis.points)):
                point = axis.points[index]
                parent_point = store.position[previous_node]
                delta = point - parent_point
                length = float(np.linalg.norm(delta))
                if length <= 1e-12 or store.size >= config.max_nodes:
                    continue
                direction = delta / length
                new_id = store.append(
                    parent=previous_node,
                    position=point,
                    direction=direction,
                    edge_length=length,
                    radius=sampled_axis_radius(axis, float(arcs[index]), config),
                    attachment_angle=math.nan,
                    birth_step=axis.point_birth_steps[index],
                    is_anchor=True,
                    is_axis_continuation=True,
                )
                node_ids.append(new_id)
                previous_node = new_id
            axis_node_ids[axis.axis_id] = node_ids
            axis_arc_values[axis.axis_id] = arcs[:len(node_ids)]
            continue

        parent_nodes = axis_node_ids.get(axis.parent_axis_id, [0])
        parent_arcs = axis_arc_values.get(
            axis.parent_axis_id,
            np.zeros(len(parent_nodes), dtype=np.float64),
        )
        nearest_index = int(np.argmin(np.abs(parent_arcs - axis.parent_arc_length)))
        previous_node = int(parent_nodes[nearest_index])
        node_ids = []
        for index in range(1, len(axis.points)):
            point = axis.points[index]
            parent_point = store.position[previous_node]
            delta = point - parent_point
            length = float(np.linalg.norm(delta))
            if length <= 1e-12 or store.size >= config.max_nodes:
                continue
            direction = delta / length
            is_first = len(node_ids) == 0
            new_id = store.append(
                parent=previous_node,
                position=point,
                direction=direction,
                edge_length=length,
                radius=sampled_axis_radius(axis, float(arcs[index]), config),
                attachment_angle=axis.parent_local_azimuth if is_first else math.nan,
                birth_step=axis.point_birth_steps[index],
                is_anchor=False,
                is_axis_continuation=not is_first,
            )
            if is_first:
                branch_node_by_axis[axis.axis_id] = new_id
            node_ids.append(new_id)
            previous_node = new_id
        axis_node_ids[axis.axis_id] = node_ids
        axis_arc_values[axis.axis_id] = arcs[1:1 + len(node_ids)]

    for axis_id, _step, reason in branch_parent_records:
        node_id = branch_node_by_axis.get(axis_id)
        if node_id is None:
            continue
        parent = int(store.parent[node_id])
        if parent >= 0:
            store.branch_opportunities[parent] += 1
            if reason == "success":
                store.probability_passes[parent] += 1
                store.successful_branches[parent] += 1
            elif reason == "angle":
                store.failed_angle[parent] += 1
            elif reason == "inflation":
                store.failed_inflation[parent] += 1
            elif reason == "spatial":
                store.failed_spatial[parent] += 1

    _resource_capture_for_sampled_store(
        store, parameters, config, resource_field, rain_fraction, steps_completed
    )
    anchor_axis = axis_store.axes[0]
    anchor_points = np.asarray(anchor_axis.points, dtype=np.float64)
    first_order_axes = [
        axis for axis in axis_store.axes
        if axis.parent_axis_id == 0 and axis.branch_generation == 1
    ]
    primary_total_arc = max(anchor_axis.total_length(), 1e-12)
    first_order_origin_depths = np.asarray([
        max(-float(interpolate_axis(anchor_axis, axis.parent_arc_length)[0][2]), 0.0)
        for axis in first_order_axes
    ], dtype=np.float64)
    first_order_origin_arc_fractions = np.asarray([
        float(np.clip(axis.parent_arc_length / primary_total_arc, 0.0, 1.0))
        for axis in first_order_axes
    ], dtype=np.float64)
    lateral_diagnostics = lateral_axis_diagnostics(
        axis_store,
        resource_field,
        bool(rain_fraction > 0.0),
        steps_completed,
        config,
    )
    store.axis_metadata = {
        "curve_model_version": CURVE_MODEL_VERSION,
        "canonical_branch_min_spacing_along_axis": (
            CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS
        ),
        "branch_retry_mode": config.branch_retry_mode,
        "branch_retry_modes_available": list(BRANCH_RETRY_MODES),
        "branch_retry_mode_is_grid_dimension": False,
        "canonical_production_branch_retry_mode": None,
        "branch_retry_mode_production_calibrated": False,
        "axis_count": len(axis_store.axes),
        "soil_surface_z": float(config.soil_surface_z),
        "max_above_surface_tolerance": float(config.max_above_surface_tolerance),
        "axis_arc_lengths": np.asarray([axis.total_length() for axis in axis_store.axes], dtype=np.float64),
        "axis_generations": np.asarray(
            [axis.branch_generation for axis in axis_store.axes], dtype=np.int32
        ),
        "axis_radius_scales": np.asarray(
            [axis.radius_scale for axis in axis_store.axes], dtype=np.float64
        ),
        "axis_parent_ids": np.asarray(
            [axis.parent_axis_id for axis in axis_store.axes], dtype=np.int32
        ),
        "axis_parent_arc_lengths": np.asarray(
            [axis.parent_arc_length for axis in axis_store.axes], dtype=np.float64
        ),
        "axis_parent_local_azimuths": np.asarray(
            [axis.parent_local_azimuth for axis in axis_store.axes], dtype=np.float64
        ),
        "axis_node_ids": [
            np.asarray(axis_node_ids.get(axis.axis_id, []), dtype=np.int32)
            for axis in axis_store.axes
        ],
        "axis_parent_local_radii": np.asarray([
            0.0 if axis.parent_axis_id < 0 else sampled_axis_radius(
                axis_store.axes[axis.parent_axis_id],
                axis.parent_arc_length,
                config,
            )
            for axis in axis_store.axes
        ], dtype=np.float64),
        "axis_basal_radii": np.asarray([
            sampled_axis_radius(axis, 0.0, config)
            for axis in axis_store.axes
        ], dtype=np.float64),
        "axis_birth_steps": np.asarray(
            [axis.birth_step for axis in axis_store.axes], dtype=np.int32
        ),
        "axis_origin_site_ids": np.asarray(
            [axis.origin_site_id for axis in axis_store.axes], dtype=np.int32
        ),
        "axis_extension_events": np.asarray(
            [axis.extension_events for axis in axis_store.axes], dtype=np.int32
        ),
        "axis_last_growth_steps": np.asarray(
            [axis.last_growth_step for axis in axis_store.axes], dtype=np.int32
        ),
        "axis_structural_allocations": np.asarray(
            [axis.structural_area_total for axis in axis_store.axes], dtype=np.float64
        ),
        "axis_points": [
            np.asarray(axis.points, dtype=np.float64) for axis in axis_store.axes
        ],
        "axis_material_arcs": [
            axis.arc_lengths() for axis in axis_store.axes
        ],
        "axis_radii": [
            axis_radii_at_arcs(axis, axis.arc_lengths(), config)
            for axis in axis_store.axes
        ],
        "axis_branch_origins": [
            np.asarray(axis.branch_origins, dtype=np.float64)
            for axis in axis_store.axes
        ],
        "axis_structural_area_events": [
            list(axis.structural_area_events) for axis in axis_store.axes
        ],
        "branch_sites": [asdict(site) for site in axis_store.branch_sites],
        "lateral_axis_diagnostics": lateral_diagnostics,
        "axis_whorl_ids": np.asarray(
            getattr(
                axis_store,
                "axis_whorl_ids",
                np.full(len(axis_store.axes), -1, dtype=np.int32),
            ),
            dtype=np.int32,
        ),
        "whorl_events": list(getattr(axis_store, "whorl_events", [])),
        "branch_origin_candidate_evaluations": int(
            axis_store.branch_origin_candidate_evaluations
        ),
        "first_order_origin_depths": first_order_origin_depths,
        "first_order_origin_arc_fractions": first_order_origin_arc_fractions,
        "curvatures": np.asarray(axis_store.curvature_values, dtype=np.float64),
        "tip_bend_angles_deg": np.asarray(axis_store.tip_bend_angles_deg, dtype=np.float64),
        "branch_emergence_angles_deg": np.asarray(axis_store.branch_emergence_angles_deg, dtype=np.float64),
        "branch_azimuth_angles": np.asarray(axis_store.branch_azimuth_angles, dtype=np.float64),
        "branch_origin_spacings": np.asarray(
            [x for x in axis_store.branch_origin_spacings if math.isfinite(float(x))],
            dtype=np.float64,
        ),
        "mean_curve_collision_samples_per_growth": _safe_ratio(
            axis_store.collision_sample_checks, axis_store.growth_attempts
        ),
        "anchor_points": anchor_points,
        "profile_retry_site_traversal_sec": float(
            axis_store.profile_retry_site_traversal_sec
        ),
        "profile_branch_probability_trials_sec": float(
            axis_store.profile_branch_probability_trials_sec
        ),
        "profile_physical_origin_search_sec": float(
            axis_store.profile_physical_origin_search_sec
        ),
        "profile_active_tip_extensions_sec": float(
            axis_store.profile_active_tip_extensions_sec
        ),
        "profile_collision_queries_sec": float(
            axis_store.profile_collision_queries_sec
        ),
        "profile_resource_direction_candidates_sec": float(
            axis_store.profile_resource_direction_candidates_sec
        ),
    }
    return store


def _axis_curve_simulation(
    parameters: SimulationParameters,
    config: SimulationConfig,
    *,
    return_store: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_interval_steps: int = 0,
    resume_checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    progress_interval_steps: int = 1,
    pause_after_checkpoint_step: int | None = None,
) -> dict[str, int | float | str] | tuple[dict[str, int | float | str], NodeStore]:
    """Schema-v26 fixed-time, all-active-tip developmental core."""

    started_at = time.perf_counter()
    rng = np.random.default_rng(parameters.seed)
    resource_field = HeterogeneousResourceField(parameters.seed, config)
    axis_store = RootAxisStore(config)
    axis_store.axes[0].resource_focus_seed = int(
        splitmix64(parameters.seed ^ 0xF0C0_5EED_0000_0001)
    )
    demand_state = ResourceDemandState(config)
    demand_state.begin_step(resource_field.environment, config)
    sampled_point_cap = effective_sampled_point_cap(config)
    target_axis_count = effective_target_axis_count(config, sampled_point_cap)
    developmental_steps = effective_developmental_steps(config)
    expected_water = np.clip(
        config.soil_water_background
        + parameters.rain_probability * config.rain_water_input,
        0.0,
        1.0,
    )
    _, _, global_support, global_starvation = local_resource_support(
        expected_water,
        config.phosphorus_concentration,
        config.nitrogen_concentration,
        config.potassium_concentration,
        config,
    )
    global_starvation = float(global_starvation)

    rainy_iterations = 0
    branch_opportunities = 0
    threshold_sum = 0.0
    probability_passes = 0
    probability_failures = 0
    rejected_origin_surface_clearance = 0
    rejected_above_soil_surface = 0
    rejected_parent_collision = 0
    rejected_other_root_collision = 0
    rejected_axis_ceiling = 0
    rejected_sample_cap = 0
    accepted_first_order_laterals = 0
    accepted_higher_order_laterals = 0
    stimulus_evaluated_probability_passes = 0
    primordium_stimulus_sum = 0.0
    initiation_uniform_sum = 0.0
    initiation_uniform_min = 1.0
    initiation_uniform_max = 0.0
    dry_probability_passes = 0
    rain_probability_passes = 0
    successful = 0
    successful_by_depth = [0, 0, 0, 0]
    failed_angle = 0
    failed_inflation = 0
    failed_spatial = 0
    status = "developmental_steps_complete"
    steps_completed = 0
    branch_parent_records: list[tuple[int, int, str]] = []
    start_step = 0
    elapsed_before_resume = 0.0
    axis_store.time_series_snapshots.append({
        "step": 0,
        "axis_count": 1,
        "sampled_points": 1,
        "effective_wetting_depth": resource_field.environment.effective_wetting_depth,
        "effective_nitrate_depth": resource_field.environment.effective_nitrate_depth,
        "effective_potassium_depth": resource_field.environment.effective_potassium_depth,
    })

    if resume_checkpoint_path is not None:
        from root_hpc_storage import load_checkpoint

        checkpoint = load_checkpoint(
            Path(resume_checkpoint_path),
            simulator_path=Path(__file__).resolve(),
            schema_version=SCHEMA_VERSION,
            config=config,
            seed=parameters.seed,
            task_index=parameters.task_index,
        )
        saved = checkpoint["state"]
        axis_store = saved["axis_store"]
        resource_field = saved["resource_field"]
        demand_state = saved["demand_state"]
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng_bit_generator_state"]
        rainy_iterations = int(saved["rainy_iterations"])
        branch_opportunities = int(saved["branch_opportunities"])
        threshold_sum = float(saved["threshold_sum"])
        probability_passes = int(saved["probability_passes"])
        probability_failures = int(saved["probability_failures"])
        rejected_origin_surface_clearance = int(
            saved["rejected_origin_surface_clearance"]
        )
        rejected_above_soil_surface = int(saved["rejected_above_soil_surface"])
        rejected_parent_collision = int(saved["rejected_parent_collision"])
        rejected_other_root_collision = int(
            saved["rejected_other_root_collision"]
        )
        rejected_axis_ceiling = int(saved["rejected_axis_ceiling"])
        rejected_sample_cap = int(saved["rejected_sample_cap"])
        accepted_first_order_laterals = int(
            saved["accepted_first_order_laterals"]
        )
        accepted_higher_order_laterals = int(
            saved["accepted_higher_order_laterals"]
        )
        stimulus_evaluated_probability_passes = int(
            saved["stimulus_evaluated_probability_passes"]
        )
        primordium_stimulus_sum = float(saved["primordium_stimulus_sum"])
        initiation_uniform_sum = float(saved["initiation_uniform_sum"])
        initiation_uniform_min = float(saved["initiation_uniform_min"])
        initiation_uniform_max = float(saved["initiation_uniform_max"])
        dry_probability_passes = int(saved["dry_probability_passes"])
        rain_probability_passes = int(saved["rain_probability_passes"])
        successful = int(saved["successful"])
        successful_by_depth = list(saved["successful_by_depth"])
        steps_completed = int(saved["steps_completed"])
        branch_parent_records = list(saved["branch_parent_records"])
        elapsed_before_resume = float(
            saved.get("elapsed_execution_time_sec", 0.0)
        )
        started_at = time.perf_counter() - elapsed_before_resume
        start_step = steps_completed
        status = "developmental_steps_complete"

    for step in range(start_step, developmental_steps):
        if (
            config.max_seconds_per_simulation > 0.0
            and time.perf_counter() - started_at >= config.max_seconds_per_simulation
        ):
            status = "time_limit"
            break
        if axis_store.estimated_sample_count() >= sampled_point_cap:
            status = "sample_cap"
            break
        raining = bool(rng.random() < parameters.rain_probability)
        if raining:
            rainy_iterations += 1
        resource_field.environment.update(step, raining, config)
        demand_state.begin_step(resource_field.environment, config)

        active_ids = axis_store.active_axis_ids()
        if not active_ids:
            status = "no_active_axes"
            break

        active_snapshot = list(active_ids)
        step_start_sample_count = axis_store.estimated_sample_count()
        active_count = len(active_snapshot)
        axis_store.active_tips_at_step_start_total += active_count
        axis_store.active_tip_count_observations += 1
        axis_store.active_tip_count_max = max(
            axis_store.active_tip_count_max, active_count
        )
        for axis_id in active_snapshot:
            axis = axis_store.axes[int(axis_id)]
            axis_store.tip_extension_attempts += 1
            if axis.is_anchor_axis:
                axis_store.primary_tip_extension_attempts += 1
            else:
                axis_store.lateral_tip_extension_attempts += 1
                if axis.branch_generation == 1:
                    axis_store.generation_1_extension_attempts += 1
                elif axis.branch_generation == 2:
                    axis_store.generation_2_extension_attempts += 1
                else:
                    axis_store.generation_3plus_extension_attempts += 1
            extension_started = time.perf_counter()
            outcome = try_extend_axis(
                axis_store,
                axis,
                parameters.thickness_increment,
                step,
                rng,
                resource_field,
                raining,
                config,
                demand_state,
            )
            axis_store.profile_active_tip_extensions_sec += (
                time.perf_counter() - extension_started
            )
            if outcome == "sample_cap":
                axis_store.tip_extensions_sample_cap_blocked += 1
                status = "sample_cap"
                break
            if outcome == "accepted":
                axis_store.tip_extensions_accepted += 1
                if axis.is_anchor_axis:
                    axis_store.primary_tip_extensions_accepted += 1
                else:
                    axis_store.lateral_tip_extensions_accepted += 1
                    if axis.branch_generation == 1:
                        axis_store.generation_1_extensions_accepted += 1
                    elif axis.branch_generation == 2:
                        axis_store.generation_2_extensions_accepted += 1
                    else:
                        axis_store.generation_3plus_extensions_accepted += 1
            elif outcome == "collision":
                axis_store.tip_extensions_collision_blocked += 1
            elif outcome == "surface":
                axis_store.tip_extensions_surface_blocked += 1
            else:
                axis_store.tip_extensions_other_blocked += 1
        if status == "sample_cap":
            break
        # New laterals enter the active snapshot on the following step.
        for axis_id in active_snapshot:
            advance_continuous_branch_sites(
                axis_store,
                axis_store.axes[int(axis_id)],
                step,
                rng,
                config,
            )
        for axis_id in active_snapshot:
            if axis_store.estimated_sample_count() >= sampled_point_cap:
                status = "sample_cap"
                break
            axis = axis_store.axes[int(axis_id)]
            traversal_started = time.perf_counter()
            eligible_sites = eligible_branch_sites(
                axis_store, axis, step, config
            )
            axis_store.profile_retry_site_traversal_sec += (
                time.perf_counter() - traversal_started
            )
            for site in eligible_sites:
                if axis_store.estimated_sample_count() >= sampled_point_cap:
                    status = "sample_cap"
                    break
                origin_arc = float(site.material_arc)
                probability_started = time.perf_counter()
                threshold = site_initiation_probability(
                    parameters.branch_probability,
                    axis.branch_generation,
                    config,
                )
                trial_number = site.trial_count + 1
                initiation_uniform = initiation_probability_uniform(
                    parameters.seed,
                    parameters.task_index,
                    axis.axis_id,
                    site.site_id,
                    trial_number,
                )
                site.trial_count = trial_number
                site.last_trial_step = step
                site.last_initiation_uniform = initiation_uniform
                site.last_initiation_threshold = threshold
                site.last_initiation_passed = initiation_uniform < threshold
                axis_store.profile_branch_probability_trials_sec += (
                    time.perf_counter() - probability_started
                )
                if config.branch_retry_mode == "single_trial":
                    site.closed_in_single_trial_mode = True
                    axis_store.branch_sites_closed_single_trial += 1
                branch_opportunities += 1
                threshold_sum += threshold
                initiation_uniform_sum += initiation_uniform
                initiation_uniform_min = min(initiation_uniform_min, initiation_uniform)
                initiation_uniform_max = max(initiation_uniform_max, initiation_uniform)
                if not site.last_initiation_passed:
                    site.failure_count += 1
                    probability_failures += 1
                    continue
                site.probability_pass_count += 1
                probability_passes += 1
                if raining:
                    rain_probability_passes += 1
                else:
                    dry_probability_passes += 1
                # Post-initiation state can affect geometry but not the probability pass.
                local_stimulus = local_primordium_stimulus(axis, origin_arc, step)
                primordium_stimulus_sum += local_stimulus
                stimulus_evaluated_probability_passes += 1
                if len(axis_store.axes) >= target_axis_count:
                    rejected_axis_ceiling += 1
                    continue

                physical_search_started = time.perf_counter()
                azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
                origin, _ = interpolate_axis(axis, origin_arc)
                water, phosphorus, nitrogen, potassium = resource_field.values(
                    origin, raining, config
                )
                _water_gate, _nutrient_gate, support_gate, starvation = local_resource_support(
                    water, phosphorus, nitrogen, potassium, config
                )
                starvation_value = float(starvation[0])
                support_value = float(support_gate[0])
                axis_store.branch_origin_starvation_sum += starvation_value
                axis_store.branch_origin_starvation_count += 1
                axis_store.resource_support_gate_sum += support_value
                axis_store.resource_support_gate_count += 1
                branch: RootAxis | None = None
                accepted_origin = origin
                physical_rejections: list[str] = []
                accepted_surface_clearance = math.inf
                parent_radius = sampled_axis_radius(axis, origin_arc, config)
                candidate_collar = candidate_branch_collar_radius(
                    parent_radius, axis.branch_generation + 1, config
                )
                focus_local_values = np.asarray([
                    float(water[0]), float(phosphorus[0]),
                    float(nitrogen[0]), float(potassium[0]),
                ], dtype=np.float64)
                branch_focus_seed = int(splitmix64(
                    parameters.seed
                    ^ ((site.site_id + 1) * 0x9E3779B97F4A7C15)
                    ^ ((step + 1) * 0xBF58476D1CE4E5B9)
                ))
                branch_focus_rng = np.random.default_rng(branch_focus_seed)
                branch_focus, branch_focus_score = draw_resource_focus(
                    branch_focus_rng, focus_local_values, demand_state, config
                )
                active_supply = demand_state.supply > CAPTURE_REPORTING_EPSILON
                environmental_support = (
                    float(np.mean(demand_state.supply[active_supply]))
                    if np.any(active_supply) else 0.0
                )
                emergence_sufficiency = float(np.clip(
                    0.55 * support_value + 0.45 * environmental_support,
                    0.0, 1.0,
                ))
                for candidate_azimuth in physical_site_azimuth_candidates(
                    azimuth, rng
                ):
                    surface_available, surface_clearance = cylindrical_surface_clearance(
                        axis,
                        origin_arc,
                        candidate_azimuth,
                        candidate_collar,
                        config,
                    )
                    if not surface_available:
                        physical_rejections.append("surface_clearance")
                        continue
                    emergence_angle = sample_branch_emergence_angle(
                        rng,
                        starvation=starvation_value,
                        phosphorus=float(phosphorus[0]),
                        nitrogen=float(nitrogen[0]),
                        potassium=float(potassium[0]),
                        resource_focus=branch_focus,
                        sufficiency=emergence_sufficiency,
                    )
                    branch, rejection_reason = axis_store.create_branch_axis(
                        axis,
                        origin_arc,
                        candidate_azimuth,
                        emergence_angle,
                        step,
                        parameters.thickness_increment,
                        rng,
                        resource_field,
                        raining,
                        demand_state,
                        starvation_signal=starvation_value,
                        site=site,
                        resource_focus=branch_focus,
                        resource_focus_score=branch_focus_score,
                        resource_focus_seed=branch_focus_seed,
                    )
                    if branch is not None:
                        accepted_surface_clearance = surface_clearance
                        break
                    if rejection_reason == "sample_cap":
                        rejected_sample_cap += 1
                        status = "sample_cap"
                        break
                    physical_rejections.append(
                        rejection_reason or "other_root_collision"
                    )
                axis_store.profile_physical_origin_search_sec += (
                    time.perf_counter() - physical_search_started
                )
                if status == "sample_cap":
                    break
                if branch is not None:
                    if math.isfinite(accepted_surface_clearance):
                        axis_store.accepted_origin_surface_clearances.append(
                            accepted_surface_clearance
                        )
                    axis_store.parent_radii_at_branch_origins.append(parent_radius)
                    successful += 1
                    if branch.branch_generation == 1:
                        accepted_first_order_laterals += 1
                    else:
                        accepted_higher_order_laterals += 1
                    branch_parent_records.append(
                        (branch.axis_id, step, "success")
                    )
                    successful_by_depth[
                        lateral_branch_depth_bucket(float(accepted_origin[2]), config)
                    ] += 1
                else:
                    final_rejection = final_physical_rejection(
                        physical_rejections
                    )
                    if final_rejection == "surface_clearance":
                        rejected_origin_surface_clearance += 1
                        if config.branch_retry_mode == "retry_open_sites":
                            site.physically_open = branch_site_has_surface_capacity(
                                axis, site, config
                            )
                            site.temporarily_surface_full = not site.physically_open
                            site.last_evaluated_parent_radius = parent_radius
                    elif final_rejection == "above_soil_surface":
                        rejected_above_soil_surface += 1
                    elif final_rejection == "parent_collision":
                        rejected_parent_collision += 1
                    else:
                        rejected_other_root_collision += 1
            if status == "sample_cap":
                break
        if status == "sample_cap":
            break
        axis_store.maximum_sample_points_in_any_step = max(
            axis_store.maximum_sample_points_in_any_step,
            axis_store.estimated_sample_count() - step_start_sample_count,
        )
        demand_state.end_step(config)
        steps_completed = step + 1
        snapshot_steps = {10, 25, 50, 100, 250, 500}
        if steps_completed in snapshot_steps or (
            steps_completed > 500
            and steps_completed & (steps_completed - 1) == 0
        ):
            directions = np.asarray(
                axis_store.accepted_extension_directions, dtype=np.float64
            )
            axis_store.time_series_snapshots.append({
                "step": int(steps_completed),
                "axis_count": int(len(axis_store.axes)),
                "sampled_points": int(axis_store.estimated_sample_count()),
                "effective_wetting_depth": resource_field.environment.effective_wetting_depth,
                "effective_nitrate_depth": resource_field.environment.effective_nitrate_depth,
                "effective_potassium_depth": resource_field.environment.effective_potassium_depth,
                "mean_direction_z": float(np.mean(directions[:, 2])) if directions.size else 0.0,
                "near_horizontal_fraction": float(np.mean(np.abs(directions[:, 2]) <= 0.30)) if directions.size else 0.0,
            })
        checkpoint_state: dict[str, object] | None = None
        checkpoint_due = bool(
            checkpoint_path is not None
            and checkpoint_interval_steps > 0
            and steps_completed % checkpoint_interval_steps == 0
        )
        progress_due = bool(
            progress_path is not None
            and max(1, int(progress_interval_steps)) > 0
            and steps_completed % max(1, int(progress_interval_steps)) == 0
        )
        if checkpoint_due or progress_due:
            elapsed_execution_time = time.perf_counter() - started_at
            if checkpoint_due:
                from root_hpc_storage import (
                    checkpoint_header,
                    save_checkpoint_atomic,
                )

                checkpoint_state = {
                    "axis_store": axis_store,
                    "resource_field": resource_field,
                    "demand_state": demand_state,
                    "rng_bit_generator_state": rng.bit_generator.state,
                    "rainy_iterations": rainy_iterations,
                    "branch_opportunities": branch_opportunities,
                    "threshold_sum": threshold_sum,
                    "probability_passes": probability_passes,
                    "probability_failures": probability_failures,
                    "rejected_origin_surface_clearance": (
                        rejected_origin_surface_clearance
                    ),
                    "rejected_above_soil_surface": rejected_above_soil_surface,
                    "rejected_parent_collision": rejected_parent_collision,
                    "rejected_other_root_collision": rejected_other_root_collision,
                    "rejected_axis_ceiling": rejected_axis_ceiling,
                    "rejected_sample_cap": rejected_sample_cap,
                    "accepted_first_order_laterals": (
                        accepted_first_order_laterals
                    ),
                    "accepted_higher_order_laterals": (
                        accepted_higher_order_laterals
                    ),
                    "stimulus_evaluated_probability_passes": (
                        stimulus_evaluated_probability_passes
                    ),
                    "primordium_stimulus_sum": primordium_stimulus_sum,
                    "initiation_uniform_sum": initiation_uniform_sum,
                    "initiation_uniform_min": initiation_uniform_min,
                    "initiation_uniform_max": initiation_uniform_max,
                    "dry_probability_passes": dry_probability_passes,
                    "rain_probability_passes": rain_probability_passes,
                    "successful": successful,
                    "successful_by_depth": successful_by_depth,
                    "steps_completed": steps_completed,
                    "branch_parent_records": branch_parent_records,
                    "elapsed_execution_time_sec": elapsed_execution_time,
                }
                save_checkpoint_atomic(
                    Path(checkpoint_path),
                    header=checkpoint_header(
                        simulator_path=Path(__file__).resolve(),
                        schema_version=SCHEMA_VERSION,
                        config=config,
                        seed=parameters.seed,
                        task_index=parameters.task_index,
                        completed_step=steps_completed,
                    ),
                    state=checkpoint_state,
                )
            if progress_due:
                from root_hpc_storage import write_progress_atomic

                try:
                    import psutil

                    resident_memory_bytes = int(
                        psutil.Process(os.getpid()).memory_info().rss
                    )
                except (ImportError, OSError):
                    resident_memory_bytes = 0
                write_progress_atomic(
                    Path(progress_path),
                    {
                        "status": "running",
                        "completed_steps": int(steps_completed),
                        "requested_steps": int(developmental_steps),
                        "axes": int(len(axis_store.axes)),
                        "branches": int(successful),
                        "sampled_points": int(
                            axis_store.estimated_sample_count()
                        ),
                        "runtime_sec": float(elapsed_execution_time),
                        "resident_memory_bytes": resident_memory_bytes,
                        "checkpoint_path": (
                            str(checkpoint_path)
                            if checkpoint_path is not None else None
                        ),
                    },
                )
        if (
            pause_after_checkpoint_step is not None
            and checkpoint_due
            and steps_completed >= int(pause_after_checkpoint_step)
        ):
            status = "checkpoint_pause"
            break

    failed_angle = rejected_origin_surface_clearance
    failed_inflation = rejected_axis_ceiling
    failed_spatial = (
        rejected_above_soil_surface
        + rejected_origin_surface_clearance
        + rejected_parent_collision
        + rejected_other_root_collision
    )
    rain_fraction = _safe_ratio(rainy_iterations, max(steps_completed, 1))
    whorl_events, axis_whorl_ids = detect_local_whorls(axis_store)
    axis_store.whorl_events = whorl_events
    axis_store.axis_whorl_ids = axis_whorl_ids
    store = axis_store_to_node_store(
        axis_store,
        parameters,
        config,
        resource_field,
        rain_fraction,
        steps_completed,
        branch_parent_records,
    )
    resource_shares = demand_state.shares()
    demand_weights = demand_state.weights(config)
    environment = resource_field.environment
    accepted_directions = np.asarray(
        axis_store.accepted_extension_directions, dtype=np.float64
    )
    lateral_mask = np.asarray(
        axis_store.accepted_extension_generations, dtype=np.int32
    ) > 0
    lateral_directions = (
        accepted_directions[lateral_mask]
        if accepted_directions.size and lateral_mask.size
        else np.empty((0, 3), dtype=np.float64)
    )
    emergence_values = np.asarray(
        axis_store.branch_emergence_angles_deg, dtype=np.float64
    )
    focus_counts = {
        name: sum(axis.resource_focus == name for axis in axis_store.axes)
        for name in RESOURCE_FOCI
    }
    focus_updates = sum(axis.resource_focus_updates for axis in axis_store.axes)
    max_upward_run = max(
        (axis.maximum_consecutive_upward_extensions for axis in axis_store.axes),
        default=0,
    )
    positions = store.position[:store.size]
    architecture_width = float(max(
        np.ptp(positions[:, 0]) if positions.size else 0.0,
        np.ptp(positions[:, 1]) if positions.size else 0.0,
    ))
    architecture_depth = float(
        config.soil_surface_z - np.min(positions[:, 2]) if positions.size else 0.0
    )
    score_means = {
        name: value / max(axis_store.direction_score_evaluations, 1)
        for name, value in axis_store.direction_score_component_sum.items()
    }
    store.axis_metadata.update({
        "target_architecture_size": int(sampled_point_cap),
        "sampled_point_safety_cap": int(sampled_point_cap),
        "target_axis_count": int(target_axis_count),
        "max_growth_iterations": int(developmental_steps),
        "developmental_steps_requested": int(developmental_steps),
        "developmental_steps_completed": int(steps_completed),
        "normal_developmental_completion": int(
            status == "developmental_steps_complete"
            and steps_completed == developmental_steps
        ),
        "stop_reason": status,
        "growth_target_reached": int(status == "developmental_steps_complete"),
        "resource_demand_feedback_enabled": int(demand_state.enabled),
        "final_resource_demand_weights": np.asarray(demand_weights, dtype=np.float64),
        "final_resource_capture_shares": np.asarray(resource_shares, dtype=np.float64),
        "resource_capture_balance_error": float(demand_state.balance_error()),
        "resource_environment_step": int(environment.current_step),
        "cumulative_rain_input": float(environment.cumulative_rain_input),
        "effective_wetting_depth": float(environment.effective_wetting_depth),
        "effective_nitrate_depth": float(environment.effective_nitrate_depth),
        "effective_potassium_depth": float(environment.effective_potassium_depth),
        "active_target_shares": demand_state.active_target_shares.copy(),
        "normalized_capture_shares": demand_state.normalized_capture_shares.copy(),
        "resource_deficiency": demand_state.deficiency.copy(),
        "resource_supply_gates": demand_state.supply.copy(),
        "resource_focus_counts": focus_counts,
        "resource_focus_updates": int(focus_updates),
        "resource_focus_assignments": dict(axis_store.resource_focus_assignments),
        "accepted_extensions_by_focus": dict(axis_store.accepted_extensions_by_focus),
        "mean_accepted_direction_x": float(np.mean(accepted_directions[:, 0])) if accepted_directions.size else 0.0,
        "mean_accepted_direction_y": float(np.mean(accepted_directions[:, 1])) if accepted_directions.size else 0.0,
        "mean_accepted_direction_z": float(np.mean(accepted_directions[:, 2])) if accepted_directions.size else 0.0,
        "median_accepted_direction_z": float(np.median(accepted_directions[:, 2])) if accepted_directions.size else 0.0,
        "fraction_accepted_direction_z_lt_minus_090": float(np.mean(accepted_directions[:, 2] < -0.90)) if accepted_directions.size else 0.0,
        "fraction_accepted_direction_z_lt_minus_070": float(np.mean(accepted_directions[:, 2] < -0.70)) if accepted_directions.size else 0.0,
        "fraction_accepted_strongly_upward": float(np.mean(accepted_directions[:, 2] > 0.40)) if accepted_directions.size else 0.0,
        "mean_lateral_emergence_angle": float(np.mean(emergence_values)) if emergence_values.size else 0.0,
        "median_lateral_emergence_angle": float(np.median(emergence_values)) if emergence_values.size else 0.0,
        "fraction_near_horizontal_lateral_segments": float(np.mean(np.abs(lateral_directions[:, 2]) <= 0.30)) if lateral_directions.size else 0.0,
        "fraction_downward_lateral_segments": float(np.mean(lateral_directions[:, 2] < -0.70)) if lateral_directions.size else 0.0,
        "fraction_mildly_upward_lateral_segments": float(np.mean((lateral_directions[:, 2] > 0.0) & (lateral_directions[:, 2] <= 0.40))) if lateral_directions.size else 0.0,
        "maximum_consecutive_upward_extensions": int(max_upward_run),
        "architecture_width": architecture_width,
        "architecture_depth": architecture_depth,
        "architecture_depth_width_ratio": _safe_ratio(architecture_depth, architecture_width),
        "direction_score_component_means_json": json.dumps(score_means, sort_keys=True),
        "direction_score_component_maxima_json": json.dumps(axis_store.direction_score_component_max, sort_keys=True),
        "resource_time_series_json": json.dumps(axis_store.time_series_snapshots, sort_keys=True),
        "global_starvation_signal": global_starvation,
        "starvation_signal_mean": _safe_ratio(
            axis_store.starvation_signal_sum, axis_store.starvation_signal_count
        ),
        "starvation_signal_at_branch_origins_mean": _safe_ratio(
            axis_store.branch_origin_starvation_sum,
            axis_store.branch_origin_starvation_count,
        ),
        "resource_support_gate_mean": _safe_ratio(
            axis_store.resource_support_gate_sum,
            axis_store.resource_support_gate_count,
        ),
        "stimulus_evaluated_probability_passes": int(
            stimulus_evaluated_probability_passes
        ),
        "mean_local_primordium_stimulus": _safe_ratio(
            primordium_stimulus_sum,
            stimulus_evaluated_probability_passes,
        ),
        "initiation_model_version": INITIATION_MODEL_VERSION,
        "initiation_random_stream_version": INITIATION_RANDOM_STREAM_VERSION,
        "initiation_probability_resource_independent": 1,
        "initiation_uniform_mean": _safe_ratio(
            initiation_uniform_sum, branch_opportunities
        ),
        "initiation_uniform_min": (
            float(initiation_uniform_min) if branch_opportunities else 0.0
        ),
        "initiation_uniform_max": (
            float(initiation_uniform_max) if branch_opportunities else 0.0
        ),
        "branch_retry_mode": config.branch_retry_mode,
        "probability_failures": int(probability_failures),
        "probability_pass_rate": _safe_ratio(
            probability_passes, branch_opportunities
        ),
        "rejected_origin_surface_clearance": int(
            rejected_origin_surface_clearance
        ),
        "rejected_above_soil_surface": int(rejected_above_soil_surface),
        "rejected_parent_collision": int(rejected_parent_collision),
        "rejected_other_root_collision": int(rejected_other_root_collision),
        "rejected_axis_ceiling": int(rejected_axis_ceiling),
        "rejected_sample_cap": int(rejected_sample_cap),
        "accepted_first_order_laterals": int(accepted_first_order_laterals),
        "accepted_higher_order_laterals": int(accepted_higher_order_laterals),
        "physical_rejection_count": int(
            rejected_origin_surface_clearance
            + rejected_above_soil_surface
            + rejected_parent_collision
            + rejected_other_root_collision
            + rejected_axis_ceiling
            + rejected_sample_cap
        ),
        "physical_rejection_rate": _safe_ratio(
            rejected_origin_surface_clearance
            + rejected_above_soil_surface
            + rejected_parent_collision
            + rejected_other_root_collision
            + rejected_axis_ceiling
            + rejected_sample_cap,
            probability_passes,
        ),
        "opportunity_accounting_error": int(
            branch_opportunities - probability_failures - probability_passes
        ),
        "probability_pass_accounting_error": int(
            probability_passes
            - successful
            - rejected_origin_surface_clearance
            - rejected_above_soil_surface
            - rejected_parent_collision
            - rejected_other_root_collision
            - rejected_axis_ceiling
            - rejected_sample_cap
        ),
        "active_tips_at_step_start_total": int(
            axis_store.active_tips_at_step_start_total
        ),
        "active_tip_count_observations": int(axis_store.active_tip_count_observations),
        "tip_extension_attempts": int(axis_store.tip_extension_attempts),
        "tip_extensions_accepted": int(axis_store.tip_extensions_accepted),
        "tip_extensions_collision_blocked": int(
            axis_store.tip_extensions_collision_blocked
        ),
        "tip_extensions_surface_blocked": int(
            axis_store.tip_extensions_surface_blocked
        ),
        "tip_extensions_sample_cap_blocked": int(
            axis_store.tip_extensions_sample_cap_blocked
        ),
        "tip_extensions_other_blocked": int(
            axis_store.tip_extensions_other_blocked
        ),
        "primary_tip_extension_attempts": int(
            axis_store.primary_tip_extension_attempts
        ),
        "primary_tip_extensions_accepted": int(
            axis_store.primary_tip_extensions_accepted
        ),
        "lateral_tip_extension_attempts": int(
            axis_store.lateral_tip_extension_attempts
        ),
        "lateral_tip_extensions_accepted": int(
            axis_store.lateral_tip_extensions_accepted
        ),
        "generation_1_extension_attempts": int(
            axis_store.generation_1_extension_attempts
        ),
        "generation_1_extensions_accepted": int(
            axis_store.generation_1_extensions_accepted
        ),
        "generation_2_extension_attempts": int(
            axis_store.generation_2_extension_attempts
        ),
        "generation_2_extensions_accepted": int(
            axis_store.generation_2_extensions_accepted
        ),
        "generation_3plus_extension_attempts": int(
            axis_store.generation_3plus_extension_attempts
        ),
        "generation_3plus_extensions_accepted": int(
            axis_store.generation_3plus_extensions_accepted
        ),
        "maximum_active_tip_count": int(axis_store.active_tip_count_max),
        "final_active_tip_count": int(len(axis_store.active_axis_ids())),
        "active_tip_attempt_accounting_error": int(
            axis_store.active_tips_at_step_start_total
            - axis_store.tip_extension_attempts
        ),
        "branch_sites_created": int(len(axis_store.branch_sites)),
        "branch_sites_currently_open": int(sum(
            site.physically_open and not site.closed_in_single_trial_mode
            for site in axis_store.branch_sites
        )),
        "branch_sites_closed_single_trial": int(sum(
            site.closed_in_single_trial_mode for site in axis_store.branch_sites
        )),
        "branch_sites_temporarily_surface_full": int(sum(
            site.temporarily_surface_full for site in axis_store.branch_sites
        )),
        "branch_sites_reopened_after_thickening": int(
            axis_store.branch_sites_reopened_after_thickening
        ),
        "branch_site_trials_total": int(sum(
            site.trial_count for site in axis_store.branch_sites
        )),
        "branch_site_first_trials": int(sum(
            site.trial_count > 0 for site in axis_store.branch_sites
        )),
        "branch_site_retry_trials": int(sum(
            max(site.trial_count - 1, 0) for site in axis_store.branch_sites
        )),
        "branch_site_probability_failures": int(probability_failures),
        "branch_site_probability_passes": int(probability_passes),
        "accepted_origin_surface_clearances": np.asarray(
            axis_store.accepted_origin_surface_clearances, dtype=np.float64
        ),
        "parent_radii_at_branch_origins": np.asarray(
            axis_store.parent_radii_at_branch_origins, dtype=np.float64
        ),
        "maximum_sample_points_in_any_step": int(
            axis_store.maximum_sample_points_in_any_step
        ),
        "growth_iterations_completed": int(steps_completed),
        "attempted_branches": int(branch_opportunities),
        "accepted_branches": int(successful),
        "collision_sample_checks": int(axis_store.collision_sample_checks),
        "curve_growth_attempts": int(axis_store.growth_attempts),
        "kd_tree_rebuilds": int(axis_store.tree_rebuild_count),
    })
    metrics = collect_metrics(
        parameters=parameters,
        config=config,
        store=store,
        status=status,
        steps_completed=steps_completed,
        rainy_iterations=rainy_iterations,
        branch_opportunities=branch_opportunities,
        threshold_sum=threshold_sum,
        probability_passes=probability_passes,
        dry_probability_passes=dry_probability_passes,
        rain_probability_passes=rain_probability_passes,
        successful=successful,
        successful_by_depth=successful_by_depth,
        failed_angle=failed_angle,
        failed_inflation=failed_inflation,
        failed_spatial=failed_spatial,
        started_at=started_at,
    )
    if progress_path is not None:
        from root_hpc_storage import write_progress_atomic

        try:
            import psutil

            resident_memory_bytes = int(
                psutil.Process(os.getpid()).memory_info().rss
            )
        except (ImportError, OSError):
            resident_memory_bytes = 0
        write_progress_atomic(
            Path(progress_path),
            {
                "status": status,
                "completed_steps": int(steps_completed),
                "requested_steps": int(developmental_steps),
                "axes": int(len(axis_store.axes)),
                "branches": int(successful),
                "sampled_points": int(axis_store.estimated_sample_count()),
                "runtime_sec": float(metrics["execution_time_sec"]),
                "resident_memory_bytes": resident_memory_bytes,
                "checkpoint_path": (
                    str(checkpoint_path)
                    if checkpoint_path is not None else None
                ),
            },
        )
    if return_store:
        return metrics, store
    return metrics


def compute_strahler_orders(store: NodeStore) -> np.ndarray:
    """Compute true Horton-Strahler order for every node in one reverse pass."""

    size = store.size
    max_child_order = np.zeros(size, dtype=np.int32)
    count_at_max = np.zeros(size, dtype=np.int32)
    order = np.ones(size, dtype=np.int32)

    for node_id in range(size - 1, -1, -1):
        child_max = int(max_child_order[node_id])
        if child_max:
            order[node_id] = child_max + (count_at_max[node_id] >= 2)
        parent = int(store.parent[node_id])
        if parent >= 0:
            current = int(order[node_id])
            if current > max_child_order[parent]:
                max_child_order[parent] = current
                count_at_max[parent] = 1
            elif current == max_child_order[parent]:
                count_at_max[parent] += 1
    return order


def compute_strahler_order(store: NodeStore) -> int:
    """Compatibility helper returning the global/root Horton-Strahler order."""

    return int(compute_strahler_orders(store)[0])


def compute_branch_generations(store: NodeStore) -> np.ndarray:
    """Return branch generation where smooth continuations keep parent generation."""

    generation = np.zeros(store.size, dtype=np.int32)
    for node_id in range(1, store.size):
        parent = int(store.parent[node_id])
        if parent < 0 or bool(store.is_anchor[node_id]):
            generation[node_id] = 0
        elif bool(store.is_axis_continuation[node_id]):
            generation[node_id] = generation[parent]
        elif bool(store.is_anchor[parent]):
            generation[node_id] = 1
        else:
            generation[node_id] = generation[parent] + 1
    return generation


BASE_RESULT_FIELDS = [
    "task_index",
    "sim_id",
    "seed",
    "status",
    "rain_probability",
    "branch_probability",
    "thickness_increment",
    "steps_requested",
    "steps_completed",
    "rainy_iterations",
    "branch_opportunities",
    "mean_effective_branch_probability",
    "total_nodes",
    "anchor_nodes",
    "lateral_nodes",
    "leaf_nodes",
    "max_children",
    "max_topological_depth",
    "strahler_order",
    "max_depth",
    "anchor_depth",
    "anchor_total_length",
    "min_z",
    "root_width_x",
    "root_width_y",
    "root_width_depth_ratio",
    "center_of_mass_z",
    "max_effective_radius",
    "total_P_captured",
    "total_N_captured",
    "branch_probability_passes",
    "dry_probability_passes",
    "rain_probability_passes",
    "successful_branches",
    "successful_branches_topsoil",
    "successful_branches_upper_subsoil",
    "successful_branches_nitrogen_layer",
    "successful_branches_deep_soil",
    "failed_angle_capacity",
    "failed_lineage_inflation",
    "failed_spatial_collision",
    "acceptance_rate",
    "execution_time_sec",
]

GLOBAL_ADDITIONAL_FIELDS = [
    "resource_model_version", "direction_model_version", "curve_model_version",
    "initiation_model_version", "initiation_random_stream_version",
    "initiation_probability_resource_independent",
    "emergent_morphology_class", "branch_retry_mode",
    "target_architecture_size", "target_axis_count", "max_growth_iterations",
    "growth_target_reached", "resource_demand_feedback_enabled",
    "developmental_steps_requested", "developmental_steps_completed",
    "developmental_fraction_completed", "normal_developmental_completion",
    "stop_reason", "sampled_point_safety_cap", "sampled_point_count",
    "sampled_point_cap_utilization", "sample_cap_reached",
    "remaining_sample_capacity", "sample_points_per_developmental_step",
    "maximum_sample_points_in_any_step",
    "final_resource_demand_water", "final_resource_demand_P",
    "final_resource_demand_N", "final_resource_demand_K",
    "final_resource_capture_share_water", "final_resource_capture_share_P",
    "final_resource_capture_share_N", "final_resource_capture_share_K",
    "resource_capture_balance_error",
    "resource_environment_step", "cumulative_rain_input",
    "effective_wetting_depth", "effective_nitrate_depth",
    "effective_potassium_depth",
    "water_active_target_share", "phosphorus_active_target_share",
    "nitrogen_active_target_share", "potassium_active_target_share",
    "water_normalized_capture_share", "phosphorus_normalized_capture_share",
    "nitrogen_normalized_capture_share", "potassium_normalized_capture_share",
    "water_deficiency", "phosphorus_deficiency", "nitrogen_deficiency",
    "potassium_deficiency", "water_demand_weight",
    "phosphorus_demand_weight", "nitrogen_demand_weight",
    "potassium_demand_weight", "water_focus_axis_count",
    "phosphorus_focus_axis_count", "nitrogen_focus_axis_count",
    "potassium_focus_axis_count", "balanced_focus_axis_count",
    "resource_focus_updates", "mean_lateral_emergence_angle",
    "water_focus_extensions_accepted", "phosphorus_focus_extensions_accepted",
    "nitrogen_focus_extensions_accepted", "potassium_focus_extensions_accepted",
    "balanced_focus_extensions_accepted",
    "mean_accepted_direction_x", "mean_accepted_direction_y",
    "mean_accepted_direction_z", "median_accepted_direction_z",
    "fraction_accepted_direction_z_lt_minus_090",
    "fraction_accepted_direction_z_lt_minus_070",
    "fraction_accepted_strongly_upward",
    "median_lateral_emergence_angle",
    "fraction_near_horizontal_lateral_segments",
    "fraction_downward_lateral_segments",
    "fraction_mildly_upward_lateral_segments",
    "maximum_consecutive_upward_extensions", "architecture_width",
    "architecture_depth", "architecture_depth_width_ratio",
    "direction_score_component_means_json",
    "direction_score_component_maxima_json", "resource_time_series_json",
    "global_starvation_signal", "starvation_signal_mean", "starvation_signal_at_branch_origins_mean",
    "resource_support_gate_mean", "width_depth_ratio",
    "low_resource_downward_response_score",
    "axis_count", "sampled_node_count",
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
    "whorl_azimuth_entropy", "fraction_laterals_in_whorls", "whorl_score",
    "stimulus_evaluated_probability_passes",
    "mean_local_primordium_stimulus",
    "initiation_uniform_mean", "initiation_uniform_min", "initiation_uniform_max",
    "probability_failures", "probability_pass_rate",
    "probability_pass_acceptance_rate",
    "rejected_origin_surface_clearance",
    "rejected_above_soil_surface", "rejected_parent_collision",
    "rejected_other_root_collision", "rejected_axis_ceiling",
    "rejected_sample_cap",
    "accepted_first_order_laterals", "accepted_higher_order_laterals",
    "physical_rejection_count", "physical_rejection_rate",
    "opportunity_accounting_error", "probability_pass_accounting_error",
    "active_tips_at_step_start_total", "active_tip_observations",
    "tip_extension_attempts", "tip_extensions_accepted",
    "tip_extensions_collision_blocked", "tip_extensions_surface_blocked",
    "tip_extensions_sample_cap_blocked", "tip_extensions_other_blocked",
    "fraction_active_tip_attempts_accepted",
    "primary_tip_extension_attempts", "primary_tip_extensions_accepted",
    "lateral_tip_extension_attempts", "lateral_tip_extensions_accepted",
    "generation_1_extension_attempts", "generation_1_extensions_accepted",
    "generation_2_extension_attempts", "generation_2_extensions_accepted",
    "generation_3plus_extension_attempts",
    "generation_3plus_extensions_accepted", "maximum_active_tip_count",
    "final_active_tip_count", "active_tip_attempt_accounting_error",
    "branch_sites_created", "branch_sites_currently_open",
    "branch_sites_closed_single_trial",
    "branch_sites_temporarily_surface_full",
    "branch_sites_reopened_after_thickening", "branch_site_trials_total",
    "branch_site_first_trials", "branch_site_retry_trials",
    "branch_site_probability_failures", "branch_site_probability_passes",
    "lateral_axis_diagnostics_json",
    "lateral_count_age_0_2", "lateral_count_age_3_5",
    "lateral_count_age_6_10", "lateral_count_age_11_25",
    "lateral_count_age_gt_25",
    "mean_lateral_length_age_0_2", "mean_lateral_length_age_3_5",
    "mean_lateral_length_age_6_10", "mean_lateral_length_age_11_25",
    "mean_lateral_length_age_gt_25",
    "accepted_extensions_age_0_2", "accepted_extensions_age_3_5",
    "accepted_extensions_age_6_10", "accepted_extensions_age_11_25",
    "accepted_extensions_age_gt_25",
    "collision_rate_age_0_2", "collision_rate_age_3_5",
    "collision_rate_age_6_10", "collision_rate_age_11_25",
    "collision_rate_age_gt_25",
    "proportion_laterals_initial_shoulder_only",
    "proportion_laterals_at_least_2_accepted_extensions",
    "proportion_laterals_at_least_5_accepted_extensions",
    "proportion_laterals_at_least_10_accepted_extensions",
    "extension_parent_collision_blocked",
    "extension_other_root_collision_blocked",
    "extension_surface_blocked", "extension_other_blocked",
    "mean_initial_radial_displacement",
    "mean_distance_from_parent_after_shoulder",
    "mean_lateral_direction_z_after_emergence",
    "fraction_laterals_curve_back_inside_parent_radius",
    "fraction_laterals_only_one_support_curve",
    "fraction_laterals_active_at_termination",
    "profile_retry_site_traversal_sec",
    "profile_branch_probability_trials_sec",
    "profile_physical_origin_search_sec",
    "profile_active_tip_extensions_sec",
    "profile_collision_queries_sec",
    "profile_resource_direction_candidates_sec",
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
    "branch_origin_candidate_evaluations",
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
    "growth_iterations_completed", "attempted_branches", "accepted_branches",
    "collision_sample_checks", "curve_growth_attempts", "kd_tree_rebuilds",
    "total_water_captured", "total_K_captured",
    "water_capture_per_total_length", "phosphorus_capture_per_total_length",
    "nitrogen_capture_per_total_length", "potassium_capture_per_total_length",
    "water_capture_per_surface_area", "phosphorus_capture_per_surface_area",
    "nitrogen_capture_per_surface_area", "potassium_capture_per_surface_area",
    "total_root_length", "total_surface_area", "total_root_volume",
    "specific_root_length", "max_root_path_length", "mean_root_path_length",
    "median_root_path_length", "max_horizontal_width", "max_width_x",
    "max_width_y", "z_range", "bounding_box_volume", "convex_hull_volume",
    "center_of_mass_x", "center_of_mass_y", "branch_points", "mean_children",
    "mean_topological_depth", "global_strahler_order",
    "number_of_strahler_orders_present", "branch_density_per_length",
    "tip_density_per_length", "lateral_to_anchor_ratio", "root_length_per_node",
    "mean_gravitropic_divergence_deg", "mean_pitch_angle_deg",
    "mean_absolute_vertical_component", "mean_lateral_component",
    "fraction_near_vertical_segments", "fraction_strongly_lateral_segments",
    "fraction_upward_segments", "fraction_near_horizontal_segments",
    "mean_branch_vertical_component", "mean_continuation_vertical_component",
    "mean_environmental_resource_signal", "mean_direction_resource_score",
    "mean_direction_resource_gate", "mean_gravitropism_score",
    "mean_lateral_exploration_score", "mean_lateral_suppression_score",
    "mean_turn_angle_deg", "p95_turn_angle_deg", "max_turn_angle_deg",
    "sharp_turn_count", "fraction_sharp_turns",
    "mean_tip_continuation_angle_deg", "p95_tip_continuation_angle_deg",
    "branch_emergence_angle_mean_deg", "branch_emergence_angle_p95_deg",
    "hard_fork_count", "branching_nodes_with_incoming",
    "fraction_hard_forks", "fraction_terminal_forks",
    "terminal_fork_count", "mean_min_child_continuation_angle_deg",
    "p95_min_child_continuation_angle_deg", "v_shape_score",
    "multi_lateral_branch_node_count", "fraction_multi_lateral_branch_nodes",
    "mean_lateral_children_per_branch_node",
    "mean_branch_incoming_alignment", "mean_branch_outward_alignment",
    "mean_branch_anchor_alignment", "mean_branch_center_alignment",
    "fraction_inward_lateral_branches", "fraction_outward_lateral_branches",
    "same_axis_direction_similarity_mean", "same_axis_direction_similarity_p95",
    "repeated_axis_direction_score",
    "mean_parent_relative_branch_emergence_angle_deg",
    "mean_tortuosity", "max_tortuosity",
    "path_efficiency_mean", "path_efficiency_min", "fraction_length_topsoil",
    "fraction_length_upper_subsoil", "fraction_length_nitrogen_layer",
    "fraction_length_deep_soil", "fraction_length_below_nitrogen_layer",
]

GENERATION_METRIC_NAMES = [
    "mean_vertical_component",
    "mean_absolute_vertical_component",
    "fraction_upward_segments",
    "fraction_near_horizontal_segments",
    "fraction_strongly_lateral_segments",
]

BRANCH_GENERATION_FIELDS = [
    f"generation_{generation}_{metric}"
    for generation in range(MAX_REPORTED_BRANCH_GENERATION + 1)
    for metric in GENERATION_METRIC_NAMES
] + [
    f"generation_gt_{MAX_REPORTED_BRANCH_GENERATION}_{metric}"
    for metric in GENERATION_METRIC_NAMES
]

STRAHLER_METRIC_NAMES = [
    "node_count", "segment_count", "leaf_count", "branch_point_count",
    "total_length", "mean_segment_length", "median_segment_length",
    "max_segment_length", "total_surface_area", "total_volume", "mean_radius",
    "max_radius", "mean_diameter", "max_diameter", "mean_abs_x_spread",
    "mean_abs_y_spread", "x_width", "y_width", "z_span", "bounding_box_volume",
    "center_of_mass_x", "center_of_mass_y", "center_of_mass_z",
    "mean_topological_depth", "max_topological_depth", "mean_children",
    "max_children", "branch_density_per_length", "tip_density_per_length",
    "fraction_total_length", "fraction_total_surface_area", "fraction_total_volume",
    "mean_vertical_component", "mean_absolute_vertical_component",
    "mean_pitch_angle_deg", "horizontal_angle_dispersion",
    "mean_gravitropic_divergence_deg", "mean_tortuosity", "path_efficiency_mean",
    "total_water_captured", "total_P_captured", "total_N_captured",
    "total_K_captured", "fraction_total_water_captured",
    "fraction_total_P_captured", "fraction_total_N_captured",
    "fraction_total_K_captured", "water_capture_per_length",
    "phosphorus_capture_per_length", "nitrogen_capture_per_length",
    "potassium_capture_per_length", "water_capture_per_surface_area",
    "phosphorus_capture_per_surface_area", "nitrogen_capture_per_surface_area",
    "potassium_capture_per_surface_area", "mean_water_availability",
    "mean_phosphorus_availability", "mean_nitrogen_availability",
    "mean_potassium_availability", "mean_water_capture_depth",
    "mean_phosphorus_capture_depth", "mean_nitrogen_capture_depth",
    "mean_potassium_capture_depth", "deepest_water_capture",
    "deepest_phosphorus_capture", "deepest_nitrogen_capture",
    "deepest_potassium_capture", "branch_opportunities", "probability_passes",
    "successful_branches", "failed_angle_capacity", "failed_lineage_inflation",
    "failed_spatial_collision", "acceptance_rate", "probability_pass_rate",
]


def _strahler_prefix(order: int | None) -> str:
    return (
        f"strahler_{order}_"
        if order is not None
        else f"strahler_gt_{MAX_REPORTED_STRAHLER_ORDER}_"
    )


RESULT_FIELDS = list(dict.fromkeys(
    BASE_RESULT_FIELDS
    + GLOBAL_ADDITIONAL_FIELDS
    + BRANCH_GENERATION_FIELDS
    + [
        _strahler_prefix(order) + metric
        for order in list(range(1, MAX_REPORTED_STRAHLER_ORDER + 1)) + [None]
        for metric in STRAHLER_METRIC_NAMES
    ]
))


def strahler_summary_rows(
    metrics: Mapping[str, int | float | str]
) -> list[dict[str, int | float | str]]:
    """Return compact rows suitable for an app table without coupling to Streamlit."""

    rows: list[dict[str, int | float | str]] = []
    for order in list(range(1, MAX_REPORTED_STRAHLER_ORDER + 1)) + [None]:
        prefix = _strahler_prefix(order)
        rows.append({
            "order": order if order is not None else f">{MAX_REPORTED_STRAHLER_ORDER}",
            "nodes": metrics[prefix + "node_count"],
            "segments": metrics[prefix + "segment_count"],
            "length": metrics[prefix + "total_length"],
            "water": metrics[prefix + "total_water_captured"],
            "phosphorus": metrics[prefix + "total_P_captured"],
            "nitrogen": metrics[prefix + "total_N_captured"],
            "potassium": metrics[prefix + "total_K_captured"],
        })
    return rows


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def classify_emergent_morphology(
    *,
    primary_axis_length: float,
    total_lateral_length: float,
    first_order_lateral_count: int,
    higher_order_lateral_count: int,
    whorl_event_count: int,
) -> str:
    """Return a diagnostic phenotype label that never feeds back into growth."""

    if whorl_event_count > 0:
        return "whorl-containing"
    primary_fraction = _safe_ratio(
        primary_axis_length,
        primary_axis_length + total_lateral_length,
    )
    if (
        primary_fraction >= 0.80
        and first_order_lateral_count <= 5
        and higher_order_lateral_count == 0
    ):
        return "taproot-dominant"
    total_laterals = first_order_lateral_count + higher_order_lateral_count
    if total_laterals <= 8 and higher_order_lateral_count <= 1:
        return "sparsely-branched"
    if higher_order_lateral_count >= 20 or total_laterals >= 60:
        return "highly-branched"
    return "distributed-branching"


def _segment_geometry(store: NodeStore) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centerline length, truncated-cone area/volume, and midpoint arrays."""

    size = store.size
    lengths = np.zeros(size, dtype=np.float64)
    areas = np.zeros(size, dtype=np.float64)
    volumes = np.zeros(size, dtype=np.float64)
    midpoints = store.position[:size].copy()
    if size <= 1:
        return lengths, areas, volumes, midpoints
    child = np.arange(1, size)
    parent = store.parent[1:size]
    delta = store.position[1:size] - store.position[parent]
    lengths[1:] = np.linalg.norm(delta, axis=1)
    parent_radius = store.radius[parent]
    child_radius = store.radius[1:size]
    slant = np.sqrt(lengths[1:] ** 2 + (parent_radius - child_radius) ** 2)
    areas[1:] = math.pi * (parent_radius + child_radius) * slant
    volumes[1:] = (
        math.pi * lengths[1:] / 3.0
        * (parent_radius ** 2 + parent_radius * child_radius + child_radius ** 2)
    )
    midpoints[1:] = 0.5 * (store.position[1:size] + store.position[parent])
    return lengths, areas, volumes, midpoints


def _tree_path_metrics(
    store: NodeStore, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = store.size
    topological_depth = np.zeros(size, dtype=np.int32)
    path_length = np.zeros(size, dtype=np.float64)
    for node_id in range(1, size):
        parent = int(store.parent[node_id])
        topological_depth[node_id] = topological_depth[parent] + 1
        path_length[node_id] = path_length[parent] + lengths[node_id]
    displacement = np.linalg.norm(store.position[:size] - store.position[0], axis=1)
    efficiency = np.ones(size, dtype=np.float64)
    np.divide(displacement, path_length, out=efficiency, where=path_length > 0.0)
    efficiency = np.clip(efficiency, 0.0, 1.0)
    tortuosity = np.ones(size, dtype=np.float64)
    np.divide(path_length, displacement, out=tortuosity, where=displacement > 0.0)
    return topological_depth, path_length, efficiency, tortuosity


def _turn_and_fork_diagnostics(
    store: NodeStore,
    config: SimulationConfig,
) -> dict[str, int | float]:
    """Quantify centerline smoothness and hard V-shaped branching artifacts."""

    size = store.size
    if size <= 1:
        return {
            "mean_turn_angle_deg": 0.0,
            "p95_turn_angle_deg": 0.0,
            "max_turn_angle_deg": 0.0,
            "sharp_turn_count": 0,
            "fraction_sharp_turns": 0.0,
            "mean_tip_continuation_angle_deg": 0.0,
            "p95_tip_continuation_angle_deg": 0.0,
            "branch_emergence_angle_mean_deg": 0.0,
            "branch_emergence_angle_p95_deg": 0.0,
            "hard_fork_count": 0,
            "branching_nodes_with_incoming": 0,
            "fraction_hard_forks": 0.0,
            "fraction_terminal_forks": 0.0,
            "terminal_fork_count": 0,
            "mean_min_child_continuation_angle_deg": 0.0,
            "p95_min_child_continuation_angle_deg": 0.0,
            "v_shape_score": 0.0,
            "multi_lateral_branch_node_count": 0,
            "fraction_multi_lateral_branch_nodes": 0.0,
            "mean_lateral_children_per_branch_node": 0.0,
            "mean_branch_incoming_alignment": 0.0,
            "mean_branch_outward_alignment": 0.0,
            "mean_branch_anchor_alignment": 0.0,
            "mean_branch_center_alignment": 0.0,
            "fraction_inward_lateral_branches": 0.0,
            "fraction_outward_lateral_branches": 0.0,
            "same_axis_direction_similarity_mean": 0.0,
            "same_axis_direction_similarity_p95": 0.0,
            "repeated_axis_direction_score": 0.0,
            "mean_parent_relative_branch_emergence_angle_deg": 0.0,
        }

    child_ids = np.arange(1, size)
    parents = store.parent[1:size]
    parent_directions = store.direction[:size][parents]
    child_directions = store.direction[1:size]
    dots = np.einsum("ij,ij->i", parent_directions, child_directions)
    turn_angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    sharp_turn_count = int(
        np.count_nonzero(turn_angles >= config.sharp_turn_threshold_degrees)
    )

    continuation_edge_mask = store.is_axis_continuation[1:size]
    continuation_angles = turn_angles[continuation_edge_mask]
    continuation_similarity = dots[continuation_edge_mask]
    lateral_branch_mask = (
        (~store.is_anchor[1:size]) & (~store.is_axis_continuation[1:size])
    )
    branch_angles = turn_angles[lateral_branch_mask]
    branch_incoming_alignment = dots[lateral_branch_mask]

    branch_child_ids = child_ids[lateral_branch_mask]
    if branch_child_ids.size:
        branch_parent_ids = store.parent[branch_child_ids]
        branch_parent_positions = store.position[:size][branch_parent_ids]
        branch_directions = store.direction[:size][branch_child_ids]
        branch_direction_xy = branch_directions[:, :2]
        branch_direction_norm = np.linalg.norm(branch_direction_xy, axis=1)
        branch_direction_unit = np.zeros_like(branch_direction_xy)
        np.divide(
            branch_direction_xy,
            branch_direction_norm[:, None],
            out=branch_direction_unit,
            where=branch_direction_norm[:, None] > 1e-12,
        )

        anchor_positions = store.position[:size][store.is_anchor[:size]]
        if anchor_positions.size:
            anchor_order = np.argsort(anchor_positions[:, 2])
            anchor_sorted = anchor_positions[anchor_order]
            anchor_z = anchor_sorted[:, 2]
            parent_z = branch_parent_positions[:, 2]
            insertion = np.searchsorted(anchor_z, parent_z)
            left = np.clip(insertion - 1, 0, anchor_sorted.shape[0] - 1)
            right = np.clip(insertion, 0, anchor_sorted.shape[0] - 1)
            left_distance = np.abs(parent_z - anchor_z[left])
            right_distance = np.abs(parent_z - anchor_z[right])
            nearest = np.where(right_distance < left_distance, right, left)
            nearest_anchor_xy = anchor_sorted[nearest, :2]
        else:
            nearest_anchor_xy = np.zeros((branch_child_ids.size, 2), dtype=np.float64)

        radial_xy = branch_parent_positions[:, :2] - nearest_anchor_xy
        radial_norm = np.linalg.norm(radial_xy, axis=1)
        radial_unit = np.zeros_like(radial_xy)
        np.divide(
            radial_xy,
            radial_norm[:, None],
            out=radial_unit,
            where=radial_norm[:, None] > 1e-12,
        )
        valid_radial = (radial_norm > 1e-12) & (branch_direction_norm > 1e-12)
        outward_alignment_values = np.einsum(
            "ij,ij->i", branch_direction_unit, radial_unit
        )
        outward_alignment = (
            outward_alignment_values[valid_radial]
            if np.any(valid_radial)
            else np.zeros(0, dtype=np.float64)
        )

        center_xy = np.mean(store.position[:size, :2], axis=0)
        center_vectors = center_xy - branch_parent_positions[:, :2]
        center_norm = np.linalg.norm(center_vectors, axis=1)
        center_unit = np.zeros_like(center_vectors)
        np.divide(
            center_vectors,
            center_norm[:, None],
            out=center_unit,
            where=center_norm[:, None] > 1e-12,
        )
        valid_center = (center_norm > 1e-12) & (branch_direction_norm > 1e-12)
        center_alignment_values = np.einsum(
            "ij,ij->i", branch_direction_unit, center_unit
        )
        center_alignment = (
            center_alignment_values[valid_center]
            if np.any(valid_center)
            else np.zeros(0, dtype=np.float64)
        )
    else:
        outward_alignment = np.zeros(0, dtype=np.float64)
        center_alignment = np.zeros(0, dtype=np.float64)

    min_child_continuation_angles: list[float] = []
    hard_forks = 0
    branching_with_incoming = 0
    lateral_child_counts: list[int] = []
    multi_lateral_nodes = 0
    for node_id in np.flatnonzero(
        (store.child_count[:size] >= 2) & (store.parent[:size] >= 0)
    ):
        true_laterals = true_lateral_child_count(store, int(node_id))
        if true_laterals > 0:
            lateral_child_counts.append(true_laterals)
            if true_laterals > 1:
                multi_lateral_nodes += 1
        child = int(store.first_child[int(node_id)])
        child_angles: list[float] = []
        has_axis_continuation = False
        while child >= 0:
            angle = vector_angle_degrees(
                store.direction[int(node_id)], store.direction[child]
            )
            child_angles.append(angle)
            if (
                angle <= config.hard_fork_continuation_angle_degrees
                or bool(store.is_axis_continuation[child])
                or bool(store.is_anchor[child])
            ):
                has_axis_continuation = True
            child = int(store.next_sibling[child])
        if not child_angles:
            continue
        branching_with_incoming += 1
        minimum_angle = float(min(child_angles))
        min_child_continuation_angles.append(minimum_angle)
        if not has_axis_continuation:
            hard_forks += 1

    min_angles = np.asarray(min_child_continuation_angles, dtype=np.float64)
    fraction_hard_forks = _safe_ratio(hard_forks, branching_with_incoming)
    lateral_child_counts_array = np.asarray(lateral_child_counts, dtype=np.float64)
    fraction_multi_lateral_nodes = _safe_ratio(
        multi_lateral_nodes, int(lateral_child_counts_array.size)
    )
    v_shape_score = max(fraction_hard_forks, fraction_multi_lateral_nodes)
    return {
        "mean_turn_angle_deg": float(np.mean(turn_angles)) if turn_angles.size else 0.0,
        "p95_turn_angle_deg": float(np.percentile(turn_angles, 95)) if turn_angles.size else 0.0,
        "max_turn_angle_deg": float(np.max(turn_angles)) if turn_angles.size else 0.0,
        "sharp_turn_count": sharp_turn_count,
        "fraction_sharp_turns": _safe_ratio(sharp_turn_count, int(turn_angles.size)),
        "mean_tip_continuation_angle_deg": float(np.mean(continuation_angles)) if continuation_angles.size else 0.0,
        "p95_tip_continuation_angle_deg": float(np.percentile(continuation_angles, 95)) if continuation_angles.size else 0.0,
        "branch_emergence_angle_mean_deg": float(np.mean(branch_angles)) if branch_angles.size else 0.0,
        "branch_emergence_angle_p95_deg": float(np.percentile(branch_angles, 95)) if branch_angles.size else 0.0,
        "hard_fork_count": hard_forks,
        "branching_nodes_with_incoming": branching_with_incoming,
        "fraction_hard_forks": fraction_hard_forks,
        "fraction_terminal_forks": fraction_hard_forks,
        "terminal_fork_count": hard_forks,
        "mean_min_child_continuation_angle_deg": float(np.mean(min_angles)) if min_angles.size else 0.0,
        "p95_min_child_continuation_angle_deg": float(np.percentile(min_angles, 95)) if min_angles.size else 0.0,
        "v_shape_score": v_shape_score,
        "multi_lateral_branch_node_count": multi_lateral_nodes,
        "fraction_multi_lateral_branch_nodes": fraction_multi_lateral_nodes,
        "mean_lateral_children_per_branch_node": float(np.mean(lateral_child_counts_array)) if lateral_child_counts_array.size else 0.0,
        "mean_branch_incoming_alignment": float(np.mean(branch_incoming_alignment)) if branch_incoming_alignment.size else 0.0,
        "mean_branch_outward_alignment": float(np.mean(outward_alignment)) if outward_alignment.size else 0.0,
        "mean_branch_anchor_alignment": float(np.mean(-outward_alignment)) if outward_alignment.size else 0.0,
        "mean_branch_center_alignment": float(np.mean(center_alignment)) if center_alignment.size else 0.0,
        "fraction_inward_lateral_branches": float(np.mean(outward_alignment < -0.15)) if outward_alignment.size else 0.0,
        "fraction_outward_lateral_branches": float(np.mean(outward_alignment > 0.15)) if outward_alignment.size else 0.0,
        "same_axis_direction_similarity_mean": float(np.mean(continuation_similarity)) if continuation_similarity.size else 0.0,
        "same_axis_direction_similarity_p95": float(np.percentile(continuation_similarity, 95)) if continuation_similarity.size else 0.0,
        "repeated_axis_direction_score": float(np.mean(continuation_angles <= 3.0)) if continuation_angles.size else 0.0,
        "mean_parent_relative_branch_emergence_angle_deg": float(np.mean(branch_angles)) if branch_angles.size else 0.0,
    }


def _curve_axis_diagnostics(store: NodeStore) -> dict[str, int | float | str]:
    """Return schema-v12 diagnostics from the continuous-axis metadata."""

    metadata = getattr(store, "axis_metadata", {}) or {}
    axis_lengths = np.asarray(
        metadata.get("axis_arc_lengths", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    )
    curvatures = np.asarray(
        metadata.get("curvatures", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    )
    bend_angles = np.asarray(
        metadata.get("tip_bend_angles_deg", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    )
    emergence_angles = np.asarray(
        metadata.get("branch_emergence_angles_deg", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    )
    azimuths = np.asarray(
        metadata.get("branch_azimuth_angles", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    )
    spacings = np.asarray(
        metadata.get("branch_origin_spacings", np.zeros(0, dtype=np.float64)),
        dtype=np.float64,
    )
    anchor_points = np.asarray(
        metadata.get("anchor_points", np.empty((0, 3), dtype=np.float64)),
        dtype=np.float64,
    )
    if anchor_points.ndim != 2 or anchor_points.shape[1] != 3:
        anchor_points = np.empty((0, 3), dtype=np.float64)

    if anchor_points.shape[0] > 1:
        anchor_deltas = anchor_points[1:] - anchor_points[:-1]
        anchor_segments = np.linalg.norm(anchor_deltas, axis=1)
        anchor_path = float(np.sum(anchor_segments))
        anchor_displacement = float(np.linalg.norm(anchor_points[-1] - anchor_points[0]))
        anchor_depth = max(0.0, float(-np.min(anchor_points[:, 2])))
        anchor_lateral = float(np.linalg.norm(anchor_points[-1, :2] - anchor_points[0, :2]))
        anchor_dirs = np.divide(
            anchor_deltas,
            anchor_segments[:, None],
            out=np.zeros_like(anchor_deltas),
            where=anchor_segments[:, None] > 1e-12,
        )
        anchor_mean_vertical_component = float(
            np.mean(np.clip(-anchor_dirs[:, 2], -1.0, 1.0))
        ) if anchor_dirs.size else 0.0
        anchor_lateral_drift_ratio = _safe_ratio(anchor_lateral, anchor_depth)
        anchor_tortuosity = _safe_ratio(anchor_path, anchor_displacement)
    else:
        anchor_mean_vertical_component = 0.0
        anchor_lateral_drift_ratio = 0.0
        anchor_tortuosity = 0.0

    if emergence_angles.size:
        p10, p50, p90 = np.percentile(emergence_angles, [10, 50, 90])
        angle_entropy = _linear_entropy(emergence_angles, 15.0, 120.0, bins=12)
        fraction_15_40 = float(np.mean((emergence_angles >= 15.0) & (emergence_angles < 40.0)))
        fraction_40_70 = float(np.mean((emergence_angles >= 40.0) & (emergence_angles < 70.0)))
        fraction_70_95 = float(np.mean((emergence_angles >= 70.0) & (emergence_angles <= 95.0)))
        fraction_gt_95 = float(np.mean(emergence_angles > 95.0))
        dominant_angle_band = max(fraction_15_40, fraction_40_70, fraction_70_95, fraction_gt_95)
        hard_v_junction_score = (
            float(
                np.clip((dominant_angle_band - 0.55) / 0.45, 0.0, 1.0)
                * (1.0 - 0.65 * angle_entropy)
            )
            if emergence_angles.size >= 5
            else 0.0
        )
    else:
        p10 = p50 = p90 = 0.0
        angle_entropy = 0.0
        fraction_15_40 = fraction_40_70 = fraction_70_95 = fraction_gt_95 = 0.0
        hard_v_junction_score = 0.0

    straight_stick_artifact_score = (
        float(np.mean(bend_angles <= 1.0)) if bend_angles.size else 0.0
    )
    surface_z = float(metadata.get("soil_surface_z", 0.0))
    surface_tolerance = float(metadata.get("max_above_surface_tolerance", 0.05))
    surface_limit = surface_z + surface_tolerance
    positions = store.position[:store.size]
    above_mask = positions[:, 2] > surface_limit
    above_surface_node_count = int(np.count_nonzero(above_mask))
    max_above_surface_z = float(np.max(positions[:, 2])) if store.size else surface_z
    if store.size > 1:
        child = np.arange(1, store.size)
        parent = store.parent[1:store.size]
        segment_lengths = np.linalg.norm(positions[1:store.size] - positions[parent], axis=1)
        segment_above = (
            (positions[1:store.size, 2] > surface_limit)
            | (positions[parent, 2] > surface_limit)
        )
        above_surface_length = float(np.sum(segment_lengths[segment_above]))
        total_segment_length = float(np.sum(segment_lengths))
    else:
        above_surface_length = 0.0
        total_segment_length = 0.0
    return {
        "curve_model_version": str(metadata.get("curve_model_version", CURVE_MODEL_VERSION)),
        "axis_count": int(metadata.get("axis_count", 1)),
        "sampled_node_count": int(store.size),
        "mean_axis_arc_length": float(np.mean(axis_lengths)) if axis_lengths.size else 0.0,
        "max_axis_arc_length": float(np.max(axis_lengths)) if axis_lengths.size else 0.0,
        "mean_curvature": float(np.mean(curvatures)) if curvatures.size else 0.0,
        "p95_curvature": float(np.percentile(curvatures, 95)) if curvatures.size else 0.0,
        "max_curvature": float(np.max(curvatures)) if curvatures.size else 0.0,
        "mean_tip_bend_angle_deg": float(np.mean(bend_angles)) if bend_angles.size else 0.0,
        "p95_tip_bend_angle_deg": float(np.percentile(bend_angles, 95)) if bend_angles.size else 0.0,
        "anchor_lateral_drift_ratio": anchor_lateral_drift_ratio,
        "anchor_mean_vertical_component": anchor_mean_vertical_component,
        "anchor_tortuosity": anchor_tortuosity,
        "branch_origin_spacing_mean": float(np.mean(spacings)) if spacings.size else 0.0,
        "branch_origin_spacing_min": float(np.min(spacings)) if spacings.size else 0.0,
        "branch_emergence_angle_mean_deg": float(np.mean(emergence_angles)) if emergence_angles.size else 0.0,
        "branch_emergence_angle_p10_deg": float(p10),
        "branch_emergence_angle_p50_deg": float(p50),
        "branch_emergence_angle_p90_deg": float(p90),
        "branch_emergence_angle_p95_deg": float(np.percentile(emergence_angles, 95)) if emergence_angles.size else 0.0,
        "branch_emergence_angle_entropy": float(angle_entropy),
        "branch_azimuth_entropy": _entropy_from_angles(azimuths),
        "fraction_branches_15_40_deg": fraction_15_40,
        "fraction_branches_40_70_deg": fraction_40_70,
        "fraction_branches_70_95_deg": fraction_70_95,
        "fraction_branches_gt_95_deg": fraction_gt_95,
        "hard_v_junction_score": hard_v_junction_score,
        "straight_stick_artifact_score": straight_stick_artifact_score,
        "above_surface_node_count": above_surface_node_count,
        "fraction_above_surface_nodes": _safe_ratio(above_surface_node_count, store.size),
        "max_above_surface_z": max_above_surface_z,
        "above_surface_length": above_surface_length,
        "fraction_above_surface_length": _safe_ratio(
            above_surface_length, total_segment_length
        ),
        "mean_curve_collision_samples_per_growth": float(
            metadata.get("mean_curve_collision_samples_per_growth", 0.0)
        ),
    }


def _per_strahler_metrics(
    store: NodeStore,
    orders: np.ndarray,
    lengths: np.ndarray,
    areas: np.ndarray,
    volumes: np.ndarray,
    midpoints: np.ndarray,
    topological_depth: np.ndarray,
    efficiency: np.ndarray,
    tortuosity: np.ndarray,
) -> dict[str, int | float]:
    output: dict[str, int | float] = {}
    size = store.size
    node_ids = np.arange(size)
    total_length = float(np.sum(lengths))
    total_area = float(np.sum(areas))
    total_volume = float(np.sum(volumes))
    resource_arrays = (
        store.water_captured[:size], store.phosphorus_captured[:size],
        store.nitrogen_captured[:size], store.potassium_captured[:size],
    )
    resource_totals = tuple(float(np.sum(values)) for values in resource_arrays)
    buckets: list[tuple[int | None, np.ndarray]] = [
        (order, orders == order) for order in range(1, MAX_REPORTED_STRAHLER_ORDER + 1)
    ]
    buckets.append((None, orders > MAX_REPORTED_STRAHLER_ORDER))
    for order, node_mask in buckets:
        prefix = _strahler_prefix(order)
        edge_mask = node_mask & (node_ids > 0)
        node_count = int(np.count_nonzero(node_mask))
        segment_count = int(np.count_nonzero(edge_mask))
        group_lengths = lengths[edge_mask]
        group_areas = areas[edge_mask]
        group_volumes = volumes[edge_mask]
        group_positions = store.position[:size][node_mask]
        group_radii = store.radius[:size][edge_mask]
        child_counts = store.child_count[:size][node_mask]
        group_length = float(np.sum(group_lengths))
        group_area = float(np.sum(group_areas))
        group_volume = float(np.sum(group_volumes))
        leaves = int(np.count_nonzero(node_mask & (store.child_count[:size] == 0)))
        branches = int(np.count_nonzero(node_mask & (store.child_count[:size] > 1)))
        if node_count:
            center = np.mean(group_positions, axis=0)
            x_width = float(np.ptp(group_positions[:, 0]))
            y_width = float(np.ptp(group_positions[:, 1]))
            z_span = float(np.ptp(group_positions[:, 2]))
            mean_abs_x = float(np.mean(np.abs(group_positions[:, 0] - center[0])))
            mean_abs_y = float(np.mean(np.abs(group_positions[:, 1] - center[1])))
        else:
            center = np.zeros(3, dtype=np.float64)
            x_width = y_width = z_span = mean_abs_x = mean_abs_y = 0.0
        directions = store.direction[:size][edge_mask]
        if segment_count:
            vertical = directions[:, 2]
            pitch = np.degrees(np.arcsin(np.clip(-vertical, -1.0, 1.0)))
            gravity = np.degrees(np.arccos(np.clip(-vertical, -1.0, 1.0)))
            azimuth = np.arctan2(directions[:, 1], directions[:, 0])
            circular_r = math.hypot(float(np.mean(np.cos(azimuth))), float(np.mean(np.sin(azimuth))))
            horizontal_dispersion = 1.0 - circular_r
        else:
            vertical = pitch = gravity = np.zeros(0)
            horizontal_dispersion = 0.0
        observations = float(np.sum(store.resource_observations[:size][node_mask]))
        captures = [float(np.sum(values[node_mask])) for values in resource_arrays]
        availability_sums = [
            float(np.sum(values[:size][node_mask]))
            for values in (
                store.water_availability_sum, store.phosphorus_availability_sum,
                store.nitrogen_availability_sum, store.potassium_availability_sum,
            )
        ]
        depth_sums = [
            float(np.sum(values[:size][node_mask]))
            for values in (
                store.water_capture_depth_sum, store.phosphorus_capture_depth_sum,
                store.nitrogen_capture_depth_sum, store.potassium_capture_depth_sum,
            )
        ]
        deepest_arrays = (
            store.deepest_water_capture, store.deepest_phosphorus_capture,
            store.deepest_nitrogen_capture, store.deepest_potassium_capture,
        )
        deepest = [
            float(np.max(values[:size][node_mask])) if node_count else 0.0
            for values in deepest_arrays
        ]
        opportunities = int(np.sum(store.branch_opportunities[:size][node_mask]))
        passes = int(np.sum(store.probability_passes[:size][node_mask]))
        successes = int(np.sum(store.successful_branches[:size][node_mask]))
        failures = [
            int(np.sum(values[:size][node_mask]))
            for values in (store.failed_angle, store.failed_inflation, store.failed_spatial)
        ]
        attempts = successes + sum(failures)
        values: dict[str, int | float] = {
            "node_count": node_count, "segment_count": segment_count,
            "leaf_count": leaves, "branch_point_count": branches,
            "total_length": group_length,
            "mean_segment_length": float(np.mean(group_lengths)) if segment_count else 0.0,
            "median_segment_length": float(np.median(group_lengths)) if segment_count else 0.0,
            "max_segment_length": float(np.max(group_lengths)) if segment_count else 0.0,
            "total_surface_area": group_area, "total_volume": group_volume,
            "mean_radius": float(np.mean(group_radii)) if segment_count else 0.0,
            "max_radius": float(np.max(group_radii)) if segment_count else 0.0,
            "mean_diameter": 2.0 * float(np.mean(group_radii)) if segment_count else 0.0,
            "max_diameter": 2.0 * float(np.max(group_radii)) if segment_count else 0.0,
            "mean_abs_x_spread": mean_abs_x, "mean_abs_y_spread": mean_abs_y,
            "x_width": x_width, "y_width": y_width, "z_span": z_span,
            "bounding_box_volume": x_width * y_width * z_span,
            "center_of_mass_x": float(center[0]), "center_of_mass_y": float(center[1]),
            "center_of_mass_z": float(center[2]),
            "mean_topological_depth": float(np.mean(topological_depth[node_mask])) if node_count else 0.0,
            "max_topological_depth": int(np.max(topological_depth[node_mask])) if node_count else 0,
            "mean_children": float(np.mean(child_counts)) if node_count else 0.0,
            "max_children": int(np.max(child_counts)) if node_count else 0,
            "branch_density_per_length": _safe_ratio(branches, group_length),
            "tip_density_per_length": _safe_ratio(leaves, group_length),
            "fraction_total_length": _safe_ratio(group_length, total_length),
            "fraction_total_surface_area": _safe_ratio(group_area, total_area),
            "fraction_total_volume": _safe_ratio(group_volume, total_volume),
            "mean_vertical_component": float(np.mean(vertical)) if segment_count else 0.0,
            "mean_absolute_vertical_component": float(np.mean(np.abs(vertical))) if segment_count else 0.0,
            "mean_pitch_angle_deg": float(np.mean(pitch)) if segment_count else 0.0,
            "horizontal_angle_dispersion": horizontal_dispersion,
            "mean_gravitropic_divergence_deg": float(np.mean(gravity)) if segment_count else 0.0,
            "mean_tortuosity": float(np.mean(tortuosity[edge_mask])) if segment_count else 0.0,
            "path_efficiency_mean": float(np.mean(efficiency[edge_mask])) if segment_count else 0.0,
            "total_water_captured": captures[0], "total_P_captured": captures[1],
            "total_N_captured": captures[2], "total_K_captured": captures[3],
            "fraction_total_water_captured": _safe_ratio(captures[0], resource_totals[0]),
            "fraction_total_P_captured": _safe_ratio(captures[1], resource_totals[1]),
            "fraction_total_N_captured": _safe_ratio(captures[2], resource_totals[2]),
            "fraction_total_K_captured": _safe_ratio(captures[3], resource_totals[3]),
            "water_capture_per_length": _safe_ratio(captures[0], group_length),
            "phosphorus_capture_per_length": _safe_ratio(captures[1], group_length),
            "nitrogen_capture_per_length": _safe_ratio(captures[2], group_length),
            "potassium_capture_per_length": _safe_ratio(captures[3], group_length),
            "water_capture_per_surface_area": _safe_ratio(captures[0], group_area),
            "phosphorus_capture_per_surface_area": _safe_ratio(captures[1], group_area),
            "nitrogen_capture_per_surface_area": _safe_ratio(captures[2], group_area),
            "potassium_capture_per_surface_area": _safe_ratio(captures[3], group_area),
            "mean_water_availability": _safe_ratio(availability_sums[0], observations),
            "mean_phosphorus_availability": _safe_ratio(availability_sums[1], observations),
            "mean_nitrogen_availability": _safe_ratio(availability_sums[2], observations),
            "mean_potassium_availability": _safe_ratio(availability_sums[3], observations),
            "mean_water_capture_depth": _safe_ratio(depth_sums[0], captures[0]),
            "mean_phosphorus_capture_depth": _safe_ratio(depth_sums[1], captures[1]),
            "mean_nitrogen_capture_depth": _safe_ratio(depth_sums[2], captures[2]),
            "mean_potassium_capture_depth": _safe_ratio(depth_sums[3], captures[3]),
            "deepest_water_capture": deepest[0], "deepest_phosphorus_capture": deepest[1],
            "deepest_nitrogen_capture": deepest[2], "deepest_potassium_capture": deepest[3],
            "branch_opportunities": opportunities, "probability_passes": passes,
            "successful_branches": successes, "failed_angle_capacity": failures[0],
            "failed_lineage_inflation": failures[1], "failed_spatial_collision": failures[2],
            "acceptance_rate": _safe_ratio(successes, attempts),
            "probability_pass_rate": _safe_ratio(passes, opportunities),
        }
        for metric in STRAHLER_METRIC_NAMES:
            output[prefix + metric] = values[metric]
    return output


def _per_generation_direction_metrics(
    store: NodeStore,
    branch_generations: np.ndarray,
) -> dict[str, int | float]:
    """Summarize vertical/lateral orientation by branch generation."""

    output: dict[str, int | float] = {}
    if store.size <= 1:
        edge_generations = np.zeros(0, dtype=np.int32)
        vertical = np.zeros(0, dtype=np.float64)
    else:
        child_ids = np.arange(1, store.size)
        edge_generations = branch_generations[child_ids]
        vertical = store.direction[child_ids, 2]
    absolute_vertical = np.abs(vertical)
    for generation in range(MAX_REPORTED_BRANCH_GENERATION + 1):
        mask = edge_generations == generation
        prefix = f"generation_{generation}_"
        if np.any(mask):
            values = vertical[mask]
            abs_values = absolute_vertical[mask]
            output[prefix + "mean_vertical_component"] = float(np.mean(values))
            output[prefix + "mean_absolute_vertical_component"] = float(np.mean(abs_values))
            output[prefix + "fraction_upward_segments"] = float(np.mean(values > 0.0))
            output[prefix + "fraction_near_horizontal_segments"] = float(np.mean(abs_values <= 0.25))
            output[prefix + "fraction_strongly_lateral_segments"] = float(np.mean(abs_values <= 0.35))
        else:
            for metric in GENERATION_METRIC_NAMES:
                output[prefix + metric] = 0.0
    overflow_prefix = f"generation_gt_{MAX_REPORTED_BRANCH_GENERATION}_"
    overflow_mask = edge_generations > MAX_REPORTED_BRANCH_GENERATION
    if np.any(overflow_mask):
        values = vertical[overflow_mask]
        abs_values = absolute_vertical[overflow_mask]
        output[overflow_prefix + "mean_vertical_component"] = float(np.mean(values))
        output[overflow_prefix + "mean_absolute_vertical_component"] = float(np.mean(abs_values))
        output[overflow_prefix + "fraction_upward_segments"] = float(np.mean(values > 0.0))
        output[overflow_prefix + "fraction_near_horizontal_segments"] = float(np.mean(abs_values <= 0.25))
        output[overflow_prefix + "fraction_strongly_lateral_segments"] = float(np.mean(abs_values <= 0.35))
    else:
        for metric in GENERATION_METRIC_NAMES:
            output[overflow_prefix + metric] = 0.0
    return output


def collect_metrics(
    *,
    parameters: SimulationParameters,
    config: SimulationConfig,
    store: NodeStore,
    status: str,
    steps_completed: int,
    rainy_iterations: int,
    branch_opportunities: int,
    threshold_sum: float,
    probability_passes: int,
    dry_probability_passes: int,
    rain_probability_passes: int,
    successful: int,
    successful_by_depth: Sequence[int],
    failed_angle: int,
    failed_inflation: int,
    failed_spatial: int,
    started_at: float,
) -> dict[str, int | float | str]:
    positions = store.position[: store.size]
    radii = store.radius[: store.size]
    child_counts = store.child_count[: store.size]
    lateral_nodes = int(np.count_nonzero(~store.is_anchor[: store.size]))
    lengths, areas, volumes, midpoints = _segment_geometry(store)
    orders = compute_strahler_orders(store)
    branch_generations = compute_branch_generations(store)
    topological_depth, path_length, efficiency, tortuosity = _tree_path_metrics(store, lengths)
    smoothness_diagnostics = _turn_and_fork_diagnostics(store, config)
    curve_axis_diagnostics = _curve_axis_diagnostics(store)
    axis_metadata = getattr(store, "axis_metadata", {}) or {}
    attempts = successful + failed_angle + failed_inflation + failed_spatial
    anchor_mask = store.is_anchor[: store.size]
    anchor_positions = positions[anchor_mask]
    anchor_depth = float(-np.min(anchor_positions[:, 2]))
    anchor_total_length = float(np.sum(store.edge_length[: store.size][anchor_mask]))
    horizontal_width = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])))
    max_depth = max(0.0, float(-np.min(positions[:, 2])))
    total_length = float(np.sum(lengths))
    total_area = float(np.sum(areas))
    total_volume = float(np.sum(volumes))
    total_resources = [
        float(np.sum(values[:store.size]))
        for values in (
            store.water_captured, store.phosphorus_captured,
            store.nitrogen_captured, store.potassium_captured,
        )
    ]
    if total_volume > 0.0:
        center = np.average(midpoints[1:], axis=0, weights=volumes[1:])
    else:
        center = np.mean(positions, axis=0)
    width_x = float(np.ptp(positions[:, 0]))
    width_y = float(np.ptp(positions[:, 1]))
    z_range = float(np.ptp(positions[:, 2]))
    branch_points = int(np.count_nonzero(child_counts > 1))
    edge_mask = np.arange(store.size) > 0
    directions = store.direction[:store.size][edge_mask]
    vertical = directions[:, 2] if directions.size else np.zeros(0)
    absolute_vertical = np.abs(vertical)
    lateral_component = (
        np.linalg.norm(directions[:, :2], axis=1) if directions.size else np.zeros(0)
    )
    pitch = np.degrees(np.arcsin(np.clip(-vertical, -1.0, 1.0)))
    gravity = np.degrees(np.arccos(np.clip(-vertical, -1.0, 1.0)))
    midpoint_z = midpoints[:, 2]
    branch_edge_mask = edge_mask & (~anchor_mask)
    lateral_branch_edge_mask = (
        edge_mask
        & (~anchor_mask)
        & (~store.is_axis_continuation[: store.size])
    )
    continuation_edge_mask = (
        edge_mask
        & store.is_axis_continuation[: store.size]
    )
    resource_observations = float(np.sum(store.resource_observations[: store.size]))
    if resource_observations > 0.0:
        mean_availability = np.array(
            [
                float(np.sum(store.water_availability_sum[: store.size])),
                float(np.sum(store.phosphorus_availability_sum[: store.size])),
                float(np.sum(store.nitrogen_availability_sum[: store.size])),
                float(np.sum(store.potassium_availability_sum[: store.size])),
            ],
            dtype=np.float64,
        ) / resource_observations
        resource_weights = np.array(
            [
                WATER_AVAILABILITY_DIRECTION_WEIGHT,
                PHOSPHORUS_AVAILABILITY_DIRECTION_WEIGHT,
                NITROGEN_AVAILABILITY_DIRECTION_WEIGHT,
                POTASSIUM_AVAILABILITY_DIRECTION_WEIGHT,
            ],
            dtype=np.float64,
        )
        total_resource_weight = float(np.sum(resource_weights))
        if total_resource_weight > 0.0:
            mean_environmental_resource_signal = float(
                np.clip(
                    np.dot(mean_availability, resource_weights)
                    / total_resource_weight,
                    0.0,
                    1.0,
                )
            )
        else:
            mean_environmental_resource_signal = 0.0
    else:
        mean_environmental_resource_signal = 0.0
    convex_hull_volume = math.nan
    if config.compute_convex_hull and store.size >= 4:
        try:
            convex_hull_volume = float(ConvexHull(positions).volume)
        except QhullError:
            convex_hull_volume = 0.0
    metrics: dict[str, int | float | str] = {
        "task_index": parameters.task_index,
        "sim_id": parameters.sim_id,
        "seed": parameters.seed,
        "status": status,
        "rain_probability": parameters.rain_probability,
        "branch_probability": parameters.branch_probability,
        "thickness_increment": parameters.thickness_increment,
        "steps_requested": config.steps,
        "steps_completed": steps_completed,
        "rainy_iterations": rainy_iterations,
        "branch_opportunities": branch_opportunities,
        "mean_effective_branch_probability": (
            threshold_sum / branch_opportunities if branch_opportunities else 0.0
        ),
        "total_nodes": store.size,
        "anchor_nodes": int(np.count_nonzero(anchor_mask)),
        "lateral_nodes": lateral_nodes,
        "leaf_nodes": int(np.count_nonzero(child_counts == 0)),
        "max_children": int(np.max(child_counts)),
        "max_topological_depth": int(np.max(topological_depth)),
        "strahler_order": int(orders[0]),
        "max_depth": max_depth,
        "anchor_depth": anchor_depth,
        "anchor_total_length": anchor_total_length,
        "min_z": float(np.min(positions[:, 2])),
        "root_width_x": width_x,
        "root_width_y": width_y,
        "root_width_depth_ratio": (
            horizontal_width / max_depth if max_depth > 0.0 else 0.0
        ),
        "center_of_mass_z": float(center[2]),
        "max_effective_radius": float(np.max(radii)),
        "total_P_captured": total_resources[1],
        "total_N_captured": total_resources[2],
        "branch_probability_passes": probability_passes,
        "dry_probability_passes": dry_probability_passes,
        "rain_probability_passes": rain_probability_passes,
        "successful_branches": successful,
        "successful_branches_topsoil": int(successful_by_depth[0]),
        "successful_branches_upper_subsoil": int(successful_by_depth[1]),
        "successful_branches_nitrogen_layer": int(successful_by_depth[2]),
        "successful_branches_deep_soil": int(successful_by_depth[3]),
        "failed_angle_capacity": failed_angle,
        "failed_lineage_inflation": failed_inflation,
        "failed_spatial_collision": failed_spatial,
        "acceptance_rate": successful / attempts if attempts else 0.0,
        "execution_time_sec": time.perf_counter() - started_at,
    }
    demand_weights = np.asarray(
        axis_metadata.get("final_resource_demand_weights", np.ones(4, dtype=np.float64)),
        dtype=np.float64,
    )
    capture_shares = np.asarray(
        axis_metadata.get("final_resource_capture_shares", np.zeros(4, dtype=np.float64)),
        dtype=np.float64,
    )
    active_targets = np.asarray(
        axis_metadata.get("active_target_shares", np.zeros(4, dtype=np.float64)),
        dtype=np.float64,
    )
    deficiencies = np.asarray(
        axis_metadata.get("resource_deficiency", np.zeros(4, dtype=np.float64)),
        dtype=np.float64,
    )
    focus_counts = dict(axis_metadata.get("resource_focus_counts", {}))
    focus_extensions = dict(axis_metadata.get("accepted_extensions_by_focus", {}))
    starvation_signal_mean = float(axis_metadata.get("starvation_signal_mean", 0.0))
    mean_branch_vertical = (
        float(np.mean(store.direction[:store.size, 2][lateral_branch_edge_mask]))
        if np.any(lateral_branch_edge_mask)
        else 0.0
    )
    upward_fraction = float(np.mean(vertical > 0.0)) if vertical.size else 0.0
    near_horizontal_fraction = (
        float(np.mean(absolute_vertical <= 0.25)) if vertical.size else 0.0
    )
    near_vertical_fraction = (
        float(np.mean(absolute_vertical >= 0.85)) if vertical.size else 0.0
    )
    low_resource_downward_response_score = starvation_signal_mean * float(
        np.clip(
            0.40 * max(0.0, -mean_branch_vertical)
            + 0.25 * near_vertical_fraction
            + 0.20 * (1.0 - upward_fraction)
            + 0.15 * (1.0 - near_horizontal_fraction),
            0.0,
            1.0,
        )
    )
    axis_lengths = np.asarray(
        axis_metadata.get("axis_arc_lengths", np.asarray([anchor_total_length])),
        dtype=np.float64,
    )
    axis_generations = np.asarray(
        axis_metadata.get("axis_generations", np.zeros(axis_lengths.size, dtype=np.int32)),
        dtype=np.int32,
    )
    if axis_generations.size != axis_lengths.size:
        axis_generations = np.resize(axis_generations, axis_lengths.size)
    primary_axis_length = float(np.sum(axis_lengths[axis_generations == 0]))
    total_lateral_length = float(np.sum(axis_lengths[axis_generations > 0]))
    lateral_axis_lengths = axis_lengths[axis_generations > 0]
    mean_lateral_axis_length = (
        float(np.mean(lateral_axis_lengths)) if lateral_axis_lengths.size else 0.0
    )
    max_lateral_axis_length = (
        float(np.max(lateral_axis_lengths)) if lateral_axis_lengths.size else 0.0
    )
    first_order_axis_lengths = axis_lengths[axis_generations == 1]
    mean_first_order_lateral_length = (
        float(np.mean(first_order_axis_lengths))
        if first_order_axis_lengths.size else 0.0
    )
    median_first_order_lateral_length = (
        float(np.median(first_order_axis_lengths))
        if first_order_axis_lengths.size else 0.0
    )
    max_first_order_lateral_length = (
        float(np.max(first_order_axis_lengths))
        if first_order_axis_lengths.size else 0.0
    )
    first_order_lateral_count = int(np.count_nonzero(axis_generations == 1))
    higher_order_lateral_count = int(np.count_nonzero(axis_generations > 1))
    primary_node_mask = branch_generations == 0
    lateral_node_mask = branch_generations > 0
    primary_axis_max_radius = (
        float(np.max(radii[primary_node_mask])) if np.any(primary_node_mask) else 0.0
    )
    anchor_node_ids = np.flatnonzero(store.is_anchor[:store.size])
    primary_axis_basal_radius = (
        float(radii[anchor_node_ids[0]]) if anchor_node_ids.size else 0.0
    )
    primary_axis_distal_tip_radius = (
        float(radii[anchor_node_ids[-1]]) if anchor_node_ids.size else 0.0
    )
    primary_axis_basal_to_tip_radius_ratio = _safe_ratio(
        primary_axis_basal_radius,
        primary_axis_distal_tip_radius,
    )
    primary_edge_ids = np.flatnonzero(
        primary_node_mask & (np.arange(store.size) > 0)
    )
    if primary_edge_ids.size:
        primary_parent_ids = store.parent[primary_edge_ids]
        primary_axis_radius_integral = float(np.sum(
            0.5
            * (radii[primary_edge_ids] + radii[primary_parent_ids])
            * lengths[primary_edge_ids]
        ))
    else:
        primary_axis_radius_integral = 0.0
    primary_axis_mean_radius = _safe_ratio(
        primary_axis_radius_integral,
        primary_axis_length,
    )
    axis_structural_allocations = np.asarray(
        axis_metadata.get(
            "axis_structural_allocations",
            np.zeros(axis_lengths.size, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    if axis_structural_allocations.size != axis_generations.size:
        axis_structural_allocations = np.resize(
            axis_structural_allocations,
            axis_generations.size,
        )
    primary_structural_allocation = float(np.sum(
        axis_structural_allocations[axis_generations == 0]
    ))
    lateral_structural_allocation = float(np.sum(
        axis_structural_allocations[axis_generations > 0]
    ))
    primary_fraction_total_structural_allocation = _safe_ratio(
        primary_structural_allocation,
        primary_structural_allocation + lateral_structural_allocation,
    )
    lateral_fraction_total_structural_allocation = _safe_ratio(
        lateral_structural_allocation,
        primary_structural_allocation + lateral_structural_allocation,
    )
    first_order_origin_depths = np.asarray(
        axis_metadata.get("first_order_origin_depths", np.empty(0)),
        dtype=np.float64,
    )
    first_order_origin_arc_fractions = np.asarray(
        axis_metadata.get("first_order_origin_arc_fractions", np.empty(0)),
        dtype=np.float64,
    )
    def origin_quantile(values: np.ndarray, percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values.size else 0.0
    mean_lateral_radius = (
        float(np.mean(radii[lateral_node_mask])) if np.any(lateral_node_mask) else 0.0
    )
    lateral_to_primary_length_ratio = _safe_ratio(
        total_lateral_length, primary_axis_length
    )
    lateral_to_primary_radius_ratio = _safe_ratio(
        mean_lateral_radius, primary_axis_max_radius
    )
    low_bp_taproot_score = float(np.clip(
        0.35 / (1.0 + lateral_to_primary_length_ratio)
        + 0.30 / (1.0 + 4.0 * lateral_to_primary_radius_ratio)
        + 0.20 * math.exp(-first_order_lateral_count / 4.0)
        + 0.15 * math.exp(-higher_order_lateral_count),
        0.0,
        1.0,
    ))
    branch_count_at_bp_001 = (
        int(first_order_lateral_count + higher_order_lateral_count)
        if abs(parameters.branch_probability - 0.01) <= 5e-10
        else 0
    )
    whorl_events = list(axis_metadata.get("whorl_events", []))
    whorl_counts = np.asarray(
        [int(event.get("branch_count", 0)) for event in whorl_events],
        dtype=np.int32,
    )
    whorl_depths = np.sort(np.asarray(
        [float(event.get("depth", 0.0)) for event in whorl_events],
        dtype=np.float64,
    ))
    whorl_azimuths = np.asarray(
        [
            float(angle)
            for event in whorl_events
            for angle in event.get("azimuths", [])
        ],
        dtype=np.float64,
    )
    axis_whorl_ids = np.asarray(
        axis_metadata.get("axis_whorl_ids", np.full(axis_lengths.size, -1)),
        dtype=np.int32,
    )
    if axis_whorl_ids.size != axis_generations.size:
        axis_whorl_ids = np.resize(axis_whorl_ids, axis_generations.size)
    whorled_lateral_count = int(np.count_nonzero(
        (axis_generations == 1) & (axis_whorl_ids >= 0)
    ))
    # Whorl diagnostics group first-order origins on the primary axis.
    total_lateral_axis_count = first_order_lateral_count
    fraction_laterals_in_whorls = _safe_ratio(
        whorled_lateral_count, total_lateral_axis_count
    )
    whorl_azimuth_entropy = _entropy_from_angles(whorl_azimuths)
    mean_branches_per_whorl = (
        float(np.mean(whorl_counts)) if whorl_counts.size else 0.0
    )
    whorl_score = float(np.clip(
        fraction_laterals_in_whorls
        * min(
            _safe_ratio(
                mean_branches_per_whorl,
                POSTHOC_WHORL_MIN_BRANCHES,
            ),
            1.0,
        )
        * whorl_azimuth_entropy,
        0.0,
        1.0,
    ))
    emergent_morphology_class = classify_emergent_morphology(
        primary_axis_length=primary_axis_length,
        total_lateral_length=total_lateral_length,
        first_order_lateral_count=first_order_lateral_count,
        higher_order_lateral_count=higher_order_lateral_count,
        whorl_event_count=int(whorl_counts.size),
    )
    developmental_steps_requested = int(
        axis_metadata.get("developmental_steps_requested", config.steps)
    )
    developmental_steps_completed = int(
        axis_metadata.get("developmental_steps_completed", steps_completed)
    )
    sampled_point_safety_cap = int(
        axis_metadata.get("sampled_point_safety_cap", effective_sampled_point_cap(config))
    )
    sampled_point_count = int(store.size)

    axis_material_arcs = list(axis_metadata.get("axis_material_arcs", []))
    axis_radius_profiles = list(axis_metadata.get("axis_radii", []))
    axis_branch_origins = list(axis_metadata.get("axis_branch_origins", []))

    def radius_profile_statistics(
        axis_index: int,
    ) -> tuple[np.ndarray, float, float]:
        """Return seven radius samples, taper fraction, and off-junction bump."""

        if axis_index >= len(axis_material_arcs) or axis_index >= len(axis_radius_profiles):
            return np.zeros(7, dtype=np.float64), 0.0, 0.0
        arc = np.asarray(axis_material_arcs[axis_index], dtype=np.float64)
        profile = np.asarray(axis_radius_profiles[axis_index], dtype=np.float64)
        if not arc.size or not profile.size or arc.size != profile.size:
            return np.zeros(7, dtype=np.float64), 0.0, 0.0
        fractions = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
        samples = np.interp(fractions * max(float(arc[-1]), 0.0), arc, profile)
        changes = np.diff(profile)
        tolerance = max(1e-12, 1e-8 * float(np.max(profile)))
        monotonic_fraction = (
            float(np.mean(changes <= tolerance)) if changes.size else 1.0
        )
        away = np.ones(changes.size, dtype=bool)
        if axis_index < len(axis_branch_origins) and changes.size:
            origins = np.asarray(axis_branch_origins[axis_index], dtype=np.float64)
            if origins.size:
                mid_arc = 0.5 * (arc[:-1] + arc[1:])
                junction_window = max(0.10, 0.30 * config.segment_length)
                away &= np.min(
                    np.abs(mid_arc[:, None] - origins[None, :]), axis=1
                ) > junction_window
        positive = changes[away] if changes.size else np.empty(0)
        max_increase = (
            float(max(0.0, np.max(positive))) if positive.size else 0.0
        )
        return samples, monotonic_fraction, max_increase

    primary_radius_samples, primary_taper_fraction, primary_radius_bump = (
        radius_profile_statistics(0)
    )
    first_order_taper_ratios: list[float] = []
    first_order_taper_fractions: list[float] = []
    for axis_index in np.flatnonzero(axis_generations == 1):
        if axis_index >= axis_lengths.size or axis_lengths[axis_index] < 1.0:
            continue
        samples, taper_fraction, _bump = radius_profile_statistics(int(axis_index))
        first_order_taper_ratios.append(_safe_ratio(samples[0], samples[-1]))
        first_order_taper_fractions.append(taper_fraction)

    branch_sites = list(axis_metadata.get("branch_sites", []))
    occupied_sites = [
        site for site in branch_sites if int(site.get("accepted_branch_count", 0)) > 0
    ]
    multi_sites = [
        site for site in occupied_sites if int(site.get("accepted_branch_count", 0)) > 1
    ]
    same_site_separations: list[float] = []
    for site in multi_sites:
        azimuths_at_site = [float(value) for value in site.get("occupied_azimuths", [])]
        for left in range(len(azimuths_at_site)):
            for right in range(left + 1, len(azimuths_at_site)):
                delta = abs(
                    (azimuths_at_site[left] - azimuths_at_site[right] + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
                same_site_separations.append(math.degrees(delta))
    accepted_axial_separations: list[float] = []
    parent_ids = np.asarray(axis_metadata.get("axis_parent_ids", []), dtype=np.int32)
    parent_arcs = np.asarray(
        axis_metadata.get("axis_parent_arc_lengths", []), dtype=np.float64
    )
    for parent_id in np.unique(parent_ids[parent_ids >= 0]):
        values = np.sort(parent_arcs[parent_ids == parent_id])
        if values.size > 1:
            accepted_axial_separations.extend(np.diff(values).tolist())
    clearances = np.asarray(
        axis_metadata.get("accepted_origin_surface_clearances", []),
        dtype=np.float64,
    )
    parent_origin_radii = np.asarray(
        axis_metadata.get("parent_radii_at_branch_origins", []),
        dtype=np.float64,
    )
    multi_parent_radii = np.asarray(
        [float(site.get("last_evaluated_parent_radius", 0.0)) for site in multi_sites],
        dtype=np.float64,
    )
    exact_parent_radii = np.asarray(
        axis_metadata.get("axis_parent_local_radii", []), dtype=np.float64
    )
    exact_basal_radii = np.asarray(
        axis_metadata.get("axis_basal_radii", []), dtype=np.float64
    )
    if exact_parent_radii.size == exact_basal_radii.size and exact_parent_radii.size > 1:
        child_parent_radius_ratios = np.divide(
            exact_basal_radii[1:],
            exact_parent_radii[1:],
            out=np.zeros_like(exact_basal_radii[1:]),
            where=exact_parent_radii[1:] > 0.0,
        )
    else:
        child_parent_radius_ratios = np.empty(0, dtype=np.float64)
    lateral_rows = list(axis_metadata.get("lateral_axis_diagnostics", []))
    age_group_metrics: dict[str, int | float] = {}
    for suffix, low, high in LATERAL_AGE_GROUPS:
        selected = [
            row for row in lateral_rows
            if int(row.get("biological_age_steps", 0)) >= low
            and (high is None or int(row.get("biological_age_steps", 0)) <= high)
        ]
        age_group_metrics[f"lateral_count_age_{suffix}"] = len(selected)
        age_group_metrics[f"mean_lateral_length_age_{suffix}"] = (
            float(np.mean([float(row.get("current_arc_length", 0.0)) for row in selected]))
            if selected else 0.0
        )
        age_group_metrics[f"accepted_extensions_age_{suffix}"] = int(sum(
            int(row.get("accepted_extensions", 0)) for row in selected
        ))
        attempts_in_group = sum(
            int(row.get("extension_attempts", 0)) for row in selected
        )
        collisions_in_group = sum(
            int(row.get("collision_blocked_extensions", 0)) for row in selected
        )
        age_group_metrics[f"collision_rate_age_{suffix}"] = _safe_ratio(
            collisions_in_group, attempts_in_group
        )
    lateral_count = len(lateral_rows)
    extension_parent_collision_blocked = int(sum(
        int(row.get("parent_collision_blocked_extensions", 0))
        for row in lateral_rows
    ))
    extension_other_root_collision_blocked = int(sum(
        int(row.get("other_root_collision_blocked_extensions", 0))
        for row in lateral_rows
    ))
    active_tip_observations = int(
        axis_metadata.get("active_tip_count_observations", 0)
    )
    metrics.update({
        "resource_model_version": RESOURCE_MODEL_VERSION,
        "direction_model_version": DIRECTION_MODEL_VERSION,
        "curve_model_version": CURVE_MODEL_VERSION,
        "initiation_model_version": INITIATION_MODEL_VERSION,
        "initiation_random_stream_version": INITIATION_RANDOM_STREAM_VERSION,
        "initiation_probability_resource_independent": 1,
        "emergent_morphology_class": emergent_morphology_class,
        "branch_retry_mode": config.branch_retry_mode,
        "target_architecture_size": int(axis_metadata.get("target_architecture_size", effective_target_architecture_size(config))),
        "target_axis_count": int(axis_metadata.get("target_axis_count", 0)),
        "max_growth_iterations": int(axis_metadata.get("max_growth_iterations", config.max_growth_iterations)),
        "growth_target_reached": int(axis_metadata.get("growth_target_reached", 0)),
        "developmental_steps_requested": developmental_steps_requested,
        "developmental_steps_completed": developmental_steps_completed,
        "developmental_fraction_completed": _safe_ratio(
            developmental_steps_completed, developmental_steps_requested
        ),
        "normal_developmental_completion": int(
            axis_metadata.get("normal_developmental_completion", 0)
        ),
        "stop_reason": str(axis_metadata.get("stop_reason", status)),
        "sampled_point_safety_cap": sampled_point_safety_cap,
        "sampled_point_count": sampled_point_count,
        "sampled_point_cap_utilization": _safe_ratio(
            sampled_point_count, sampled_point_safety_cap
        ),
        "sample_cap_reached": int(
            str(axis_metadata.get("stop_reason", status)) == "sample_cap"
        ),
        "remaining_sample_capacity": max(
            sampled_point_safety_cap - sampled_point_count, 0
        ),
        "sample_points_per_developmental_step": _safe_ratio(
            sampled_point_count, developmental_steps_completed
        ),
        "maximum_sample_points_in_any_step": int(
            axis_metadata.get("maximum_sample_points_in_any_step", 0)
        ),
        "resource_demand_feedback_enabled": int(axis_metadata.get("resource_demand_feedback_enabled", int(config.enable_resource_demand_feedback))),
        "final_resource_demand_water": float(demand_weights[0]) if demand_weights.size >= 4 else 1.0,
        "final_resource_demand_P": float(demand_weights[1]) if demand_weights.size >= 4 else 1.0,
        "final_resource_demand_N": float(demand_weights[2]) if demand_weights.size >= 4 else 1.0,
        "final_resource_demand_K": float(demand_weights[3]) if demand_weights.size >= 4 else 1.0,
        "final_resource_capture_share_water": float(capture_shares[0]) if capture_shares.size >= 4 else 0.0,
        "final_resource_capture_share_P": float(capture_shares[1]) if capture_shares.size >= 4 else 0.0,
        "final_resource_capture_share_N": float(capture_shares[2]) if capture_shares.size >= 4 else 0.0,
        "final_resource_capture_share_K": float(capture_shares[3]) if capture_shares.size >= 4 else 0.0,
        "resource_capture_balance_error": float(axis_metadata.get("resource_capture_balance_error", 0.0)),
        "resource_environment_step": int(axis_metadata.get("resource_environment_step", -1)),
        "cumulative_rain_input": float(axis_metadata.get("cumulative_rain_input", 0.0)),
        "effective_wetting_depth": float(axis_metadata.get("effective_wetting_depth", 0.0)),
        "effective_nitrate_depth": float(axis_metadata.get("effective_nitrate_depth", 0.0)),
        "effective_potassium_depth": float(axis_metadata.get("effective_potassium_depth", 0.0)),
        "water_active_target_share": float(active_targets[0]) if active_targets.size >= 4 else 0.0,
        "phosphorus_active_target_share": float(active_targets[1]) if active_targets.size >= 4 else 0.0,
        "nitrogen_active_target_share": float(active_targets[2]) if active_targets.size >= 4 else 0.0,
        "potassium_active_target_share": float(active_targets[3]) if active_targets.size >= 4 else 0.0,
        "water_normalized_capture_share": float(capture_shares[0]) if capture_shares.size >= 4 else 0.0,
        "phosphorus_normalized_capture_share": float(capture_shares[1]) if capture_shares.size >= 4 else 0.0,
        "nitrogen_normalized_capture_share": float(capture_shares[2]) if capture_shares.size >= 4 else 0.0,
        "potassium_normalized_capture_share": float(capture_shares[3]) if capture_shares.size >= 4 else 0.0,
        "water_deficiency": float(deficiencies[0]) if deficiencies.size >= 4 else 0.0,
        "phosphorus_deficiency": float(deficiencies[1]) if deficiencies.size >= 4 else 0.0,
        "nitrogen_deficiency": float(deficiencies[2]) if deficiencies.size >= 4 else 0.0,
        "potassium_deficiency": float(deficiencies[3]) if deficiencies.size >= 4 else 0.0,
        "water_demand_weight": float(demand_weights[0]) if demand_weights.size >= 4 else 0.0,
        "phosphorus_demand_weight": float(demand_weights[1]) if demand_weights.size >= 4 else 0.0,
        "nitrogen_demand_weight": float(demand_weights[2]) if demand_weights.size >= 4 else 0.0,
        "potassium_demand_weight": float(demand_weights[3]) if demand_weights.size >= 4 else 0.0,
        "water_focus_axis_count": int(focus_counts.get("water", 0)),
        "phosphorus_focus_axis_count": int(focus_counts.get("phosphorus", 0)),
        "nitrogen_focus_axis_count": int(focus_counts.get("nitrogen", 0)),
        "potassium_focus_axis_count": int(focus_counts.get("potassium", 0)),
        "balanced_focus_axis_count": int(focus_counts.get("balanced", 0)),
        "resource_focus_updates": int(axis_metadata.get("resource_focus_updates", 0)),
        "water_focus_extensions_accepted": int(focus_extensions.get("water", 0)),
        "phosphorus_focus_extensions_accepted": int(focus_extensions.get("phosphorus", 0)),
        "nitrogen_focus_extensions_accepted": int(focus_extensions.get("nitrogen", 0)),
        "potassium_focus_extensions_accepted": int(focus_extensions.get("potassium", 0)),
        "balanced_focus_extensions_accepted": int(focus_extensions.get("balanced", 0)),
        "mean_accepted_direction_x": float(axis_metadata.get("mean_accepted_direction_x", 0.0)),
        "mean_accepted_direction_y": float(axis_metadata.get("mean_accepted_direction_y", 0.0)),
        "mean_accepted_direction_z": float(axis_metadata.get("mean_accepted_direction_z", 0.0)),
        "median_accepted_direction_z": float(axis_metadata.get("median_accepted_direction_z", 0.0)),
        "fraction_accepted_direction_z_lt_minus_090": float(axis_metadata.get("fraction_accepted_direction_z_lt_minus_090", 0.0)),
        "fraction_accepted_direction_z_lt_minus_070": float(axis_metadata.get("fraction_accepted_direction_z_lt_minus_070", 0.0)),
        "fraction_accepted_strongly_upward": float(axis_metadata.get("fraction_accepted_strongly_upward", 0.0)),
        "mean_lateral_emergence_angle": float(axis_metadata.get("mean_lateral_emergence_angle", 0.0)),
        "median_lateral_emergence_angle": float(axis_metadata.get("median_lateral_emergence_angle", 0.0)),
        "fraction_near_horizontal_lateral_segments": float(axis_metadata.get("fraction_near_horizontal_lateral_segments", 0.0)),
        "fraction_downward_lateral_segments": float(axis_metadata.get("fraction_downward_lateral_segments", 0.0)),
        "fraction_mildly_upward_lateral_segments": float(axis_metadata.get("fraction_mildly_upward_lateral_segments", 0.0)),
        "maximum_consecutive_upward_extensions": int(axis_metadata.get("maximum_consecutive_upward_extensions", 0)),
        "architecture_width": float(axis_metadata.get("architecture_width", 0.0)),
        "architecture_depth": float(axis_metadata.get("architecture_depth", 0.0)),
        "architecture_depth_width_ratio": float(axis_metadata.get("architecture_depth_width_ratio", 0.0)),
        "direction_score_component_means_json": str(axis_metadata.get("direction_score_component_means_json", "{}")),
        "direction_score_component_maxima_json": str(axis_metadata.get("direction_score_component_maxima_json", "{}")),
        "resource_time_series_json": str(axis_metadata.get("resource_time_series_json", "[]")),
        "global_starvation_signal": float(axis_metadata.get("global_starvation_signal", 0.0)),
        "starvation_signal_mean": starvation_signal_mean,
        "starvation_signal_at_branch_origins_mean": float(axis_metadata.get("starvation_signal_at_branch_origins_mean", 0.0)),
        "resource_support_gate_mean": float(axis_metadata.get("resource_support_gate_mean", 0.0)),
        "growth_iterations_completed": int(axis_metadata.get("growth_iterations_completed", steps_completed)),
        "attempted_branches": int(axis_metadata.get("attempted_branches", branch_opportunities)),
        "accepted_branches": int(axis_metadata.get("accepted_branches", successful)),
        "collision_sample_checks": int(axis_metadata.get("collision_sample_checks", 0)),
        "curve_growth_attempts": int(axis_metadata.get("curve_growth_attempts", 0)),
        "kd_tree_rebuilds": int(axis_metadata.get("kd_tree_rebuilds", 0)),
        "width_depth_ratio": horizontal_width / max_depth if max_depth > 0.0 else 0.0,
        "low_resource_downward_response_score": low_resource_downward_response_score,
        "primary_axis_length": primary_axis_length,
        "total_lateral_length": total_lateral_length,
        "mean_lateral_axis_length": mean_lateral_axis_length,
        "max_lateral_axis_length": max_lateral_axis_length,
        "mean_first_order_lateral_length": mean_first_order_lateral_length,
        "median_first_order_lateral_length": median_first_order_lateral_length,
        "max_first_order_lateral_length": max_first_order_lateral_length,
        "lateral_to_primary_length_ratio": lateral_to_primary_length_ratio,
        "primary_axis_basal_radius": primary_axis_basal_radius,
        "primary_axis_max_radius": primary_axis_max_radius,
        "primary_axis_distal_tip_radius": primary_axis_distal_tip_radius,
        "primary_axis_basal_to_tip_radius_ratio": primary_axis_basal_to_tip_radius_ratio,
        "primary_axis_mean_radius": primary_axis_mean_radius,
        "primary_axis_radius_integral": primary_axis_radius_integral,
        "primary_structural_allocation": primary_structural_allocation,
        "lateral_structural_allocation": lateral_structural_allocation,
        "primary_fraction_total_structural_allocation": (
            primary_fraction_total_structural_allocation
        ),
        "lateral_fraction_total_structural_allocation": (
            lateral_fraction_total_structural_allocation
        ),
        "mean_lateral_radius": mean_lateral_radius,
        "lateral_to_primary_radius_ratio": lateral_to_primary_radius_ratio,
        "low_bp_taproot_score": low_bp_taproot_score,
        "branch_count_at_bp_001": branch_count_at_bp_001,
        "first_order_lateral_count": first_order_lateral_count,
        "higher_order_lateral_count": higher_order_lateral_count,
        "whorl_event_count": int(whorl_counts.size),
        "mean_branches_per_whorl": mean_branches_per_whorl,
        "max_branches_per_whorl": int(np.max(whorl_counts)) if whorl_counts.size else 0,
        "whorl_depth_spacing_mean": (
            float(np.mean(np.diff(whorl_depths))) if whorl_depths.size > 1 else 0.0
        ),
        "whorl_azimuth_entropy": whorl_azimuth_entropy,
        "fraction_laterals_in_whorls": fraction_laterals_in_whorls,
        "whorl_score": whorl_score,
        "stimulus_evaluated_probability_passes": int(
            axis_metadata.get("stimulus_evaluated_probability_passes", 0)
        ),
        "mean_local_primordium_stimulus": float(
            axis_metadata.get("mean_local_primordium_stimulus", 0.0)
        ),
        "initiation_uniform_mean": float(
            axis_metadata.get("initiation_uniform_mean", 0.0)
        ),
        "initiation_uniform_min": float(
            axis_metadata.get("initiation_uniform_min", 0.0)
        ),
        "initiation_uniform_max": float(
            axis_metadata.get("initiation_uniform_max", 0.0)
        ),
        "probability_failures": int(
            axis_metadata.get(
                "probability_failures",
                max(0, branch_opportunities - probability_passes),
            )
        ),
        "probability_pass_rate": _safe_ratio(
            probability_passes, branch_opportunities
        ),
        "probability_pass_acceptance_rate": _safe_ratio(
            successful, probability_passes
        ),
        "rejected_origin_surface_clearance": int(
            axis_metadata.get("rejected_origin_surface_clearance", 0)
        ),
        "rejected_above_soil_surface": int(
            axis_metadata.get("rejected_above_soil_surface", 0)
        ),
        "rejected_parent_collision": int(
            axis_metadata.get("rejected_parent_collision", 0)
        ),
        "rejected_other_root_collision": int(
            axis_metadata.get("rejected_other_root_collision", 0)
        ),
        "rejected_axis_ceiling": int(
            axis_metadata.get("rejected_axis_ceiling", 0)
        ),
        "rejected_sample_cap": int(
            axis_metadata.get("rejected_sample_cap", 0)
        ),
        "accepted_first_order_laterals": int(
            axis_metadata.get("accepted_first_order_laterals", 0)
        ),
        "accepted_higher_order_laterals": int(
            axis_metadata.get("accepted_higher_order_laterals", 0)
        ),
        "physical_rejection_count": int(
            axis_metadata.get("physical_rejection_count", 0)
        ),
        "physical_rejection_rate": float(
            axis_metadata.get("physical_rejection_rate", 0.0)
        ),
        "opportunity_accounting_error": int(
            axis_metadata.get("opportunity_accounting_error", 0)
        ),
        "probability_pass_accounting_error": int(
            axis_metadata.get("probability_pass_accounting_error", 0)
        ),
        "branch_origin_candidate_evaluations": int(
            axis_metadata.get("branch_origin_candidate_evaluations", 0)
        ),
        "active_tips_at_step_start_total": int(
            axis_metadata.get("active_tips_at_step_start_total", 0)
        ),
        "active_tip_observations": active_tip_observations,
        "tip_extension_attempts": int(
            axis_metadata.get("tip_extension_attempts", 0)
        ),
        "tip_extensions_accepted": int(
            axis_metadata.get("tip_extensions_accepted", 0)
        ),
        "tip_extensions_collision_blocked": int(
            axis_metadata.get("tip_extensions_collision_blocked", 0)
        ),
        "tip_extensions_surface_blocked": int(
            axis_metadata.get("tip_extensions_surface_blocked", 0)
        ),
        "tip_extensions_sample_cap_blocked": int(
            axis_metadata.get("tip_extensions_sample_cap_blocked", 0)
        ),
        "tip_extensions_other_blocked": int(
            axis_metadata.get("tip_extensions_other_blocked", 0)
        ),
        "fraction_active_tip_attempts_accepted": _safe_ratio(
            int(axis_metadata.get("tip_extensions_accepted", 0)),
            int(axis_metadata.get("tip_extension_attempts", 0)),
        ),
        "primary_tip_extension_attempts": int(
            axis_metadata.get("primary_tip_extension_attempts", 0)
        ),
        "primary_tip_extensions_accepted": int(
            axis_metadata.get("primary_tip_extensions_accepted", 0)
        ),
        "lateral_tip_extension_attempts": int(
            axis_metadata.get("lateral_tip_extension_attempts", 0)
        ),
        "lateral_tip_extensions_accepted": int(
            axis_metadata.get("lateral_tip_extensions_accepted", 0)
        ),
        "generation_1_extension_attempts": int(
            axis_metadata.get("generation_1_extension_attempts", 0)
        ),
        "generation_1_extensions_accepted": int(
            axis_metadata.get("generation_1_extensions_accepted", 0)
        ),
        "generation_2_extension_attempts": int(
            axis_metadata.get("generation_2_extension_attempts", 0)
        ),
        "generation_2_extensions_accepted": int(
            axis_metadata.get("generation_2_extensions_accepted", 0)
        ),
        "generation_3plus_extension_attempts": int(
            axis_metadata.get("generation_3plus_extension_attempts", 0)
        ),
        "generation_3plus_extensions_accepted": int(
            axis_metadata.get("generation_3plus_extensions_accepted", 0)
        ),
        "maximum_active_tip_count": int(
            axis_metadata.get("maximum_active_tip_count", 0)
        ),
        "final_active_tip_count": int(
            axis_metadata.get("final_active_tip_count", 0)
        ),
        "active_tip_attempt_accounting_error": int(
            axis_metadata.get("active_tip_attempt_accounting_error", 0)
        ),
        "branch_sites_created": int(axis_metadata.get("branch_sites_created", 0)),
        "branch_sites_currently_open": int(
            axis_metadata.get("branch_sites_currently_open", 0)
        ),
        "branch_sites_closed_single_trial": int(
            axis_metadata.get("branch_sites_closed_single_trial", 0)
        ),
        "branch_sites_temporarily_surface_full": int(
            axis_metadata.get("branch_sites_temporarily_surface_full", 0)
        ),
        "branch_sites_reopened_after_thickening": int(
            axis_metadata.get("branch_sites_reopened_after_thickening", 0)
        ),
        "branch_site_trials_total": int(
            axis_metadata.get("branch_site_trials_total", 0)
        ),
        "branch_site_first_trials": int(
            axis_metadata.get("branch_site_first_trials", 0)
        ),
        "branch_site_retry_trials": int(
            axis_metadata.get("branch_site_retry_trials", 0)
        ),
        "branch_site_probability_failures": int(
            axis_metadata.get("branch_site_probability_failures", 0)
        ),
        "branch_site_probability_passes": int(
            axis_metadata.get("branch_site_probability_passes", 0)
        ),
        "lateral_axis_diagnostics_json": json.dumps(
            lateral_rows, sort_keys=True
        ),
        **age_group_metrics,
        "proportion_laterals_initial_shoulder_only": _safe_ratio(
            sum(bool(row.get("only_initial_shoulder", False)) for row in lateral_rows),
            lateral_count,
        ),
        "proportion_laterals_at_least_2_accepted_extensions": _safe_ratio(
            sum(int(row.get("accepted_extensions", 0)) >= 2 for row in lateral_rows),
            lateral_count,
        ),
        "proportion_laterals_at_least_5_accepted_extensions": _safe_ratio(
            sum(int(row.get("accepted_extensions", 0)) >= 5 for row in lateral_rows),
            lateral_count,
        ),
        "proportion_laterals_at_least_10_accepted_extensions": _safe_ratio(
            sum(int(row.get("accepted_extensions", 0)) >= 10 for row in lateral_rows),
            lateral_count,
        ),
        "extension_parent_collision_blocked": extension_parent_collision_blocked,
        "extension_other_root_collision_blocked": (
            extension_other_root_collision_blocked
        ),
        "extension_surface_blocked": int(sum(
            int(row.get("surface_blocked_extensions", 0)) for row in lateral_rows
        )),
        "extension_other_blocked": int(sum(
            int(row.get("other_blocked_extensions", 0)) for row in lateral_rows
        )),
        "mean_initial_radial_displacement": (
            float(np.mean([
                float(row.get("initial_radial_displacement", 0.0))
                for row in lateral_rows
            ])) if lateral_rows else 0.0
        ),
        "mean_distance_from_parent_after_shoulder": (
            float(np.mean([
                float(row.get("distance_from_parent_after_shoulder", 0.0))
                for row in lateral_rows
            ])) if lateral_rows else 0.0
        ),
        "mean_lateral_direction_z_after_emergence": (
            float(np.mean([
                float(row.get("mean_direction_z_after_emergence", 0.0))
                for row in lateral_rows
                if int(row.get("accepted_extensions", 0)) > 0
            ]))
            if any(int(row.get("accepted_extensions", 0)) > 0 for row in lateral_rows)
            else 0.0
        ),
        "fraction_laterals_curve_back_inside_parent_radius": _safe_ratio(
            sum(bool(row.get("curves_back_inside_parent_radius", False)) for row in lateral_rows),
            lateral_count,
        ),
        "fraction_laterals_only_one_support_curve": _safe_ratio(
            sum(bool(row.get("only_one_support_curve", False)) for row in lateral_rows),
            lateral_count,
        ),
        "fraction_laterals_active_at_termination": _safe_ratio(
            sum(bool(row.get("current_active_state", False)) for row in lateral_rows),
            lateral_count,
        ),
        "profile_retry_site_traversal_sec": float(
            axis_metadata.get("profile_retry_site_traversal_sec", 0.0)
        ),
        "profile_branch_probability_trials_sec": float(
            axis_metadata.get("profile_branch_probability_trials_sec", 0.0)
        ),
        "profile_physical_origin_search_sec": float(
            axis_metadata.get("profile_physical_origin_search_sec", 0.0)
        ),
        "profile_active_tip_extensions_sec": float(
            axis_metadata.get("profile_active_tip_extensions_sec", 0.0)
        ),
        "profile_collision_queries_sec": float(
            axis_metadata.get("profile_collision_queries_sec", 0.0)
        ),
        "profile_resource_direction_candidates_sec": float(
            axis_metadata.get("profile_resource_direction_candidates_sec", 0.0)
        ),
        "multi_branch_site_count": len(multi_sites),
        "maximum_branches_at_one_site": max(
            [int(site.get("accepted_branch_count", 0)) for site in occupied_sites],
            default=0,
        ),
        "mean_branches_per_occupied_site": (
            float(np.mean([
                int(site.get("accepted_branch_count", 0)) for site in occupied_sites
            ])) if occupied_sites else 0.0
        ),
        "fraction_branches_from_multi_branch_sites": _safe_ratio(
            sum(int(site.get("accepted_branch_count", 0)) for site in multi_sites),
            successful,
        ),
        "same_site_min_azimuth_separation_deg": (
            float(np.min(same_site_separations)) if same_site_separations else 0.0
        ),
        "same_site_mean_azimuth_separation_deg": (
            float(np.mean(same_site_separations)) if same_site_separations else 0.0
        ),
        "minimum_accepted_axial_origin_separation": (
            float(np.min(accepted_axial_separations))
            if accepted_axial_separations else 0.0
        ),
        "accepted_origin_surface_clearance_min": (
            float(np.min(clearances)) if clearances.size else 0.0
        ),
        "accepted_origin_surface_clearance_mean": (
            float(np.mean(clearances)) if clearances.size else 0.0
        ),
        "parent_radius_at_branch_origin_mean": (
            float(np.mean(parent_origin_radii)) if parent_origin_radii.size else 0.0
        ),
        "parent_radius_at_multi_branch_site_mean": (
            float(np.mean(multi_parent_radii)) if multi_parent_radii.size else 0.0
        ),
        "branch_origin_child_parent_radius_ratio_mean": (
            float(np.mean(child_parent_radius_ratios))
            if child_parent_radius_ratios.size else 0.0
        ),
        "branch_origin_child_parent_radius_ratio_max": (
            float(np.max(child_parent_radius_ratios))
            if child_parent_radius_ratios.size else 0.0
        ),
        "first_order_origin_depth_min": (
            float(np.min(first_order_origin_depths))
            if first_order_origin_depths.size else 0.0
        ),
        "first_order_origin_depth_p10": origin_quantile(
            first_order_origin_depths, 10.0
        ),
        "first_order_origin_depth_p25": origin_quantile(
            first_order_origin_depths, 25.0
        ),
        "first_order_origin_depth_median": origin_quantile(
            first_order_origin_depths, 50.0
        ),
        "first_order_origin_depth_p75": origin_quantile(
            first_order_origin_depths, 75.0
        ),
        "first_order_origin_depth_p90": origin_quantile(
            first_order_origin_depths, 90.0
        ),
        "first_order_origin_depth_max": (
            float(np.max(first_order_origin_depths))
            if first_order_origin_depths.size else 0.0
        ),
        "first_order_origin_arc_fraction_mean": (
            float(np.mean(first_order_origin_arc_fractions))
            if first_order_origin_arc_fractions.size else 0.0
        ),
        "first_order_origin_arc_fraction_p10": origin_quantile(
            first_order_origin_arc_fractions, 10.0
        ),
        "first_order_origin_arc_fraction_p50": origin_quantile(
            first_order_origin_arc_fractions, 50.0
        ),
        "first_order_origin_arc_fraction_p90": origin_quantile(
            first_order_origin_arc_fractions, 90.0
        ),
        "fraction_first_order_origins_in_proximal_10_percent": (
            float(np.mean(first_order_origin_arc_fractions <= 0.10))
            if first_order_origin_arc_fractions.size else 0.0
        ),
        "fraction_first_order_origins_in_proximal_25_percent": (
            float(np.mean(first_order_origin_arc_fractions <= 0.25))
            if first_order_origin_arc_fractions.size else 0.0
        ),
        "fraction_first_order_origins_in_middle_50_percent": (
            float(np.mean(
                (first_order_origin_arc_fractions > 0.25)
                & (first_order_origin_arc_fractions <= 0.75)
            ))
            if first_order_origin_arc_fractions.size else 0.0
        ),
        "fraction_first_order_origins_in_distal_25_percent": (
            float(np.mean(first_order_origin_arc_fractions > 0.75))
            if first_order_origin_arc_fractions.size else 0.0
        ),
        "primary_radius_at_10_percent": float(primary_radius_samples[1]),
        "primary_radius_at_25_percent": float(primary_radius_samples[2]),
        "primary_radius_at_50_percent": float(primary_radius_samples[3]),
        "primary_radius_at_75_percent": float(primary_radius_samples[4]),
        "primary_radius_at_90_percent": float(primary_radius_samples[5]),
        "primary_taper_monotonic_fraction": primary_taper_fraction,
        "primary_max_local_radius_increase_away_from_junction": primary_radius_bump,
        "mean_first_order_basal_tip_radius_ratio": (
            float(np.mean(first_order_taper_ratios))
            if first_order_taper_ratios else 0.0
        ),
        "mean_first_order_taper_monotonic_fraction": (
            float(np.mean(first_order_taper_fractions))
            if first_order_taper_fractions else 0.0
        ),
        "mean_extension_direction_z": (
            float(np.mean(vertical)) if vertical.size else 0.0
        ),
        "median_extension_direction_z": (
            float(np.median(vertical)) if vertical.size else 0.0
        ),
        "fraction_extensions_direction_z_lt_minus_08": (
            float(np.mean(vertical < -0.8)) if vertical.size else 0.0
        ),
        "total_water_captured": total_resources[0], "total_K_captured": total_resources[3],
        "water_capture_per_total_length": _safe_ratio(total_resources[0], total_length),
        "phosphorus_capture_per_total_length": _safe_ratio(total_resources[1], total_length),
        "nitrogen_capture_per_total_length": _safe_ratio(total_resources[2], total_length),
        "potassium_capture_per_total_length": _safe_ratio(total_resources[3], total_length),
        "water_capture_per_surface_area": _safe_ratio(total_resources[0], total_area),
        "phosphorus_capture_per_surface_area": _safe_ratio(total_resources[1], total_area),
        "nitrogen_capture_per_surface_area": _safe_ratio(total_resources[2], total_area),
        "potassium_capture_per_surface_area": _safe_ratio(total_resources[3], total_area),
        "total_root_length": total_length, "total_surface_area": total_area,
        "total_root_volume": total_volume,
        "specific_root_length": _safe_ratio(total_length, total_volume),
        "max_root_path_length": float(np.max(path_length)),
        "mean_root_path_length": float(np.mean(path_length[edge_mask])) if store.size > 1 else 0.0,
        "median_root_path_length": float(np.median(path_length[edge_mask])) if store.size > 1 else 0.0,
        "max_horizontal_width": max(width_x, width_y), "max_width_x": width_x,
        "max_width_y": width_y, "z_range": z_range,
        "bounding_box_volume": width_x * width_y * z_range,
        "convex_hull_volume": convex_hull_volume,
        "center_of_mass_x": float(center[0]), "center_of_mass_y": float(center[1]),
        "branch_points": branch_points, "mean_children": float(np.mean(child_counts)),
        "mean_topological_depth": float(np.mean(topological_depth)),
        "global_strahler_order": int(orders[0]),
        "number_of_strahler_orders_present": int(np.unique(orders).size),
        "branch_density_per_length": _safe_ratio(branch_points, total_length),
        "tip_density_per_length": _safe_ratio(int(np.count_nonzero(child_counts == 0)), total_length),
        "lateral_to_anchor_ratio": _safe_ratio(lateral_nodes, int(np.count_nonzero(anchor_mask))),
        "root_length_per_node": _safe_ratio(total_length, store.size),
        "mean_gravitropic_divergence_deg": float(np.mean(gravity)) if gravity.size else 0.0,
        "mean_pitch_angle_deg": float(np.mean(pitch)) if pitch.size else 0.0,
        "mean_absolute_vertical_component": float(np.mean(absolute_vertical)) if vertical.size else 0.0,
        "mean_lateral_component": float(np.mean(lateral_component)) if lateral_component.size else 0.0,
        "fraction_near_vertical_segments": near_vertical_fraction,
        "fraction_strongly_lateral_segments": float(np.mean(absolute_vertical <= 0.35)) if vertical.size else 0.0,
        "fraction_upward_segments": upward_fraction,
        "fraction_near_horizontal_segments": near_horizontal_fraction,
        "mean_branch_vertical_component": mean_branch_vertical,
        "mean_continuation_vertical_component": float(np.mean(store.direction[:store.size, 2][continuation_edge_mask])) if np.any(continuation_edge_mask) else 0.0,
        "mean_environmental_resource_signal": mean_environmental_resource_signal,
        "mean_direction_resource_score": float(np.mean(store.direction_resource_score[:store.size][branch_edge_mask])) if np.any(branch_edge_mask) else 0.0,
        "mean_direction_resource_gate": float(np.mean(store.direction_resource_gate[:store.size][branch_edge_mask])) if np.any(branch_edge_mask) else 0.0,
        "mean_gravitropism_score": float(np.mean(store.direction_gravitropism_score[:store.size][branch_edge_mask])) if np.any(branch_edge_mask) else 0.0,
        "mean_lateral_exploration_score": float(np.mean(store.direction_lateral_exploration_score[:store.size][branch_edge_mask])) if np.any(branch_edge_mask) else 0.0,
        "mean_lateral_suppression_score": float(np.mean(store.direction_lateral_suppression_score[:store.size][branch_edge_mask])) if np.any(branch_edge_mask) else 0.0,
        **smoothness_diagnostics,
        **curve_axis_diagnostics,
        "mean_tortuosity": float(np.mean(tortuosity[edge_mask])) if store.size > 1 else 0.0,
        "max_tortuosity": float(np.max(tortuosity[edge_mask])) if store.size > 1 else 0.0,
        "path_efficiency_mean": float(np.mean(efficiency[edge_mask])) if store.size > 1 else 0.0,
        "path_efficiency_min": float(np.min(efficiency[edge_mask])) if store.size > 1 else 0.0,
        "fraction_length_topsoil": _safe_ratio(float(np.sum(lengths[(midpoint_z >= config.phosphorus_z_low) & edge_mask])), total_length),
        "fraction_length_upper_subsoil": _safe_ratio(float(np.sum(lengths[(midpoint_z < config.phosphorus_z_low) & (midpoint_z > config.nitrogen_z_high) & edge_mask])), total_length),
        "fraction_length_nitrogen_layer": _safe_ratio(float(np.sum(lengths[(midpoint_z <= config.nitrogen_z_high) & (midpoint_z >= config.nitrogen_z_low) & edge_mask])), total_length),
        "fraction_length_deep_soil": _safe_ratio(float(np.sum(lengths[(midpoint_z < config.nitrogen_z_low) & edge_mask])), total_length),
        "fraction_length_below_nitrogen_layer": _safe_ratio(float(np.sum(lengths[(midpoint_z < config.nitrogen_z_low) & edge_mask])), total_length),
    })
    metrics.update(_per_strahler_metrics(
        store, orders, lengths, areas, volumes, midpoints,
        topological_depth, efficiency, tortuosity,
    ))
    metrics.update(_per_generation_direction_metrics(store, branch_generations))
    return metrics


def accumulate_resource_capture(
    store: NodeStore,
    sweep_size: int,
    water: np.ndarray,
    phosphorus: np.ndarray,
    nitrogen: np.ndarray,
    potassium: np.ndarray,
    config: SimulationConfig,
) -> None:
    """Accumulate per-node potential uptake using exposed segment surface area."""

    store.resource_observations[:sweep_size] += 1
    for values, target in (
        (water, store.water_availability_sum),
        (phosphorus, store.phosphorus_availability_sum),
        (nitrogen, store.nitrogen_availability_sum),
        (potassium, store.potassium_availability_sum),
    ):
        target[:sweep_size] += values
    if sweep_size <= 1:
        return
    child = np.arange(1, sweep_size)
    parent = store.parent[1:sweep_size]
    segment_length = np.linalg.norm(
        store.position[1:sweep_size] - store.position[parent], axis=1
    )
    exposed_area = 2.0 * math.pi * store.radius[1:sweep_size] * segment_length
    depth = np.maximum(-store.position[1:sweep_size, 2], 0.0)
    resource_specs = (
        (water, config.water_capture_per_iteration, store.water_captured,
         store.water_capture_depth_sum, store.deepest_water_capture),
        (phosphorus, config.phosphorus_capture_per_iteration, store.phosphorus_captured,
         store.phosphorus_capture_depth_sum, store.deepest_phosphorus_capture),
        (nitrogen, config.nitrogen_capture_per_iteration, store.nitrogen_captured,
         store.nitrogen_capture_depth_sum, store.deepest_nitrogen_capture),
        (potassium, config.potassium_capture_per_iteration, store.potassium_captured,
         store.potassium_capture_depth_sum, store.deepest_potassium_capture),
    )
    for availability, rate, captured_total, depth_total, deepest in resource_specs:
        captured = availability[1:sweep_size] * rate * exposed_area
        captured_total[1:sweep_size] += captured
        depth_total[1:sweep_size] += captured * depth
        positive = captured > CAPTURE_REPORTING_EPSILON
        if np.any(positive):
            selected = child[positive]
            deepest[selected] = np.maximum(deepest[selected], depth[positive])


def accumulate_resource_capture_for_nodes(
    store: NodeStore,
    node_ids: Sequence[int] | np.ndarray,
    raining: bool,
    config: SimulationConfig,
) -> None:
    """Incremental event-mode uptake for newly created segment endpoints."""

    ids = np.asarray(node_ids, dtype=np.int64)
    ids = ids[(ids > 0) & (ids < store.size)]
    if ids.size == 0:
        return
    water, phosphorus, nitrogen, potassium = resource_values(
        store.position[ids, 2], raining, config
    )
    store.resource_observations[ids] += 1
    for values, target in (
        (water, store.water_availability_sum),
        (phosphorus, store.phosphorus_availability_sum),
        (nitrogen, store.nitrogen_availability_sum),
        (potassium, store.potassium_availability_sum),
    ):
        target[ids] += values

    parent = store.parent[ids]
    segment_length = np.linalg.norm(store.position[ids] - store.position[parent], axis=1)
    exposed_area = 2.0 * math.pi * store.radius[ids] * segment_length
    depth = np.maximum(-store.position[ids, 2], 0.0)
    resource_specs = (
        (water, config.water_capture_per_iteration, store.water_captured,
         store.water_capture_depth_sum, store.deepest_water_capture),
        (phosphorus, config.phosphorus_capture_per_iteration, store.phosphorus_captured,
         store.phosphorus_capture_depth_sum, store.deepest_phosphorus_capture),
        (nitrogen, config.nitrogen_capture_per_iteration, store.nitrogen_captured,
         store.nitrogen_capture_depth_sum, store.deepest_nitrogen_capture),
        (potassium, config.potassium_capture_per_iteration, store.potassium_captured,
         store.potassium_capture_depth_sum, store.deepest_potassium_capture),
    )
    for availability, rate, captured_total, depth_total, deepest in resource_specs:
        captured = availability * rate * exposed_area
        captured_total[ids] += captured
        depth_total[ids] += captured * depth
        positive = captured > CAPTURE_REPORTING_EPSILON
        if np.any(positive):
            deepest[ids[positive]] = np.maximum(deepest[ids[positive]], depth[positive])


def run_simulation(
    parameters: SimulationParameters,
    config: SimulationConfig,
    *,
    return_store: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_interval_steps: int = 0,
    resume_checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    progress_interval_steps: int = 1,
    pause_after_checkpoint_step: int | None = None,
) -> dict[str, int | float | str] | tuple[dict[str, int | float | str], NodeStore]:
    """Run one deterministic schema-v23 continuous-axis simulation.

    Every active snapshot tip receives one bounded extension attempt. Poisson
    material sites then receive initiation trials, and transport-area range
    additions determine local physical radii. Sampled-point limits are
    transactional technical caps only.
    """

    parameters.validate()
    config.validate()
    return _axis_curve_simulation(
        parameters,
        config,
        return_store=return_store,
        checkpoint_path=checkpoint_path,
        checkpoint_interval_steps=checkpoint_interval_steps,
        resume_checkpoint_path=resume_checkpoint_path,
        progress_path=progress_path,
        progress_interval_steps=progress_interval_steps,
        pause_after_checkpoint_step=pause_after_checkpoint_step,
    )

def index_radius(store: NodeStore) -> float:
    return float(np.max(store.radius[: store.size]))


_WORKER_CONFIG: SimulationConfig | None = None
_WORKER_MASTER_SEED = 0


def initialize_worker(config: SimulationConfig, master_seed: int) -> None:
    global _WORKER_CONFIG, _WORKER_MASTER_SEED
    _WORKER_CONFIG = config
    _WORKER_MASTER_SEED = int(master_seed)


def worker_run(task_index: int) -> dict[str, int | float | str]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("worker was not initialized")
    parameters = parameters_for_task(task_index, _WORKER_MASTER_SEED)
    result = run_simulation(parameters, _WORKER_CONFIG)
    assert isinstance(result, dict)
    return result


def task_indices(
    *,
    start: int,
    stop: int,
    shard_id: int,
    num_shards: int,
    completed: np.ndarray | None,
) -> Iterator[int]:
    first = start + ((shard_id - start) % num_shards)
    for task_index in range(first, stop, num_shards):
        if completed is None or not bool(completed[task_index]):
            yield task_index


def count_shard_tasks(start: int, stop: int, shard_id: int, num_shards: int) -> int:
    first = start + ((shard_id - start) % num_shards)
    if first >= stop:
        return 0
    return 1 + (stop - 1 - first) // num_shards


def completion_bitmap(path: Path, resume: bool) -> np.memmap:
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"completion bitmap exists: {path}; pass --resume or choose another output"
            )
        bitmap = np.lib.format.open_memmap(path, mode="r+")
        if bitmap.shape != (TOTAL_GRID_TASKS,) or bitmap.dtype != np.bool_:
            raise ValueError(f"invalid completion bitmap: {path}")
        return bitmap
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.bool_,
        shape=(TOTAL_GRID_TASKS,),
    )


def write_metadata(
    path: Path,
    config: SimulationConfig,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "model": "elastic-root-architecture-hpc",
        "schema_version": SCHEMA_VERSION,
        "resource_model_version": RESOURCE_MODEL_VERSION,
        "direction_model_version": DIRECTION_MODEL_VERSION,
        "curve_model_version": CURVE_MODEL_VERSION,
        "initiation_model_version": INITIATION_MODEL_VERSION,
        "initiation_random_stream_version": INITIATION_RANDOM_STREAM_VERSION,
        "initiation_probability_resource_independent": True,
        "canonical_branch_min_spacing_along_axis": (
            CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS
        ),
        "branch_retry_mode": config.branch_retry_mode,
        "branch_retry_modes_available": list(BRANCH_RETRY_MODES),
        "branch_retry_mode_is_grid_dimension": False,
        "canonical_production_branch_retry_mode": None,
        "branch_retry_mode_production_calibrated": False,
        "max_reported_strahler_order": MAX_REPORTED_STRAHLER_ORDER,
        "capture_reporting_epsilon": CAPTURE_REPORTING_EPSILON,
        "strahler_overflow_bucket": f"gt_{MAX_REPORTED_STRAHLER_ORDER}",
        "convex_hull_enabled": config.compute_convex_hull,
        "result_field_count": len(RESULT_FIELDS),
        "result_fields": RESULT_FIELDS,
        "resource_model_notes": {
            "water": "background plus a smooth time-dependent wetting-front pulse on a semi-infinite z profile",
            "phosphorus": "shallow retained exponential profile with negligible rain transport",
            "nitrogen": "mobile profile and 3-D patches shifted downward by cumulative rain",
            "potassium": "intermediate-mobility profile with a weaker rain-dependent shift",
            "heterogeneity": "schema v26 preserves seeded smooth 3-D water/P/N/K hotspots over absolute z-anchored dynamic profiles",
            "capture": "dimensionless potential uptake per exposed surface area; no depletion PDE",
        },
        "curve_model_notes": {
            "summary": "schema v26 preserves fixed developmental time, all-active-tip growth, continuous material-arc sites, cylindrical collar clearance, transport-path area tapering, and a two-extension geometry-based lateral escape state",
            "morphology": "architecture is never selected: taproot dominance, distributed branching, and local whorls emerge from continuous parameters; emergent_morphology_class is post-hoc only",
            "branching": "sites are a seeded Poisson point process on newly mature material arc; sites are not fixed support nodes or rigid non-overlapping intervals",
            "single_trial_semantics": "B.P. is a one-time initiation probability per newly mature site; every site closes after its first probability draw",
            "retry_open_sites_semantics": "B.P. is a per-step initiation hazard at each physically open mature site; at most one fresh-azimuth draw occurs per site per step and thickening can reopen full sites",
            "initiation_probability": "threshold=lineage_branch_probability(configured B.P., parent_generation); generation 0 is exactly configured B.P.; no water, rain, P, N, K, demand, focus, support, starvation, stimulus, geometry, or prior acceptance enters the threshold",
            "initiation_random_stream": "each probability variate is a counter-based function of the simulation seed, task index, axis lineage identifier, site identifier, and one-based trial number; it is isolated from all mutable post-initiation random streams",
            "posthoc_whorls": "after growth only, first-order origins within the fixed diagnostic axial window are classified as whorl-like groups; detection outputs never enter growth",
            "tip_elongation": "tips extend by curvature-limited cubic Hermite samples rather than one rigid straight segment",
            "branch_origin": "lateral roots originate from mature arc-length positions along parent axes, not only from pre-existing node junctions",
            "canonical_branch_min_spacing_along_axis": "0.20 is the mean exponential inter-site gap for Streamlit, smoke, single-run, and batch/HPC entry points; it is not a hard minimum or grid dimension",
            "branch_angles": "lateral emergence varies continuously from a 10-35 degree scarcity shoulder toward a broad rich balanced distribution near 80-95 degrees, with shallow P/K and deeper water/N focus shifts",
            "surface_constraint": "candidate direction scores penalize movement above the soil surface and candidate Hermite curves are rejected if sampled points exceed the configured surface tolerance",
            "developmental_steps": "config.steps is the canonical fixed biological duration for every grid task; max_growth_iterations is only a deprecated explicit override",
            "sampled_point_safety_cap": "max_sampled_points limits numerical support geometry and memory; reaching it is incomplete sample_cap termination and never normal completion",
            "post_initiation_growth": "every active snapshot tip receives one bounded attempt; primary length=max(anchor_min_segment_length,anchor_segment_length(step)); for axis_age=max(0,step-birth_step), lateral length=max(lateral_min_segment_length,segment_length*lateral_relative_elongation*lateral_generation_length_decay^(generation-1)*exp(-axis_age/lateral_elongation_decay_timescale)); the same strictly positive age-decaying law applies at every B.P., and B.P. and resources never remove the attempt",
            "removed_v22_allocation_logic": "no plant-wide budget, ranking, subset selection, waiting priority, or protected-primary allocation exists",
            "transport_area": "self dA=T.I.*grown_length*structural_self_area_coefficient*F over [axis base,current tip], F=1 primary or structural_lateral_self_area_fraction lateral; ancestor depth d receives dA=T.I.*grown_length*structural_ancestor_area_coefficient*structural_ancestor_transport_decay^(d-1) over [ancestor base,descendant attachment]; a proximal area correction max(0,A_child/q^2-A_parent), q=branch_origin_child_parent_radius_ratio_limit, guarantees child basal radius <=q*parent local radius",
            "radius_constitutive_equation": "radius(material_arc)=sqrt((pi*tip_baseline_radius^2 + sum of covering range-addition dA)/pi)",
            "range_addition": "events store (inclusive end_material_arc, area_increment); sorted ends and reverse cumulative area are cached per axis version, and radius queries use searchsorted",
            "cylindrical_surface_clearance": "sqrt(delta_arc^2 + (mean_local_parent_radius*wrapped_delta_theta)^2) - branch_collar_clearance_factor*(candidate_collar+existing_collar) - branch_collar_safety_margin >= 0",
            "collision": "candidate curves are checked as sampled swept centerline points against the KD-tree-backed spatial index",
            "newborn_lateral_escape": "for the first two accepted post-emergence extensions only, a geometry-derived parent-relative radial component is required and the existing attachment-collar parent exemption remains active; B.P., rain, and resource state do not enter this state, and full collision/direction behavior resumes afterward",
        },
        "direction_model_notes": {
            "summary": "every active terminal tip attempts curvature-limited elongation from persistence, stochastic, horizontal, gravity, and resource-gradient candidates; resources rank directions but never grant growth permission",
            "branch_candidate_space": "new lateral candidates are constructed from the parent root tangent plus a local perpendicular frame; gravity, plagiotropism, resources, radial balance, and stochasticity rank those candidates but do not define them",
            "low_resource_behavior": "B.P. controls initiation independently of nutrients; near-zero local water/P/N/K availability gives accepted branches a visible shoulder, then steepens/downward-bends and radially confines them until physical crowding limits growth",
            "high_resource_behavior": "P promotes shallow lateral foraging, rain-leached N promotes deeper/downward foraging, K promotes intermediate-depth foraging, and balanced support permits a fuller architecture",
            "branch_probability": "branch_probability has an exact lineage-only statistical meaning; resources and transient local stimulus never alter a site probability draw and act only after a pass",
            "active_tip_elongation": "the beginning-of-step active snapshot receives exactly one bounded attempt per tip; accepted new laterals wait until the following step",
            "lateral_initiation_zone": "material sites require minimum axis age and remain beyond the protected base and behind the immature tip zone",
            "lateral_emergence": "every accepted event creates one shorter parent-relative lateral at a continuous material coordinate and fresh azimuth; multiple same-site branches are permitted when cylindrical surface and 3D curve clearance permit",
            "v_shape_diagnostics": "hard forks are branching nodes where no child continues within the configured continuation-angle threshold; v_shape_score also includes repeated lateral children from the same point",
            "repeated_direction_diagnostics": "diagnostics include upward/near-horizontal fractions, branch-vs-continuation vertical components, generation-wise vertical metrics, and same-axis direction-similarity metrics",
            "resource_demand_feedback": "schema v26 preserves v24 capture normalization, absent-supply gating, active-target renormalization, smoothed deficiencies, and bounded demand; all affect only post-pass focus/direction diagnostics, never B.P.",
            "branch_opportunities": "site density is independent of B.P.; retry interpretation is explicitly selected and recorded",
            "probability_pass_audit": "opportunities=failures+passes and passes=accepted+physical rejections; each rejection is assigned once to surface-clearance, above-surface, parent-collision, other-root-collision, sample-cap, or axis-ceiling",
            "structural_allocation": "accepted extension thickens its own proximal path and recursively attenuated ancestor paths; no descendant contribution applies distal to its attachment",
            "tip_resource_focus": "each axis receives one reproducible stochastic water/P/N/K/balanced focus; it persists for a minimum interval, can update afterward, and can never select a resource with zero environmental supply",
            "bounded_upward": "meaningful positive resource gain and below-surface position are required; candidate z is bounded and accepted consecutive upward extensions cannot exceed the configured maximum of two",
            "direction_score": "score = 1.35*persistence + sufficiency-conditioned gravity + 3.40*sufficiency*horizontal + 1.20*focus_absolute_value + 5.00*positive_focus_gain + surface_penalty + bounded_upward_penalty; newborn laterals add only a two-extension geometry-based outward score and constraint",
            "derived_direction_logic": "poor support strengthens gravity continuously; rich support strengthens plagiotropism continuously, while axis-specific focus and local 3-D gradients distribute trajectories.",
            "fixed_internal_availability_weights": {
                "water": WATER_AVAILABILITY_DIRECTION_WEIGHT,
                "phosphorus": PHOSPHORUS_AVAILABILITY_DIRECTION_WEIGHT,
                "nitrogen": NITROGEN_AVAILABILITY_DIRECTION_WEIGHT,
                "potassium": POTASSIUM_AVAILABILITY_DIRECTION_WEIGHT,
                "availability_score_weight": RESOURCE_AVAILABILITY_SCORE_WEIGHT,
            },
        },
        "total_grid_tasks": TOTAL_GRID_TASKS,
        "grid": {
            "thickness": {"start": 0.1, "stop": 7.0, "step": 0.1},
            "rain_probability": {"start": 0.01, "stop": 0.99, "step": 0.01},
            "branch_probability": {"start": 0.01, "stop": 0.99, "step": 0.01},
            "replicates": GRID_REPLICATES,
            "branch_spacing_is_dimension": False,
            "branch_retry_mode_is_dimension": False,
        },
        "config": asdict(config),
        "master_seed": args.master_seed,
        "task_start": args.task_start,
        "task_stop": args.task_stop,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    serialized = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError(
                f"existing metadata does not match this run configuration: {path}"
            )
        return
    path.write_text(serialized)


def reconcile_completion_from_csv(csv_path: Path, completed: np.ndarray) -> int:
    """Make the CSV authoritative after a crash between its fsync and bitmap fsync."""

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    recovered = 0
    with csv_path.open(newline="") as source:
        for row in csv.DictReader(source):
            raw = row.get("task_index")
            if raw is None:
                continue
            try:
                task_index = int(raw)
            except ValueError:
                continue
            if 0 <= task_index < TOTAL_GRID_TASKS and not completed[task_index]:
                completed[task_index] = True
                recovered += 1
    if recovered:
        completed.flush()
    return recovered


def validate_existing_csv_schema(csv_path: Path) -> None:
    """Prevent accidental resume into an older or altered CSV schema."""

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="") as source:
        header = next(csv.reader(source), [])
    if header != RESULT_FIELDS:
        raise ValueError(
            f"existing CSV schema is incompatible with schema version {SCHEMA_VERSION}: {csv_path}"
        )


def run_batch(args: argparse.Namespace, config: SimulationConfig) -> None:
    start = args.task_start
    stop = TOTAL_GRID_TASKS if args.task_stop is None else args.task_stop
    if not 0 <= start <= stop <= TOTAL_GRID_TASKS:
        raise ValueError(
            f"task range must satisfy 0 <= start <= stop <= {TOTAL_GRID_TASKS}"
        )
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = (
        ""
        if args.num_shards == 1
        else f"_shard-{args.shard_id:05d}-of-{args.num_shards:05d}"
    )
    csv_path = outdir / f"root_architecture_results{suffix}.csv"
    bitmap_path = csv_path.with_suffix(csv_path.suffix + ".done.npy")
    metadata_path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")

    if csv_path.exists() and not args.resume:
        raise FileExistsError(
            f"output exists: {csv_path}; pass --resume or choose another output"
        )
    completed = completion_bitmap(bitmap_path, args.resume)
    if args.resume:
        validate_existing_csv_schema(csv_path)
        recovered = reconcile_completion_from_csv(csv_path, completed)
        if recovered:
            print(f"recovered {recovered:,} completion markers from existing CSV")
    write_metadata(metadata_path, config, args)

    scheduled_total = count_shard_tasks(
        start, stop, args.shard_id, args.num_shards
    )
    already_completed = int(
        np.count_nonzero(completed[start:stop])
    )
    pending = task_indices(
        start=start,
        stop=stop,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        completed=completed,
    )

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    print(
        f"Grid={TOTAL_GRID_TASKS:,}; shard tasks={scheduled_total:,}; "
        f"bitmap completed in selected range={already_completed:,}; workers={args.cores}"
    )

    mode = "a" if csv_path.exists() else "w"
    processed = 0
    started = time.perf_counter()
    mp_context = mp.get_context(args.start_method)

    with csv_path.open(mode, newline="", buffering=1024 * 1024) as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
            output.flush()

        pending_markers: list[int] = []
        with mp_context.Pool(
            processes=args.cores,
            initializer=initialize_worker,
            initargs=(config, args.master_seed),
            maxtasksperchild=args.max_tasks_per_worker,
        ) as pool:
            results = pool.imap_unordered(
                worker_run,
                pending,
                chunksize=args.pool_chunksize,
            )
            for result in results:
                writer.writerow(result)
                processed += 1
                task_index = int(result["task_index"])
                pending_markers.append(task_index)
                if processed % args.checkpoint_every == 0:
                    output.flush()
                    os.fsync(output.fileno())
                    completed[np.asarray(pending_markers, dtype=np.int64)] = True
                    completed.flush()
                    pending_markers.clear()
                    elapsed = time.perf_counter() - started
                    rate = processed / elapsed if elapsed else 0.0
                    print(
                        f"completed {processed:,} this run at {rate:.2f} simulations/s; "
                        f"last task={task_index}",
                        flush=True,
                    )
        output.flush()
        os.fsync(output.fileno())
        if pending_markers:
            completed[np.asarray(pending_markers, dtype=np.int64)] = True
        completed.flush()

    elapsed = time.perf_counter() - started
    print(
        f"Batch complete: {processed:,} simulations written to {csv_path} "
        f"in {elapsed:.2f}s"
    )


def run_single(args: argparse.Namespace, config: SimulationConfig) -> None:
    parameters = SimulationParameters(
        rain_probability=args.rain_probability,
        branch_probability=args.branch_probability,
        thickness_increment=args.thickness_increment,
        seed=args.seed,
        sim_id="single",
    )
    result = run_simulation(parameters, config)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_smoke(config: SimulationConfig) -> None:
    smoke_target = min(
        effective_target_architecture_size(config),
        min(config.max_nodes, 10_000),
    )
    smoke_config = SimulationConfig(
        **{
            **asdict(config),
            "steps": min(config.steps, 25),
            "max_nodes": min(config.max_nodes, 10_000),
            "max_sampled_points": smoke_target,
            "target_architecture_size": smoke_target,
            "interactive_safety_cap": smoke_target,
        }
    )
    cases = (
        (0.0, 0.99, 0.1, 101),
        (0.5, 0.25, 0.5, 202),
        (0.9, 0.9, 0.1, 303),
        (0.9, 0.9, 7.0, 404),
    )
    for rain, branch, thickness, seed in cases:
        parameters = SimulationParameters(
            rain, branch, thickness, seed, f"smoke-{seed}"
        )
        result = run_simulation(parameters, smoke_config)
        assert isinstance(result, dict)
        print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Elastic Root Architecture Engine — deterministic HPC edition"
    )
    parser.add_argument(
        "--mode",
        choices=("single", "batch", "smoke"),
        default="single",
        help="single simulation, sharded full batch, or short smoke benchmark",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100_000,
        help="technical cap on sampled curve support points used for collision/rendering",
    )
    parser.add_argument(
        "--max-sampled-points",
        type=int,
        default=100_000,
        help="technical sampled-point memory/geometry cap; reaching it is incomplete",
    )
    parser.add_argument(
        "--target-architecture-size",
        type=int,
        default=0,
        help="deprecated safety-cap alias; 0 disables the alias",
    )
    parser.add_argument(
        "--target-axis-count",
        type=int,
        default=0,
        help="deprecated optional axis safety ceiling; 0 uses sampled-point capacity",
    )
    parser.add_argument(
        "--max-growth-iterations",
        type=int,
        default=0,
        help="deprecated explicit developmental-step override; 0 uses --steps",
    )
    parser.add_argument("--interactive-safety-cap", type=int, default=100_000)
    parser.add_argument("--max-seconds-per-simulation", type=float, default=300.0)
    parser.add_argument("--angle-candidates", type=int, default=24)
    parser.add_argument("--segment-length", type=float, default=0.5)
    parser.add_argument("--anchor-initial-segment-length", type=float, default=0.30)
    parser.add_argument("--anchor-min-segment-length", type=float, default=0.05)
    parser.add_argument("--anchor-decay-timescale", type=float, default=75.0)
    parser.add_argument(
        "--branch-min-spacing-along-axis",
        type=float,
        default=CANONICAL_BRANCH_MIN_SPACING_ALONG_AXIS,
        help=(
            "mean exponential gap between continuous material-arc branch sites "
            "(canonical fixed value: 0.20; not a grid dimension)"
        ),
    )
    parser.add_argument(
        "--branch-retry-mode",
        choices=BRANCH_RETRY_MODES,
        default="single_trial",
        help=(
            "single_trial: one B.P. draw per new site; retry_open_sites: one "
            "B.P. hazard draw per open mature site per developmental step"
        ),
    )
    parser.add_argument("--branch-min-distance-from-tip", type=float, default=0.45)
    parser.add_argument("--branch-min-distance-from-base", type=float, default=0.25)
    parser.add_argument("--base-radius", type=float, default=0.05)
    parser.add_argument("--balloon-scale", type=float, default=0.05)
    parser.add_argument("--nutrient-sensitivity", type=float, default=5.0)
    parser.add_argument("--nutrient-sensing-distance", type=float, default=5.0)
    parser.add_argument("--soil-water-background", type=float, default=0.20)
    parser.add_argument("--rain-water-input", type=float, default=0.80)
    parser.add_argument("--water-infiltration-depth", type=float, default=6.0)
    parser.add_argument("--water-sensing-distance", type=float, default=4.0)
    parser.add_argument("--phosphorus-concentration", type=float, default=0.90)
    parser.add_argument("--nitrogen-concentration", type=float, default=0.80)
    parser.add_argument("--potassium-concentration", type=float, default=0.70)
    parser.add_argument(
        "--disable-resource-demand-feedback",
        action="store_true",
        help="turn off schema-v23 availability-aware water/P/N/K direction feedback",
    )
    parser.add_argument("--gravitropism-weight", type=float, default=1.15)
    parser.add_argument("--plagiotropism-weight", type=float, default=0.18)
    parser.add_argument("--upward-growth-penalty", type=float, default=2.75)
    parser.add_argument("--baseline-downward-bias", type=float, default=0.35)
    parser.add_argument("--low-resource-lateral-suppression", type=float, default=1.05)
    parser.add_argument("--resource-lateral-exploration-weight", type=float, default=0.42)
    parser.add_argument("--resource-signal-half-saturation", type=float, default=0.25)
    parser.add_argument("--tip-direction-persistence", type=float, default=1.95)
    parser.add_argument("--tip-max-bend-degrees", type=float, default=14.0)
    parser.add_argument("--lateral-radial-balance-weight", type=float, default=0.15)
    parser.add_argument("--soil-surface-z", type=float, default=0.0)
    parser.add_argument("--max-above-surface-tolerance", type=float, default=0.05)
    parser.add_argument("--above-surface-penalty", type=float, default=10.0)
    parser.add_argument(
        "--allow-above-surface-curves",
        action="store_true",
        help="disable the schema-v12 default that rejects sampled root curves above the soil surface",
    )
    parser.add_argument(
        "--compute-convex-hull", action="store_true",
        help="enable optional O(n log n) convex-hull volume calculation",
    )
    parser.add_argument(
        "--strict-lineage-collisions",
        action="store_true",
        help=(
            "enable the legacy global ancestor collision veto; the elastic default "
            "avoids whole-tree branching freezes"
        ),
    )

    parser.add_argument("--rain-probability", type=float, default=0.5)
    parser.add_argument("--branch-probability", type=float, default=0.1)
    parser.add_argument("--thickness-increment", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--outdir", default="./results")
    parser.add_argument("--cores", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--master-seed", type=int, default=20260617)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-stop", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pool-chunksize", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-tasks-per-worker", type=int, default=1_000)
    parser.add_argument(
        "--start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        steps=args.steps,
        max_sampled_points=args.max_sampled_points,
        target_architecture_size=args.target_architecture_size,
        target_axis_count=args.target_axis_count,
        max_growth_iterations=args.max_growth_iterations,
        interactive_safety_cap=args.interactive_safety_cap,
        segment_length=args.segment_length,
        anchor_initial_segment_length=args.anchor_initial_segment_length,
        anchor_min_segment_length=args.anchor_min_segment_length,
        anchor_decay_timescale=args.anchor_decay_timescale,
        branch_min_spacing_along_axis=args.branch_min_spacing_along_axis,
        branch_retry_mode=args.branch_retry_mode,
        branch_min_distance_from_tip=args.branch_min_distance_from_tip,
        branch_min_distance_from_base=args.branch_min_distance_from_base,
        base_radius=args.base_radius,
        balloon_scale=args.balloon_scale,
        angle_candidates=args.angle_candidates,
        nutrient_sensitivity=args.nutrient_sensitivity,
        nutrient_sensing_distance=args.nutrient_sensing_distance,
        soil_water_background=args.soil_water_background,
        rain_water_input=args.rain_water_input,
        water_infiltration_depth=args.water_infiltration_depth,
        water_sensing_distance=args.water_sensing_distance,
        phosphorus_concentration=args.phosphorus_concentration,
        nitrogen_concentration=args.nitrogen_concentration,
        potassium_concentration=args.potassium_concentration,
        enable_resource_demand_feedback=not args.disable_resource_demand_feedback,
        gravitropism_weight=args.gravitropism_weight,
        plagiotropism_weight=args.plagiotropism_weight,
        upward_growth_penalty=args.upward_growth_penalty,
        baseline_downward_bias=args.baseline_downward_bias,
        low_resource_lateral_suppression=args.low_resource_lateral_suppression,
        resource_lateral_exploration_weight=args.resource_lateral_exploration_weight,
        resource_signal_half_saturation=args.resource_signal_half_saturation,
        tip_direction_persistence=args.tip_direction_persistence,
        tip_max_bend_degrees=args.tip_max_bend_degrees,
        lateral_radial_balance_weight=args.lateral_radial_balance_weight,
        soil_surface_z=args.soil_surface_z,
        max_above_surface_tolerance=args.max_above_surface_tolerance,
        above_surface_penalty=args.above_surface_penalty,
        reject_above_surface_curves=not args.allow_above_surface_curves,
        compute_convex_hull=args.compute_convex_hull,
        max_nodes=args.max_nodes,
        max_seconds_per_simulation=args.max_seconds_per_simulation,
        strict_lineage_collisions=args.strict_lineage_collisions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cores < 1:
        parser.error("--cores must be at least 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if args.pool_chunksize < 1 or args.checkpoint_every < 1:
        parser.error("chunk and checkpoint sizes must be positive")

    config = config_from_args(args)
    config.validate()
    if args.mode == "batch":
        run_batch(args, config)
    elif args.mode == "smoke":
        run_smoke(config)
    else:
        run_single(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
