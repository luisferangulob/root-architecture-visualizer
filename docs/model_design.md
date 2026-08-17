# Model Design

This document records the scientific and numerical assumptions implemented by the schema-v26 root-architecture engine. It is descriptive documentation: the source code and regression fixtures remain the executable specification.

## Developmental Clock and Event Ordering

Developmental steps represent biological age. At the beginning of each completed step, the engine takes a snapshot of all active axes. Every axis in that snapshot receives exactly one bounded, curvature-limited extension attempt. Resource availability can influence the direction and length of an attempt, but it does not select which active tips are allowed to try growing.

Branch-site maturation and initiation follow extension attempts while using the same beginning-of-step snapshot. Accepted nodes are inserted immediately so later operations in the step encounter them during collision checks. Newly accepted lateral axes do not elongate or initiate branches until the next developmental step.

Sampled support-point limits are technical memory, collision-query, and rendering safeguards. They are not developmental targets. Reaching a point or time limit before the requested duration produces an incomplete-run status.

## Continuous Axes and Directional Growth

Each root axis is represented as a continuous material-arc curve sampled at support points. Short extensions use cubic Hermite interpolation between endpoints and tangents. Local orthonormal frames, cone sampling, and spherical interpolation generate bounded 3D direction changes without an explicit quaternion state.

Terminal direction selection combines:

- persistence from the previous tangent;
- gravitropic and plagiotropic components;
- local water, phosphorus, nitrogen, and potassium signals;
- a persistent stochastic resource focus;
- soil-surface rejection and upward-growth penalties;
- collision feasibility and maximum-bend constraints.

Candidate selection uses a Gumbel-max step, preserving stochastic preference while allowing gravity to dominate when no resource gradient supports lateral exploration. Rich local support relaxes gravitational preference and permits broader plagiotropic foraging. Upward excursions require positive resource gain and are limited to two consecutive accepted extensions.

The primary anchor follows a positive, smoothly decaying elongation law with a configurable minimum. Lateral meristem activity also decays with axis age while retaining a positive floor, so active tips continue to receive meaningful attempts at late developmental times.

## Branch Sites and Initiation

Branch sites form a seeded Poisson point process along newly mature material arc. The canonical mean spacing is `0.20`; spacing is an expected exponential gap, not a hard minimum or fixed-grid dimension.

Two retry modes are supported:

- `single_trial`: branch probability is evaluated once when a mature site first becomes eligible, after which the site closes.
- `retry_open_sites`: branch probability is a per-step hazard evaluated at each open mature site.

The configured branch probability controls lineage initiation. Rainfall, resource concentration, whole-plant demand, stochastic resource focus, starvation state, local primordium stimulus, collision state, and rendering never alter the probability threshold or its random draw.

Descendant initiation is attenuated by a fixed lineage-generation relationship. Once a probability trial passes, physical geometry determines whether a branch can emerge. Post-initiation extension length and transport-area growth never read branch probability.

Initiation draws use a counter-based SplitMix64 stream keyed by simulation seed, task index, axis lineage, site identifier, and trial number. This isolates initiation from environmental, direction, emergence, collision, and rendering random draws.

## Emergence Geometry and Self-Avoidance

Branch candidates are constructed relative to the parent tangent and a local perpendicular frame. Emergence angle follows local resource sufficiency and the axis's stochastic resource focus rather than a fixed angle.

Branch collars use clearance on the current cylindrical parent surface. For two origins at axial separation `Δz`, azimuthal separation `Δθ`, and parent radius `r`, the surface distance combines axial separation with the circumferential chord `2 r sin(Δθ / 2)`. Because `r` changes as the parent thickens, previously crowded circumferential space can become available later.

New lateral axes receive a short geometry-only escape corridor. This keeps their initial extensions visibly separated from the parent before normal direction and collision behavior resumes. The corridor does not depend on branch probability or resource state.

Curve insertions are checked transactionally against the spatial index. The primary anchor is explicitly exempt from root-root collision rejection so a dense crown cannot permanently block downward growth. Dynamic points added after the most recent spatial-tree rebuild are checked exactly.

Descendant axes are laid out again from their current parent surfaces once per iteration. This deterministic linear-time pass produces the model's balloon-like outward displacement while preserving parent-relative attachment.

## Environmental Resource Model

Water, phosphorus, nitrogen, and potassium are dimensionless availability indices evaluated in absolute soil coordinates anchored at `z = 0`.

Rain advances a smooth wetting-front pulse through a semi-infinite vertical profile; the deep tail does not become a uniform lower layer. Phosphorus remains shallow and comparatively immobile. Nitrate and potassium shift downward at distinct rain-dependent rates without rescaling the coordinate system.

Resource capture rates apply per unit exposed cylindrical surface area per iteration. Whole-plant demand is normalized by capture rate, gated by actual supply, smoothed over time, and distributed among persistent stochastic water, phosphorus, nitrogen, potassium, and balanced tip focuses.

Availability normalization retains absolute scale: a trace concentration such as `0.01` remains weak rather than becoming a full-strength signal merely because it is the only active resource. Under-capture can amplify an available gradient but cannot manufacture supply.

Starvation diagnostics use absolute local availability. Starvation never directly blocks branch initiation; it acts after initiation by favoring steeper, more confined growth so geometric crowding and collision can limit architecture naturally. Scarcity produces a visible shoulder near 10–35 degrees, while balanced rich support permits a broad distribution approaching horizontal growth.

## Thickness and Transport Area

Scientific radius is derived from cross-sectional area:

```text
radius = sqrt(cross_sectional_area / pi)
```

Growth records spatial range-addition events of the form `(end_material_arc, cross_sectional_area_increment)`. Each event applies from the axis base through its stored end coordinate. Events are sorted and reverse-cumulated per axis version so radius queries can use binary search.

An extension contributes area along its own proximal path. Descendant extension also contributes recursively attenuated area from each ancestor base to the descendant's attachment point. These additions vary radius continuously along every axis.

After attenuated additions, a child-to-root correction enforces the branch collar bound. The correction is cross-sectional area, never a direct radius increment, and applies only proximal to the attachment. Reverse ancestry processing makes the guarantee recursive without changing centerline topology.

## Post-Growth Diagnostics

Transient primordium stimulus can rank physical azimuths after a probability pass but cannot create sites or modify initiation probability. Contributions older than six stimulus lifetimes are below 0.25%, allowing reverse traversal to stop without scanning an axis's entire history.

Whorl detection is also post-growth only. It groups nearby first-order origins on the primary axis; higher-order descendants are excluded from the denominator so later branching does not change whether the original group is classified as whorl-like.

Reported outputs include topology, length, depth, width, convex-hull metrics when enabled, branching counts, generation and Horton-Strahler summaries, direction diagnostics, resource capture, radius profiles, stopping status, and profiling measurements.

## Reproducibility and Parameter Sweeps

All simulation random streams are deterministic functions of the master seed and task index. Fixed-grid task mapping covers:

```text
70 thickness increments
× 99 rain probabilities
× 99 branch probabilities
× 5 replicates
= 3,430,350 simulations
```

Branch-site spacing and retry mode are configuration choices but are not dimensions of that fixed grid. Local runs, sharded multiprocessing batches, and Slurm-array workers use the same deterministic task mapping.

The companion simulator repository checks invariants across initiation, geometry, resources, taper, checkpoint resume, storage, and HPC orchestration. Its compact schema-v26 fixture stores cryptographic hashes of the full deterministic scientific result record, geometry, topology, radii, axis metadata, and branch-site state for a representative case. This repository separately tests the rendering and application layer.

## HPC and Storage Semantics

Long runs can be executed as five-replicate Slurm arrays. Immutable manifests record configuration, task mapping, source hashes, and execution provenance. Checkpoints validate simulator identity, schema, configuration, seed, and task index before resume.

Result bundles store lossless NumPy arrays and metadata separately, enabling memory-mapped loading and rendering-only levels of detail. Level-of-detail selection never changes the stored scientific result.

The default X-disk location and allowed partition names are site-oriented defaults. `ROOT_HPC_RUNS_DIR` can redirect run storage, while cluster partition settings may require adaptation for another institution.
