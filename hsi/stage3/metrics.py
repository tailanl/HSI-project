"""Unchanged current numerical motion metrics; no lineage or release authority.

Official source identities must be explicitly supplied for the compatibility metrics.
The external implementation files themselves are not included in this package.
"""
from __future__ import annotations
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.ndimage import distance_transform_edt

FOOT_IDS = (10, 11)
ROLE_IDS = {
    "declared_contact": (0, 1, 2),
    "support_chain": (4, 5, 7, 8, 10, 11),
    "axial_posture": (3, 6, 9, 12, 15),
    "upper_limb": (13, 14, 16, 17, 18, 19, 20, 21),
}
FORBIDDEN_RESULT_FAMILIES = ("p373", "p520", "p522")

class EvaluationContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise EvaluationContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, *, label: str = "artifact") -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    reject_result_path(resolved, label)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def reject_result_path(path: Path, label: str) -> None:
    for part in Path(path).parts:
        normalized = re.sub(r"[^a-z0-9]+", "_", part.lower())
        tokens = set(normalized.split("_"))
        for family in FORBIDDEN_RESULT_FAMILIES:
            if family in tokens or normalized.startswith(family + "_"):
                raise EvaluationContractError(
                    f"{label} resolves through forbidden cached-result family {family.upper()}: {path}"
                )


def summarize(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    require(np.isfinite(data).all(), "metric input contains non-finite values")
    return {
        "count": int(data.size),
        "mean": None if not data.size else float(data.mean()),
        "p95": None if not data.size else float(np.quantile(data, 0.95)),
        "maximum": None if not data.size else float(data.max()),
        "minimum": None if not data.size else float(data.min()),
    }


def heading_yaw(joints: np.ndarray) -> np.ndarray:
    right = joints[:, 2] - joints[:, 1]
    forward = np.stack((-right[:, 1], right[:, 0]), axis=-1)
    norm = np.linalg.norm(forward, axis=-1)
    require(bool(np.all(norm > 1e-8)), "motion has a degenerate hip heading")
    return np.arctan2(forward[:, 1], forward[:, 0])


def angular_error(values: np.ndarray, target: float) -> np.ndarray:
    delta = np.asarray(values) - float(target)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def terminal_metrics(
    joints: np.ndarray, packet: Mapping[str, Any], *, tail_frames: int
) -> dict[str, Any]:
    terminal = packet["nodes"][-1]
    payload = terminal["full_smplx_keypose"]
    target_joints = np.asarray(payload["joints_world_zup"], dtype=np.float64)
    target_root = np.asarray(payload["root_xyz_yaw"], dtype=np.float64)
    contacts = tuple(int(value) for value in terminal.get("contact_joint_ids", ()))
    require(contacts and len(set(contacts)) == len(contacts),
            "terminal contact joint IDs are missing/duplicated")
    tail = min(int(tail_frames), len(joints))
    error = np.linalg.norm(joints[-tail:] - target_joints[None], axis=-1)
    yaws = heading_yaw(joints)
    root_xy = np.linalg.norm(joints[-tail:, 0, :2] - target_root[None, :2], axis=-1)
    root_z = np.abs(joints[-tail:, 0, 2] - target_root[2])
    root_yaw = angular_error(yaws[-tail:], target_root[3])
    roles: dict[str, Any] = {}
    for name, default_ids in ROLE_IDS.items():
        ids = contacts if name == "declared_contact" else default_ids
        values = error[:, list(ids)]
        roles[name] = {
            "joint_ids": list(ids),
            "tail_error_m": summarize(values),
            "terminal_error_m": summarize(values[-1]),
        }
    return {
        "tail_frames": tail,
        "target_root_xyz_yaw": target_root.astype(float).tolist(),
        "terminal_root_xyz_yaw": [
            *joints[-1, 0].astype(float).tolist(), float(yaws[-1])
        ],
        "root_xy_error_m": summarize(root_xy),
        "root_z_error_m": summarize(root_z),
        "root_yaw_error_deg": summarize(np.degrees(root_yaw)),
        "full22_tail_error_m": summarize(error),
        "full22_terminal_error_m": summarize(error[-1]),
        "rolewise": roles,
    }


def point_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    line = np.asarray(polyline, dtype=np.float64)
    require(line.ndim == 2 and line.shape[1] == 2 and len(line) >= 2,
            "Stage3 route polyline needs at least two XY points")
    starts, vectors = line[:-1], np.diff(line, axis=0)
    lengths2 = np.sum(vectors * vectors, axis=1)
    distances = []
    for point in points:
        alpha = np.sum((point - starts) * vectors, axis=1) / np.maximum(lengths2, 1e-12)
        projections = starts + np.clip(alpha, 0.0, 1.0)[:, None] * vectors
        distances.append(float(np.linalg.norm(projections - point, axis=1).min()))
    return np.asarray(distances, dtype=np.float64)


def terminal_phase_start(
    metadata: Mapping[str, Any], packet: Mapping[str, Any], total_frames: int
) -> tuple[int, dict[str, Any]]:
    terminal_ordinal = int(packet["nodes"][-1]["ordinal"])
    candidates: list[tuple[int, int]] = []
    for row in metadata.get("primitive_records", []):
        if not isinstance(row, Mapping) or row.get("active_external_ordinal") != terminal_ordinal:
            continue
        receipt = row.get("e226_receipt")
        if not isinstance(receipt, Mapping):
            continue
        generated = receipt.get("generated_frames_in_prefix")
        primitive = row.get("primitive_id")
        if isinstance(generated, int) and generated >= 0 and isinstance(primitive, int):
            candidates.append((generated, primitive))
    prefix = metadata.get("p478_causal_prefix", {})
    initial = prefix.get("initial_history_frames", 0) if isinstance(prefix, Mapping) else 0
    if candidates and isinstance(initial, int) and initial >= 0:
        generated, primitive = min(candidates)
        frame = min(total_frames, initial + generated)
        return frame, {
            "source": "first_terminal_active_primitive_generated_prefix",
            "terminal_external_ordinal": terminal_ordinal,
            "first_terminal_primitive_id": primitive,
            "initial_history_frames": initial,
            "generated_prefix_frames_before_terminal": generated,
            "terminal_start_frame_inclusive": frame,
        }
    return total_frames, {
        "source": "metadata_phase_split_unavailable_all_frames_route_audited",
        "terminal_external_ordinal": terminal_ordinal,
        "terminal_start_frame_inclusive": total_frames,
    }


def route_metrics(
    joints: np.ndarray, packet: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    route_nodes = [
        np.asarray(node["root_xyz_yaw"], dtype=np.float64)[:2]
        for node in packet["nodes"][:-1]
        if not bool(node.get("orientation_only", False))
    ]
    route = np.stack(route_nodes)
    terminal_start, split = terminal_phase_start(metadata, packet, len(joints))
    navigation_stop = max(1, terminal_start)
    navigation_xy = joints[:navigation_stop, 0, :2]
    distances = point_polyline_distance(navigation_xy, route)
    start_distance = np.linalg.norm(navigation_xy - route[0][None], axis=1)
    goal_distance = np.linalg.norm(navigation_xy - route[-1][None], axis=1)
    contact_transition = np.linalg.norm(
        joints[terminal_start:, 0, :2] - route[-1][None], axis=1
    ) if terminal_start < len(joints) else np.asarray([], dtype=np.float64)
    return {
        "route_node_count": int(len(route)),
        "navigation_frame_count": int(navigation_stop),
        "phase_split": split,
        "root_to_route_distance_m": summarize(distances),
        "initial_root_to_route_start_m": float(start_distance[0]),
        "minimum_root_to_route_goal_m": float(goal_distance.min()),
        "navigation_end_root_to_route_goal_m": float(goal_distance[-1]),
        "net_route_progress_m": float(goal_distance[0] - goal_distance[-1]),
        "post_navigation_contact_transition_from_route_goal_m": summarize(contact_transition),
    }


def motion_quality(joints: np.ndarray, *, fps: float, foot_height: float, official_sources: Mapping[str, Path]) -> dict[str, Any]:
    feet = joints[:, list(FOOT_IDS)]
    displacement = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)
    contacts = ((feet[:-1, :, 2] <= foot_height)
                & (feet[1:, :, 2] <= foot_height))
    slide = displacement[contacts] * fps
    acceleration = np.diff(joints, n=2, axis=0) * fps ** 2
    jerk = np.diff(joints, n=3, axis=0) * fps ** 3
    official = official_remogen_motion_quality(joints, sources=official_sources)
    return {
        "fps": float(fps),
        "foot_joint_ids": list(FOOT_IDS),
        "foot_contact_height_m": float(foot_height),
        "contact_foot_sample_count": int(contacts.sum()),
        "contact_foot_slip_m_per_s": summarize(slide),
        "joint_acceleration_m_per_s2": summarize(np.linalg.norm(acceleration, axis=-1)),
        "joint_jerk_m_per_s3": summarize(np.linalg.norm(jerk, axis=-1)),
        "joint_jerk_rms_m_per_s3": (
            None if not jerk.size else float(np.sqrt(np.mean(jerk ** 2)))
        ),
        "official_remogen_compatible": official,
    }


def official_remogen_motion_quality(joints: np.ndarray, *, sources: Mapping[str, Path]) -> dict[str, Any]:
    """Exact numpy transcription of the official evaluated scalars.

    ReMoGen's LINGO evaluator calls ``calc_skate(pred)`` and
    ``calc_jerk(pred).max()`` from ``metrics/metrics_unified.py``.  It does not
    call the separately defined ``calculate_skating_ratio`` helper.
    """

    world = np.asarray(joints, dtype=np.float64)

    def skate_terms(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feet = value[:, list(FOOT_IDS)]
        heights = feet[:, :, 2]
        displacement_3d = np.linalg.norm(np.diff(feet, axis=0), axis=-1)
        consecutive_max_height = np.maximum(heights[:-1], heights[1:])
        height_ratio = np.clip(consecutive_max_height / 0.03, 0.0, 1.0)
        contact_weight = 2.0 - np.power(2.0, height_ratio)
        return displacement_3d * contact_weight, contact_weight, heights

    raw_weighted, raw_weight, world_heights = skate_terms(world)
    # The official driver defaults to enable_normalize=True and subtracts the
    # first GT pelvis XYZ before calling calc_skate. Query-only P523 has no GT;
    # its exact non-oracle proxy uses the fresh generated frame-0 pelvis, which
    # is the query-static start bound by the packet.
    normalized = world - world[0, 0][None, None, :]
    driver_weighted, driver_weight, _driver_heights = skate_terms(normalized)
    # Official calc_jerk: third finite difference, L1 over XYZ, maximum joint.
    third = np.diff(world, n=3, axis=0)
    jerk_per_frame = np.max(np.sum(np.abs(third), axis=-1), axis=-1)
    consecutive_contact = ((world_heights[:-1] < 0.03)
                           & (world_heights[1:] < 0.03))
    return {
        "definition": "exact_calc_skate_and_calc_jerk_used_by_official_lingo_evaluator",
        "source": {
            "metrics_unified": artifact(sources["metrics_unified"], label="official metric code"),
            "lingo_evaluation_driver": artifact(
                sources["lingo_evaluation_driver"], label="official evaluation driver"
            ),
        },
        "foot_joint_ids": list(FOOT_IDS),
        "foot_floor_threshold_m": 0.03,
        "foot_skate_m_per_frame": float(driver_weighted.mean()),
        "foot_skate_weighted_displacement_m_per_frame": summarize(
            driver_weighted
        ),
        "driver_normalization": {
            "official_driver_default": "subtract_GT_frame0_pelvis_XYZ",
            "p523_query_only_proxy": "subtract_fresh_generated_frame0_pelvis_XYZ",
            "gt_or_future_motion_read": False,
            "proxy_equals_official_when_query_start_matches_GT_start": True,
        },
        "driver_normalized_foot_contact_weight": summarize(driver_weight),
        "raw_world_calc_skate_m_per_frame": float(raw_weighted.mean()),
        "raw_world_foot_contact_weight": summarize(raw_weight),
        "world_floor_foot_contact_consecutive_pair_fraction": float(
            consecutive_contact.mean()
        ),
        "world_floor_any_foot_contact_consecutive_pair_fraction": float(
            consecutive_contact.any(axis=1).mean()
        ),
        "world_floor_terminal_any_foot_below_3cm": bool(
            np.any(world_heights[-1] < 0.03)
        ),
        "jerk_l1_max_joint_per_frame_m": summarize(jerk_per_frame),
        "jerk_sequence_max_m_per_frame3": float(jerk_per_frame.max()),
        "fps_scaling_applied": False,
        "ground_truth_difference_or_AUJ_computed": False,
        "higher_or_lower": {
            "foot_skate_m_per_frame": "lower_is_better_but_zero_is_vacuous_without_contact",
            "world_floor_foot_contact_fraction": "higher_is_more_floor_contact_not_an_official_quality_scalar",
            "jerk_sequence_max_m_per_frame3": "lower_is_better",
        },
    }


def occupancy_to_zup(raw: np.ndarray, layout: str) -> tuple[np.ndarray, str, str]:
    value = np.asarray(raw)
    require(value.ndim == 3, "raw occupancy must be 3D")
    resolved = str(layout)
    if resolved == "auto":
        if value.shape == (300, 100, 400) or int(np.argmin(value.shape)) == 1:
            resolved = "lingo_yup"
        elif value.shape == (300, 400, 100) or int(np.argmin(value.shape)) == 2:
            resolved = "xyz_zup"
        else:
            raise EvaluationContractError("cannot infer occupancy layout")
    if resolved == "lingo_yup":
        value = np.transpose(value, (0, 2, 1))[:, ::-1, :]
        conversion = "transpose_0_2_1_then_flip_new_y"
    elif resolved == "xyz_zup":
        conversion = "identity_xyz_zup"
    else:
        raise EvaluationContractError("occupancy layout must be auto/lingo_yup/xyz_zup")
    if np.issubdtype(value.dtype, np.floating):
        require(np.isfinite(value).all(), "occupancy contains non-finite values")
        value = value > 0.5
    else:
        value = value.astype(bool, copy=False)
    return np.ascontiguousarray(value), resolved, conversion


def conservative_pool(value: np.ndarray, factor: int) -> np.ndarray:
    require(int(factor) >= 1, "SDF downsample must be positive")
    if factor == 1:
        return np.ascontiguousarray(value)
    shape = np.asarray(value.shape)
    out = (shape + factor - 1) // factor
    padded_shape = out * factor
    padded = np.pad(value, tuple((0, int(b - a)) for a, b in zip(shape, padded_shape)))
    return np.ascontiguousarray(padded.reshape(
        int(out[0]), factor, int(out[1]), factor, int(out[2]), factor
    ).any(axis=(1, 3, 5)))


def build_signed_sdf(
    occupancy: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    spacing = (upper - lower) / np.asarray(occupancy.shape, dtype=np.float64)
    outside = distance_transform_edt(~occupancy, sampling=spacing)
    inside = distance_transform_edt(occupancy, sampling=spacing)
    return np.asarray(outside - inside, dtype=np.float32), spacing


def sample_grid(
    grid: np.ndarray, points: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    shape = np.asarray(grid.shape, dtype=np.int64)
    coords = (flat - lower) / (upper - lower) * shape
    indices = np.floor(coords).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < shape), axis=1)
    clipped = np.clip(indices, 0, shape - 1)
    values = grid[clipped[:, 0], clipped[:, 1], clipped[:, 2]].astype(np.float64)
    values[~inside] = np.nan
    return values.reshape(points.shape[:-1]), inside.reshape(points.shape[:-1])


def collision_metrics(
    joints: np.ndarray,
    raw_sdf: np.ndarray,
    structural_sdf: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    contacts: Sequence[int],
    tail_frames: int,
    terminal_start: int,
    shallow_floor_band_m: float,
    deep_floor_penetration_m: float,
) -> dict[str, Any]:
    values, bounds = sample_grid(raw_sdf, joints, lower, upper)
    structural_values, structural_bounds = sample_grid(
        structural_sdf, joints, lower, upper
    )
    allowed = np.zeros_like(bounds)
    tail = min(int(tail_frames), len(joints))
    allowed[-tail:, list(contacts)] = True
    forbidden = bounds & ~allowed
    collision = forbidden & (values < 0)
    depths = -values[collision]
    structural_forbidden = structural_bounds & ~allowed
    structural_collision = structural_forbidden & (structural_values < 0)
    structural_depths = -structural_values[structural_collision]
    split = max(0, min(int(terminal_start), len(joints)))

    def phase(start: int, stop: int) -> dict[str, Any]:
        phase_forbidden = forbidden[start:stop]
        phase_collision = collision[start:stop]
        return {
            "frames": int(stop - start),
            "forbidden_joint_frame_count": int(phase_forbidden.sum()),
            "forbidden_collision_count": int(phase_collision.sum()),
            "forbidden_collision_fraction": (
                None if not phase_forbidden.any()
                else float(phase_collision.sum() / phase_forbidden.sum())
            ),
            "colliding_frame_count": int(np.any(phase_collision, axis=1).sum()),
        }

    def structural_phase(start: int, stop: int) -> dict[str, Any]:
        phase_forbidden = structural_forbidden[start:stop]
        phase_collision = structural_collision[start:stop]
        return {
            "frames": int(stop - start),
            "forbidden_joint_frame_count": int(phase_forbidden.sum()),
            "forbidden_collision_count": int(phase_collision.sum()),
            "forbidden_collision_fraction": (
                None if not phase_forbidden.any()
                else float(phase_collision.sum() / phase_forbidden.sum())
            ),
            "colliding_frame_count": int(np.any(phase_collision, axis=1).sum()),
        }

    foot_z = joints[:, list(FOOT_IDS), 2]
    shallow = ((foot_z >= -float(deep_floor_penetration_m))
               & (foot_z <= float(shallow_floor_band_m)))
    deep = foot_z < -float(deep_floor_penetration_m)
    deep_depth = np.maximum(-foot_z[deep], 0.0)
    return {
        "scope": "all_22_world_joints_target_contact_exempt_terminal_tail_only",
        "target_component_kept_in_scene_sdf": True,
        "in_bounds_fraction": float(bounds.mean()),
        "forbidden_joint_frame_count": int(forbidden.sum()),
        "forbidden_collision_count": int(collision.sum()),
        "forbidden_collision_fraction": (
            None if not forbidden.any() else float(collision.sum() / forbidden.sum())
        ),
        "forbidden_penetration_depth_m": summarize(depths),
        "colliding_frame_count": int(np.any(collision, axis=1).sum()),
        "navigation": phase(0, split),
        "terminal_transition": phase(split, len(joints)),
        "decomposition": {
            "policy": (
                "raw occupancy collision is retained; release collision excludes "
                "the floor slab and reports shallow/deep foot-floor geometry separately"
            ),
            "shallow_floor_contact": {
                "foot_joint_ids": list(FOOT_IDS),
                "z_band_world_m": [
                    -float(deep_floor_penetration_m), float(shallow_floor_band_m)
                ],
                "foot_frame_count": int(shallow.sum()),
                "foot_frame_fraction": float(shallow.mean()),
                "classification": "allowed_floor_contact_not_furniture_or_wall_collision",
            },
            "deep_floor_penetration": {
                "threshold_below_floor_m": float(deep_floor_penetration_m),
                "foot_frame_count": int(deep.sum()),
                "foot_frame_fraction": float(deep.mean()),
                "penetration_depth_m": summarize(deep_depth),
                "classification": "physical_failure",
            },
            "structural_nonfloor_collision": {
                "scope": "furniture_wall_and_nonfloor_occupancy",
                "forbidden_joint_frame_count": int(structural_forbidden.sum()),
                "forbidden_collision_count": int(structural_collision.sum()),
                "forbidden_collision_fraction": (
                    None if not structural_forbidden.any()
                    else float(structural_collision.sum() / structural_forbidden.sum())
                ),
                "forbidden_penetration_depth_m": summarize(structural_depths),
                "colliding_frame_count": int(np.any(structural_collision, axis=1).sum()),
                "navigation": structural_phase(0, split),
                "terminal_transition": structural_phase(split, len(joints)),
            },
        },
    }


def fullmesh_metrics(
    path: Path | None,
    raw_sdf: np.ndarray,
    structural_sdf: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    stage2: Mapping[str, np.ndarray],
    tail_frames: int,
    terminal_start: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if path is None:
        return ({"available": False, "reason": "no_P519_skinned_vertex_archive"}, None)
    reject_result_path(path, "P519 skinned vertices")
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        require("vertices_world" in archive.files, "P519 archive lacks vertices_world")
        vertices = np.asarray(archive["vertices_world"], dtype=np.float32)
    require(vertices.ndim == 3 and vertices.shape[1:] == (10475, 3),
            "P519 vertices must be [T,10475,3]")
    butt_ids = np.asarray(stage2.get("fixed_generated_butt_vertex_indices", []), dtype=np.int64)
    require(butt_ids.size > 0 and int(butt_ids.min()) >= 0
            and int(butt_ids.max()) < 10475, "Stage2 lacks valid fixed butt vertex IDs")
    values, bounds = sample_grid(raw_sdf, vertices, lower, upper)
    structural_values, structural_bounds = sample_grid(
        structural_sdf, vertices, lower, upper
    )
    allowed = np.zeros_like(bounds)
    tail = min(int(tail_frames), len(vertices))
    allowed[-tail:, butt_ids] = True
    forbidden = bounds & ~allowed
    collision = forbidden & (values < 0)
    depth = -values[collision]
    structural_forbidden = structural_bounds & ~allowed
    structural_collision = structural_forbidden & (structural_values < 0)
    structural_depth = -structural_values[structural_collision]
    split = max(0, min(int(terminal_start), len(vertices)))

    def phase(start: int, stop: int) -> dict[str, Any]:
        phase_forbidden = forbidden[start:stop]
        phase_collision = collision[start:stop]
        return {
            "frames": int(stop - start),
            "forbidden_vertex_frame_count": int(phase_forbidden.sum()),
            "forbidden_collision_count": int(phase_collision.sum()),
            "forbidden_collision_fraction": (
                None if not phase_forbidden.any()
                else float(phase_collision.sum() / phase_forbidden.sum())
            ),
            "colliding_frame_count": int(np.any(phase_collision, axis=1).sum()),
        }
    return ({
        "available": True,
        "scope": "all_10475_vertices_target_butt_contact_exempt_terminal_tail_only",
        "target_component_kept_in_scene_sdf": True,
        "frame_count": int(len(vertices)),
        "vertex_count": 10475,
        "forbidden_vertex_frame_count": int(forbidden.sum()),
        "forbidden_collision_count": int(collision.sum()),
        "forbidden_collision_fraction": float(collision.sum() / max(1, forbidden.sum())),
        "forbidden_penetration_depth_m": summarize(depth),
        "colliding_frame_count": int(np.any(collision, axis=1).sum()),
        "stage2_butt_contact_vertex_count": int(len(butt_ids)),
        "navigation": phase(0, split),
        "terminal_transition": phase(split, len(vertices)),
        "structural_nonfloor": {
            "forbidden_vertex_frame_count": int(structural_forbidden.sum()),
            "forbidden_collision_count": int(structural_collision.sum()),
            "forbidden_collision_fraction": float(
                structural_collision.sum() / max(1, structural_forbidden.sum())
            ),
            "forbidden_penetration_depth_m": summarize(structural_depth),
            "colliding_frame_count": int(np.any(structural_collision, axis=1).sum()),
        },
    }, artifact(path, label="P519 skinned vertices"))

