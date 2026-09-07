"""Strict atomic fusion: complete-link, consensus vertices, final exclusivity.

Numerical thresholds and support identities preserve the existing method.
This module does not load a model, execute another source file, or read a task.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from . import sam_geometry as sam
Observation = sam.Observation
SceneInstanceError = sam.SceneInstanceError
OUTPUT_SCHEMA = "p515.lingo_sam31_atomic_multiview_instances.v1"
READY_STATUS = "atomic_instances_ready"
@dataclass(frozen=True)
class AtomicFusionThresholds:
    """Preregistered conservative gates for one physical instance."""

    # P506 exact shared-original-OBJ-vertex rule.
    strong_exact_vertex_minimum: int = 80
    strong_exact_minimum_support_fraction: float = 0.16
    strong_centroid_distance_max_m: float = 0.62

    # A non-strong pair may be compatible, but can never initiate a merge.
    compatible_exact_vertex_minimum: int = 48
    compatible_exact_minimum_support_fraction: float = 0.08
    compatible_centroid_distance_max_m: float = 0.62
    compatible_surface_near_fraction_minimum: float = 0.08
    compatible_surface_aabb_fraction_minimum: float = 0.12
    compatible_surface_centroid_distance_max_m: float = 0.38
    compatible_enclosed_aabb_fraction_minimum: float = 0.35
    compatible_enclosed_centroid_distance_max_m: float = 0.25
    compatible_enclosed_one_sided_near_fraction_minimum: float = 0.12

    # Component-level complete/average-link and purity rules.
    component_pair_compatibility_purity_minimum: float = 1.0
    component_average_compatibility_score_minimum: float = 0.58
    component_strong_pair_density_minimum: float = 0.50
    component_centroid_diameter_max_m: float = 0.62
    component_member_strong_incidence_minimum: float = 1.0
    component_view_purity_minimum: float = 1.0

DEFAULT_THRESHOLDS = AtomicFusionThresholds()
SAME_VIEW_3D_CONTAINMENT_MINIMUM = 0.70
FINAL_ATOM_OVERLAP_MINIMUM_FRACTION_MAX = 0.25

def _require_observation_support(row: Observation) -> None:
    support = np.asarray(row.support)
    sam.require(support.ndim == 1 and len(support) > 0, f"{row.observation_id} has empty support")
    sam.require(np.all(support[:-1] < support[1:]), f"{row.observation_id} support is not sorted unique")


def _support_sha256(support: np.ndarray) -> str:
    """Hash a sorted support set in an architecture-independent encoding."""

    canonical = np.asarray(support, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _component_supports(
    members: tuple[int, ...], observations: list[Observation]
) -> tuple[np.ndarray, np.ndarray]:
    """Return union support and support seen from at least two distinct views."""

    if not members:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    view_vertex_pairs = np.concatenate(
        [
            np.column_stack(
                (
                    np.full(len(observations[index].support), observations[index].view_index, dtype=np.int64),
                    np.asarray(observations[index].support, dtype=np.int64),
                )
            )
            for index in members
        ],
        axis=0,
    )
    # An observation is already sorted/unique, but this also protects the
    # distinct-view count if multiple masks from one view ever reach this gate.
    unique_pairs = np.unique(view_vertex_pairs, axis=0)
    union_support, distinct_view_counts = np.unique(
        unique_pairs[:, 1], return_counts=True
    )
    consensus_support = union_support[distinct_view_counts >= 2]
    return (
        np.asarray(union_support, dtype=np.int64),
        np.asarray(consensus_support, dtype=np.int64),
    )


def _same_view_3d_deduplicate(
    observations: list[Observation],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Greedy deterministic 3-D NMS within each camera view."""

    active_indices: list[int] = []
    suppressions: list[dict[str, Any]] = []
    view_indices = sorted({row.view_index for row in observations})
    for view_index in view_indices:
        ranked = sorted(
            (index for index, row in enumerate(observations) if row.view_index == view_index),
            key=lambda index: (
                -len(observations[index].support),
                -float(observations[index].quality),
                observations[index].observation_id,
                index,
            ),
        )
        kept_in_view: list[int] = []
        for candidate_index in ranked:
            candidate = observations[candidate_index]
            conflict: tuple[int, int, float, float] | None = None
            for kept_index in kept_in_view:
                kept = observations[kept_index]
                exact_count = int(
                    np.intersect1d(candidate.support, kept.support, assume_unique=True).size
                )
                minimum_count = min(len(candidate.support), len(kept.support))
                containment = exact_count / max(minimum_count, 1)
                union_count = len(candidate.support) + len(kept.support) - exact_count
                jaccard = exact_count / max(union_count, 1)
                if containment >= SAME_VIEW_3D_CONTAINMENT_MINIMUM:
                    conflict = (kept_index, exact_count, containment, jaccard)
                    break
            if conflict is None:
                kept_in_view.append(candidate_index)
                active_indices.append(candidate_index)
                continue
            kept_index, exact_count, containment, jaccard = conflict
            suppressions.append(
                {
                    "view_index": int(view_index),
                    "kept_observation_id": observations[kept_index].observation_id,
                    "suppressed_observation_id": candidate.observation_id,
                    "exact_vertex_count": exact_count,
                    "minimum_support_fraction": float(containment),
                    "jaccard": float(jaccard),
                    "threshold": SAME_VIEW_3D_CONTAINMENT_MINIMUM,
                    "reason": "same_view_3d_support_containment_at_least_0p70",
                }
            )
    return sorted(active_indices), suppressions


def _compatibility_score(audit: Mapping[str, Any], thresholds: AtomicFusionThresholds) -> float:
    """Bounded evidence score; it does not by itself create an edge."""

    exact_score = min(
        float(audit["exact_minimum_support_fraction"])
        / thresholds.strong_exact_minimum_support_fraction,
        1.0,
    )
    near_score = min(
        min(
            float(audit["left_near_fraction_0p055m"]),
            float(audit["right_near_fraction_0p055m"]),
        )
        / thresholds.compatible_surface_near_fraction_minimum,
        1.0,
    )
    aabb_score = min(
        float(audit["padded_aabb_intersection_fraction"])
        / thresholds.compatible_surface_aabb_fraction_minimum,
        1.0,
    )
    centroid_score = max(
        0.0,
        1.0
        - float(audit["centroid_distance_m"])
        / thresholds.compatible_centroid_distance_max_m,
    )
    return float(0.45 * exact_score + 0.20 * near_score + 0.20 * aabb_score + 0.15 * centroid_score)


def atomic_pair_audit(
    left: Observation,
    right: Observation,
    vertices: np.ndarray,
    thresholds: AtomicFusionThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Audit whether two masks are a strong edge or merely compatible."""

    _require_observation_support(left)
    _require_observation_support(right)
    base = dict(sam.pair_audit(left, right, vertices))
    exact_count = int(base["exact_vertex_count"])
    exact_fraction = float(base["exact_minimum_support_fraction"])
    centroid_distance = float(base["centroid_distance_m"])
    min_near = min(
        float(base["left_near_fraction_0p055m"]),
        float(base["right_near_fraction_0p055m"]),
    )
    max_near = max(
        float(base["left_near_fraction_0p055m"]),
        float(base["right_near_fraction_0p055m"]),
    )
    aabb_fraction = float(base["padded_aabb_intersection_fraction"])
    distinct_views = left.view_index != right.view_index

    strong = bool(
        distinct_views
        and exact_count >= thresholds.strong_exact_vertex_minimum
        and exact_fraction >= thresholds.strong_exact_minimum_support_fraction
        and centroid_distance <= thresholds.strong_centroid_distance_max_m
    )
    exact_compatible = bool(
        exact_count >= thresholds.compatible_exact_vertex_minimum
        and exact_fraction >= thresholds.compatible_exact_minimum_support_fraction
        and centroid_distance <= thresholds.compatible_centroid_distance_max_m
    )
    surface_compatible = bool(
        min_near >= thresholds.compatible_surface_near_fraction_minimum
        and aabb_fraction >= thresholds.compatible_surface_aabb_fraction_minimum
        and centroid_distance <= thresholds.compatible_surface_centroid_distance_max_m
    )
    enclosed_compatible = bool(
        aabb_fraction >= thresholds.compatible_enclosed_aabb_fraction_minimum
        and centroid_distance <= thresholds.compatible_enclosed_centroid_distance_max_m
        and max_near >= thresholds.compatible_enclosed_one_sided_near_fraction_minimum
    )
    compatible = bool(
        distinct_views and (strong or exact_compatible or surface_compatible or enclosed_compatible)
    )
    base.update(
        {
            # Override the legacy permissive merge flag with the P515 strong edge.
            "legacy_p508_merge_gate_pass": bool(base["merge_gate_pass"]),
            "merge_gate_pass": strong,
            "strong_pair_edge_pass": strong,
            "component_compatible_pair_pass": compatible,
            "distinct_view_pair": distinct_views,
            "exact_compatible_gate_pass": exact_compatible,
            "surface_compatible_gate_pass": surface_compatible,
            "enclosed_compatible_gate_pass": enclosed_compatible,
        }
    )
    base["compatibility_score"] = _compatibility_score(base, thresholds) if distinct_views else 0.0
    return base


def _component_audit(
    members: tuple[int, ...],
    observations: list[Observation],
    pair_audits: Mapping[tuple[int, int], Mapping[str, Any]],
    vertices: np.ndarray,
    thresholds: AtomicFusionThresholds,
) -> dict[str, Any]:
    pairs = [(members[a], members[b]) for a in range(len(members)) for b in range(a + 1, len(members))]
    rows = [pair_audits[(min(left, right), max(left, right))] for left, right in pairs]
    compatible_count = sum(bool(row["component_compatible_pair_pass"]) for row in rows)
    strong_count = sum(bool(row["strong_pair_edge_pass"]) for row in rows)
    pair_count = len(rows)
    compatibility_purity = compatible_count / max(pair_count, 1)
    strong_density = strong_count / max(pair_count, 1)
    average_score = float(np.mean([float(row["compatibility_score"]) for row in rows])) if rows else 0.0
    diameter = max((float(row["centroid_distance_m"]) for row in rows), default=0.0)
    view_count = len({observations[index].view_index for index in members})
    view_purity = view_count / max(len(members), 1)

    strong_incident: set[int] = set()
    for (left, right), row in zip(pairs, rows):
        if bool(row["strong_pair_edge_pass"]):
            strong_incident.update((left, right))
    member_strong_incidence = len(strong_incident) / max(len(members), 1)

    union_support, consensus_support = _component_supports(members, observations)
    union_xyz = vertices[union_support]
    union_centre = np.median(union_xyz, axis=0)
    consensus_xyz = vertices[consensus_support] if len(consensus_support) else None
    consensus_centre = (
        np.median(consensus_xyz, axis=0) if consensus_xyz is not None else None
    )
    passed = bool(
        len(members) >= 2
        and view_count >= 2
        and len(consensus_support) >= thresholds.strong_exact_vertex_minimum
        and compatibility_purity >= thresholds.component_pair_compatibility_purity_minimum
        and average_score >= thresholds.component_average_compatibility_score_minimum
        and strong_density >= thresholds.component_strong_pair_density_minimum
        and diameter <= thresholds.component_centroid_diameter_max_m
        and member_strong_incidence >= thresholds.component_member_strong_incidence_minimum
        and view_purity >= thresholds.component_view_purity_minimum
    )
    failures: list[str] = []
    if view_count < 2:
        failures.append("fewer_than_two_distinct_views")
    if len(consensus_support) < thresholds.strong_exact_vertex_minimum:
        failures.append("consensus_support_below_strong_exact_vertex_minimum")
    if view_purity < thresholds.component_view_purity_minimum:
        failures.append("same_view_mask_conflict")
    if compatibility_purity < thresholds.component_pair_compatibility_purity_minimum:
        failures.append("complete_link_pair_incompatibility")
    if average_score < thresholds.component_average_compatibility_score_minimum:
        failures.append("average_link_score_below_threshold")
    if strong_density < thresholds.component_strong_pair_density_minimum:
        failures.append("strong_pair_density_below_threshold")
    if member_strong_incidence < thresholds.component_member_strong_incidence_minimum:
        failures.append("member_without_strong_edge")
    if diameter > thresholds.component_centroid_diameter_max_m:
        failures.append("component_centroid_diameter_exceeded")
    return {
        "member_indices": list(members),
        "member_observation_ids": [observations[index].observation_id for index in members],
        "visible_view_indices": sorted({observations[index].view_index for index in members}),
        "observation_count": len(members),
        "pair_count": pair_count,
        "compatible_pair_count": compatible_count,
        "strong_pair_count": strong_count,
        "pair_compatibility_purity": compatibility_purity,
        "average_compatibility_score": average_score,
        "strong_pair_density": strong_density,
        "member_strong_incidence": member_strong_incidence,
        "view_purity": view_purity,
        "centroid_diameter_m": diameter,
        # The union remains audit evidence only.  Publication uses exclusively
        # vertices independently observed by at least two distinct views.
        "support_publication_policy": "vertices_observed_by_at_least_two_distinct_views",
        "minimum_distinct_views_per_published_vertex": 2,
        "union_support_vertex_count": int(len(union_support)),
        "union_support_vertex_ids_little_endian_int64_sha256": _support_sha256(union_support),
        "consensus_support_vertex_count": int(len(consensus_support)),
        "consensus_support_vertex_ids_little_endian_int64_sha256": _support_sha256(consensus_support),
        "union_only_vertex_count": int(len(union_support) - len(consensus_support)),
        "consensus_fraction_of_union": float(
            len(consensus_support) / max(len(union_support), 1)
        ),
        "union_centroid_world_zup_m": union_centre.tolist(),
        "union_bounds_world_zup_m": [
            union_xyz.min(axis=0).tolist(),
            union_xyz.max(axis=0).tolist(),
        ],
        "consensus_centroid_world_zup_m": (
            consensus_centre.tolist() if consensus_centre is not None else None
        ),
        "consensus_bounds_world_zup_m": (
            [consensus_xyz.min(axis=0).tolist(), consensus_xyz.max(axis=0).tolist()]
            if consensus_xyz is not None
            else None
        ),
        # Backward-compatible names describe the actually published support.
        "support_vertex_count": int(len(consensus_support)),
        "centroid_world_zup_m": (
            consensus_centre.tolist() if consensus_centre is not None else None
        ),
        "bounds_world_zup_m": (
            [consensus_xyz.min(axis=0).tolist(), consensus_xyz.max(axis=0).tolist()]
            if consensus_xyz is not None
            else None
        ),
        "component_gate_pass": passed,
        "gate_failures": failures,
    }


def _component_publication_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Priority used when two otherwise valid atoms claim the same support."""

    return (
        -len(row["visible_view_indices"]),
        -float(row["pair_compatibility_purity"]),
        -float(row["average_compatibility_score"]),
        -float(row["strong_pair_density"]),
        -float(row["consensus_fraction_of_union"]),
        float(row["centroid_diameter_m"]),
        tuple(sorted(row["member_observation_ids"])),
    )


def _support_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    overlap_count = int(np.intersect1d(left, right, assume_unique=True).size)
    minimum_count = min(len(left), len(right))
    union_count = len(left) + len(right) - overlap_count
    return {
        "overlap_vertex_count": overlap_count,
        "overlap_minimum_support_fraction": float(overlap_count / max(minimum_count, 1)),
        "jaccard": float(overlap_count / max(union_count, 1)),
    }


def strict_fuse_observations_with_audit(
    observations: list[Observation],
    vertices: np.ndarray,
    thresholds: AtomicFusionThresholds = DEFAULT_THRESHOLDS,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    """Agglomerate only component-compatible strong edges, never single-link."""

    for observation in observations:
        _require_observation_support(observation)
    active_indices, same_view_suppressions = _same_view_3d_deduplicate(observations)
    pair_audits: dict[tuple[int, int], dict[str, Any]] = {}
    strong_edges: list[tuple[int, int, dict[str, Any]]] = []
    pair_records: list[dict[str, Any]] = []
    for left_position, left_index in enumerate(active_indices):
        for right_index in active_indices[left_position + 1 :]:
            audit = atomic_pair_audit(
                observations[left_index], observations[right_index], vertices, thresholds
            )
            pair_audits[(left_index, right_index)] = audit
            record = {
                "left_observation_id": observations[left_index].observation_id,
                "right_observation_id": observations[right_index].observation_id,
                **audit,
            }
            pair_records.append(record)
            if bool(audit["strong_pair_edge_pass"]):
                strong_edges.append((left_index, right_index, audit))

    strong_edges.sort(
        key=lambda row: (
            -float(row[2]["compatibility_score"]),
            -int(row[2]["exact_vertex_count"]),
            float(row[2]["centroid_distance_m"]),
            observations[row[0]].observation_id,
            observations[row[1]].observation_id,
        )
    )
    active: dict[int, tuple[int, ...]] = {index: (index,) for index in active_indices}
    owner = {index: index for index in active_indices}
    next_component_id = len(observations)
    merge_decisions: list[dict[str, Any]] = []
    for left_index, right_index, edge_audit in strong_edges:
        left_component, right_component = owner[left_index], owner[right_index]
        if left_component == right_component:
            continue
        candidate = tuple(sorted(active[left_component] + active[right_component]))
        component_audit = _component_audit(
            candidate, observations, pair_audits, vertices, thresholds
        )
        accepted = bool(component_audit["component_gate_pass"])
        merge_decisions.append(
            {
                "trigger_edge": [
                    observations[left_index].observation_id,
                    observations[right_index].observation_id,
                ],
                "trigger_edge_compatibility_score": float(edge_audit["compatibility_score"]),
                "accepted": accepted,
                "candidate_component": component_audit,
            }
        )
        if not accepted:
            continue
        del active[left_component]
        del active[right_component]
        active[next_component_id] = candidate
        for member in candidate:
            owner[member] = next_component_id
        next_component_id += 1

    pre_exclusivity_rows: list[dict[str, Any]] = []
    unpublished: list[dict[str, Any]] = []
    for members in active.values():
        component_audit = _component_audit(members, observations, pair_audits, vertices, thresholds)
        if bool(component_audit["component_gate_pass"]):
            component_audit["pre_exclusivity_component_gate_pass"] = True
            pre_exclusivity_rows.append(component_audit)
        else:
            component_audit["pre_exclusivity_component_gate_pass"] = False
            component_audit["publication_gate_pass"] = False
            component_audit["publication_veto_reasons"] = ["component_gate_failed"]
            unpublished.append(component_audit)

    # A valid component can still be a duplicate atomic object.  Publish the
    # strongest deterministic claimant and fail closed on the weaker one when
    # more than 25 percent of the smaller consensus support is shared.
    ranked_candidates = sorted(pre_exclusivity_rows, key=_component_publication_rank)
    kept_ranked: list[tuple[dict[str, Any], np.ndarray]] = []
    exclusivity_vetoes: list[dict[str, Any]] = []
    ranking_contract = [
        "more_distinct_views",
        "higher_pair_compatibility_purity",
        "higher_average_compatibility_score",
        "higher_strong_pair_density",
        "higher_consensus_fraction_of_union",
        "smaller_centroid_diameter",
        "lexicographically_smaller_member_observation_ids",
    ]
    for priority_rank, component in enumerate(ranked_candidates):
        members = tuple(int(index) for index in component["member_indices"])
        _, consensus_support = _component_supports(members, observations)
        conflicts: list[dict[str, Any]] = []
        for kept_component, kept_support in kept_ranked:
            overlap = _support_overlap(consensus_support, kept_support)
            if (
                overlap["overlap_minimum_support_fraction"]
                > FINAL_ATOM_OVERLAP_MINIMUM_FRACTION_MAX
            ):
                conflicts.append(
                    {
                        "kept_member_observation_ids": kept_component[
                            "member_observation_ids"
                        ],
                        **overlap,
                    }
                )
        component["exclusivity_priority_rank"] = priority_rank
        if conflicts:
            component["publication_gate_pass"] = False
            component["publication_veto_reasons"] = [
                "final_atom_consensus_support_overlap_exceeds_0p25"
            ]
            component["publication_conflicts"] = copy.deepcopy(conflicts)
            unpublished.append(component)
            exclusivity_vetoes.append(
                {
                    "vetoed_member_observation_ids": component[
                        "member_observation_ids"
                    ],
                    "reason": "final_atom_consensus_support_overlap_exceeds_0p25",
                    "threshold": FINAL_ATOM_OVERLAP_MINIMUM_FRACTION_MAX,
                    "conflicts": copy.deepcopy(conflicts),
                    "exclusivity_priority_rank": priority_rank,
                }
            )
            continue
        component["publication_gate_pass"] = True
        component["publication_veto_reasons"] = []
        kept_ranked.append((component, consensus_support))

    # Retain P508's geometric union ordering for stable instance IDs, while the
    # NPZ payload written below is the consensus support in exactly this order.
    kept_geometric = sorted(
        kept_ranked,
        key=lambda item: tuple(
            np.round(item[0]["union_centroid_world_zup_m"], 6).tolist()
        ),
    )
    accepted_rows = [component for component, _ in kept_geometric]
    groups = [list(component["member_indices"]) for component in accepted_rows]
    published_supports = [support.copy() for _, support in kept_geometric]

    final_overlap_records: list[dict[str, Any]] = []
    maximum_final_overlap = 0.0
    for left_index in range(len(accepted_rows)):
        for right_index in range(left_index + 1, len(accepted_rows)):
            overlap = _support_overlap(
                published_supports[left_index], published_supports[right_index]
            )
            maximum_final_overlap = max(
                maximum_final_overlap,
                float(overlap["overlap_minimum_support_fraction"]),
            )
            final_overlap_records.append(
                {
                    "left_member_observation_ids": accepted_rows[left_index][
                        "member_observation_ids"
                    ],
                    "right_member_observation_ids": accepted_rows[right_index][
                        "member_observation_ids"
                    ],
                    **overlap,
                }
            )
    sam.require(
        maximum_final_overlap <= FINAL_ATOM_OVERLAP_MINIMUM_FRACTION_MAX,
        "final atomic support exclusivity gate drift",
    )

    final_owner: dict[int, int] = {
        member: group_index for group_index, members in enumerate(groups) for member in members
    }
    positive_edges: list[dict[str, Any]] = []
    for left_index, right_index, audit in strong_edges:
        positive_edges.append(
            {
                "left_observation_id": observations[left_index].observation_id,
                "right_observation_id": observations[right_index].observation_id,
                "accepted_into_same_published_component": bool(
                    left_index in final_owner
                    and right_index in final_owner
                    and final_owner[left_index] == final_owner[right_index]
                ),
                **audit,
            }
        )
    unpublished_observation_ids = {
        observation_id
        for row in unpublished
        for observation_id in row["member_observation_ids"]
    }
    unpublished_observation_ids.update(
        row["suppressed_observation_id"] for row in same_view_suppressions
    )
    audit = {
        "schema": "p515.atomic_cross_view_fusion_audit.v1",
        "algorithm": (
            "same-view 3-D NMS, strong-edge agglomeration, consensus-support "
            "publication, and deterministic final support exclusivity"
        ),
        "single_link_union_find_used": False,
        "thresholds": asdict(thresholds),
        "observation_count": len(observations),
        "active_observation_count_after_same_view_3d_deduplication": len(active_indices),
        "active_observation_ids_after_same_view_3d_deduplication": [
            observations[index].observation_id for index in active_indices
        ],
        "same_view_3d_containment_threshold": SAME_VIEW_3D_CONTAINMENT_MINIMUM,
        "same_view_3d_suppressed_observation_count": len(same_view_suppressions),
        "same_view_3d_suppressions": same_view_suppressions,
        "all_pair_audits": pair_records,
        "strong_pair_edge_count": len(strong_edges),
        "strong_pair_edges": copy.deepcopy(positive_edges),
        "merge_decisions": merge_decisions,
        "published_support_policy": "vertices_observed_by_at_least_two_distinct_views",
        "minimum_distinct_views_per_published_vertex": 2,
        "union_support_published": False,
        "pre_exclusivity_component_count": len(pre_exclusivity_rows),
        "final_exclusivity_overlap_minimum_support_fraction_max": (
            FINAL_ATOM_OVERLAP_MINIMUM_FRACTION_MAX
        ),
        "final_exclusivity_ranking_contract": ranking_contract,
        "exclusivity_vetoed_component_count": len(exclusivity_vetoes),
        "exclusivity_vetoed_components": exclusivity_vetoes,
        "published_component_pair_support_overlaps": final_overlap_records,
        "maximum_published_pair_overlap_minimum_support_fraction": maximum_final_overlap,
        "published_component_count": len(accepted_rows),
        "published_components": accepted_rows,
        "published_union_support_vertex_count_total": int(
            sum(row["union_support_vertex_count"] for row in accepted_rows)
        ),
        "published_consensus_support_vertex_count_total": int(
            sum(row["consensus_support_vertex_count"] for row in accepted_rows)
        ),
        "unpublished_component_count": len(unpublished),
        "unpublished_components": unpublished,
        "unpublished_observation_ids": sorted(unpublished_observation_ids),
    }
    return groups, positive_edges, audit


def _rewrite_support_archive_with_consensus(
    result: Mapping[str, Any],
    audit: Mapping[str, Any],
    published_supports: list[np.ndarray],
) -> dict[str, Any]:
    """Verify the frozen P508 union payload, then atomically publish consensus."""

    instances = list(result.get("instances", []))
    components = list(audit["published_components"])
    sam.require(
        len(instances) == len(components) == len(published_supports),
        "published support/component/instance count drift",
    )
    support_path = sam.checked_artifact(
        result.get("support_vertices"), "P508 union support vertices"
    )
    with np.load(support_path, allow_pickle=False) as archive:
        sam.require(
            set(archive.files)
            == {str(instance["support_vertices_file_key"]) for instance in instances},
            "P508 union support archive key drift",
        )
        for instance, component, consensus_support in zip(
            instances, components, published_supports
        ):
            key = str(instance["support_vertices_file_key"])
            union_support = np.asarray(archive[key], dtype=np.int64)
            sam.require(
                union_support.ndim == 1
                and np.all(union_support[:-1] < union_support[1:]),
                f"{key} P508 union support is not sorted unique",
            )
            sam.require(
                len(union_support) == int(component["union_support_vertex_count"]),
                f"{key} P508 union support count drift",
            )
            sam.require(
                _support_sha256(union_support)
                == component["union_support_vertex_ids_little_endian_int64_sha256"],
                f"{key} P508 union support identity drift",
            )
            sam.require(
                len(consensus_support)
                == int(component["consensus_support_vertex_count"]),
                f"{key} consensus support count drift",
            )
            sam.require(
                _support_sha256(consensus_support)
                == component["consensus_support_vertex_ids_little_endian_int64_sha256"],
                f"{key} consensus support identity drift",
            )

    consensus_payload = {
        str(instance["support_vertices_file_key"]): np.asarray(support, dtype=np.int64)
        for instance, support in zip(instances, published_supports)
    }
    temporary = support_path.with_name(
        support_path.name + f".consensus.tmp.{os.getpid()}"
    )
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **consensus_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, support_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sam.artifact(support_path)


def decorate_receipt(result: dict[str, Any], audit: dict[str, Any],
                     published_supports: list[np.ndarray]) -> dict[str, Any]:
    sam.require(
        len(result.get("instances", [])) == len(audit["published_components"]),
        "published component/instance count drift",
    )
    for instance, component in zip(result["instances"], audit["published_components"]):
        sam.require(
            np.allclose(
                np.asarray(instance["centroid_world_zup_m"], dtype=np.float64),
                np.asarray(component["union_centroid_world_zup_m"], dtype=np.float64),
                atol=1e-9,
                rtol=0.0,
            ),
            "atomic component/instance geometric ordering drift",
        )
    result["support_vertices"] = _rewrite_support_archive_with_consensus(
        result, audit, published_supports
    )
    for instance, component in zip(result["instances"], audit["published_components"]):
        instance["atomic_source_observation_ids"] = component["member_observation_ids"]
        instance["support_vertex_count"] = int(
            component["consensus_support_vertex_count"]
        )
        instance["centroid_world_zup_m"] = copy.deepcopy(
            component["consensus_centroid_world_zup_m"]
        )
        instance["bounds_world_zup_m"] = copy.deepcopy(
            component["consensus_bounds_world_zup_m"]
        )
        instance["published_support_policy"] = component[
            "support_publication_policy"
        ]
        instance["minimum_distinct_views_per_published_vertex"] = 2
        instance["atomic_fusion_audit"] = {
            key: copy.deepcopy(value)
            for key, value in component.items()
            if key not in {"member_indices"}
        }

    legacy_source = result.get("source")
    result["schema"] = OUTPUT_SCHEMA
    result["status"] = READY_STATUS
    result["atomic_instance_fusion"] = audit
    result["runtime"].update(
        {
            "strong_pair_edge_count": int(audit["strong_pair_edge_count"]),
            "published_atomic_component_count": int(audit["published_component_count"]),
            "unpublished_atomic_component_count": int(audit["unpublished_component_count"]),
            "same_view_3d_suppressed_observation_count": int(
                audit["same_view_3d_suppressed_observation_count"]
            ),
            "exclusivity_vetoed_component_count": int(
                audit["exclusivity_vetoed_component_count"]
            ),
        }
    )
    result["query_contract"].update(
        {
            "single_link_union_find_used": False,
            "p506_exact_support_strong_edge_required": True,
            "component_complete_link_compatibility_required": True,
            "component_average_link_score_and_density_required": True,
            "component_centroid_diameter_gate_required": True,
            "one_mask_per_view_component_purity_required": True,
            "same_view_3d_support_containment_deduplication": True,
            "same_view_3d_support_containment_threshold": (
                SAME_VIEW_3D_CONTAINMENT_MINIMUM
            ),
            "published_support_requires_two_distinct_views_per_vertex": True,
            "union_support_retained_for_audit_but_not_published": True,
            "final_atom_consensus_overlap_minimum_fraction_max": (
                FINAL_ATOM_OVERLAP_MINIMUM_FRACTION_MAX
            ),
            "rejected_or_single_view_masks_published_as_instances": False,
        }
    )
    result["implementation_lineage"] = {
        "observation_geometry_source": sam.artifact(Path(sam.__file__)),
        "inference_orchestrator_source": legacy_source,
        "replacement_boundary": (
            "same-view 3-D deduplication, cross-view atomic fusion, and "
            "consensus-support publication"
        ),
    }
    # Atomic segmentation is a successful intermediate stage, not permission
    # to publish a final target.  Make the next consumer explicit so a
    # transactional controller can advance instead of treating a directory's
    # existence as completion.
    result["stage_gate_pass"] = True
    result["publish_gate_pass"] = False
    result["next_action"] = {
        "type": "run_native_multiview_classification_on_atomic_instances",
        "reason": "strict atomic instances are ready but no semantic target has been verified",
        "required_source_schema": OUTPUT_SCHEMA,
    }
    result["source"] = sam.artifact(Path(__file__))
    result.pop("receipt_payload_sha256", None)
    result["receipt_payload_sha256"] = sam.canonical_hash(result)
    return result
