"""CPU-only regression tests for P515 atomic SAM3.1 instance fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from hsi.stage1.perception import atomic


def _observation(
    observation_id: str,
    view_index: int,
    support: np.ndarray,
    vertices: np.ndarray,
) -> atomic.Observation:
    support = np.asarray(support, dtype=np.int64)
    xyz = vertices[support]
    return atomic.Observation(
        observation_id=observation_id,
        view_index=view_index,
        view_id=f"view_{view_index:02d}",
        proposal_id=f"ANON_{observation_id}",
        mask=np.ones((20, 20), dtype=bool),
        support=support,
        bounds=np.stack((xyz.min(axis=0), xyz.max(axis=0))),
        centroid=np.median(xyz, axis=0),
        quality=1.0,
        mask_audit={},
        prompt={"x": 0, "y": 0, "width": 20, "height": 20},
    )


def _compact_vertices(count: int) -> np.ndarray:
    return np.stack(
        (
            np.linspace(0.0, 0.10, count),
            0.01 * np.sin(np.linspace(0.0, 4.0, count)),
            0.01 * np.cos(np.linspace(0.0, 4.0, count)),
        ),
        axis=1,
    )


def test_p506_strong_edge_requires_both_80_vertices_and_16_percent() -> None:
    vertices = _compact_vertices(700)
    left = _observation("A", 0, np.arange(0, 100), vertices)
    right = _observation("B", 1, np.arange(20, 120), vertices)
    audit = atomic.atomic_pair_audit(left, right, vertices)
    assert audit["exact_vertex_count"] == 80
    assert audit["exact_minimum_support_fraction"] == 0.8
    assert audit["strong_pair_edge_pass"] is True

    # Still 80 exact vertices, but only 13.3 percent of the smaller support.
    vertices_large = _compact_vertices(1200)
    wide_left = _observation("C", 2, np.arange(0, 600), vertices_large)
    wide_right = _observation("D", 3, np.arange(520, 1120), vertices_large)
    audit = atomic.atomic_pair_audit(wide_left, wide_right, vertices_large)
    assert audit["exact_vertex_count"] == 80
    assert np.isclose(audit["exact_minimum_support_fraction"], 80 / 600)
    assert audit["strong_pair_edge_pass"] is False


def test_two_distinct_views_with_strong_exact_support_publish_one_instance() -> None:
    vertices = _compact_vertices(120)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(20, 120), vertices),
    ]
    groups, edges, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == [[0, 1]]
    assert len(edges) == 1
    assert edges[0]["accepted_into_same_published_component"] is True
    assert audit["published_component_count"] == 1
    assert audit["unpublished_component_count"] == 0
    component = audit["published_components"][0]
    assert component["union_support_vertex_count"] == 120
    assert component["consensus_support_vertex_count"] == 80
    assert component["union_only_vertex_count"] == 40
    assert component["support_vertex_count"] == 80
    assert component["support_publication_policy"] == (
        "vertices_observed_by_at_least_two_distinct_views"
    )


def test_ab_bc_strong_chain_cannot_merge_when_ac_is_incompatible() -> None:
    # B is an impure bridge mask: 80 vertices on object A and 80 on object C.
    # UnionFind would publish A+B+C.  Complete-link and diameter gates must not.
    left_xyz = _compact_vertices(100)
    right_xyz = _compact_vertices(100) + np.asarray([1.0, 0.0, 0.0])
    vertices = np.concatenate((left_xyz, right_xyz), axis=0)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(20, 180), vertices),
        _observation("C", 2, np.arange(100, 200), vertices),
    ]
    groups, edges, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)

    assert sum(edge["strong_pair_edge_pass"] for edge in edges) == 2
    assert all(len(group) == 2 for group in groups)
    assert not any(set(group) == {0, 1, 2} for group in groups)
    assert audit["published_component_count"] == 1
    assert len(audit["unpublished_observation_ids"]) == 1
    rejected = [row for row in audit["merge_decisions"] if not row["accepted"]]
    assert rejected
    failures = rejected[-1]["candidate_component"]["gate_failures"]
    assert "complete_link_pair_incompatibility" in failures
    assert "component_centroid_diameter_exceeded" in failures


def test_average_link_allows_two_strong_edges_plus_one_compatible_pair() -> None:
    vertices = _compact_vertices(140)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(20, 120), vertices),
        _observation("C", 2, np.arange(40, 140), vertices),
    ]
    groups, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == [[0, 1, 2]]
    component = audit["published_components"][0]
    assert component["compatible_pair_count"] == 3
    assert component["strong_pair_count"] == 2
    assert np.isclose(component["strong_pair_density"], 2 / 3)
    assert component["component_gate_pass"] is True


def test_same_view_duplicate_is_suppressed_before_cross_view_fusion() -> None:
    vertices = _compact_vertices(120)
    support = np.arange(0, 100)
    observations = [
        _observation("A", 0, support, vertices),
        _observation("B", 1, support, vertices),
        _observation("C", 0, support, vertices),
    ]
    groups, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert len({observations[index].view_index for index in groups[0]}) == 2
    assert len(audit["unpublished_observation_ids"]) == 1
    assert audit["same_view_3d_suppressed_observation_count"] == 1
    suppression = audit["same_view_3d_suppressions"][0]
    assert suppression["kept_observation_id"] == "A"
    assert suppression["suppressed_observation_id"] == "C"
    assert suppression["minimum_support_fraction"] == 1.0
    assert suppression["reason"] == "same_view_3d_support_containment_at_least_0p70"


def test_same_view_partial_3d_containment_at_0p70_is_deduplicated() -> None:
    vertices = _compact_vertices(120)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 0, np.arange(20, 120), vertices),
        _observation("C", 1, np.arange(0, 100), vertices),
    ]
    groups, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == [[0, 2]]
    assert audit["active_observation_ids_after_same_view_3d_deduplication"] == ["A", "C"]
    suppression = audit["same_view_3d_suppressions"][0]
    assert suppression["kept_observation_id"] == "A"
    assert suppression["suppressed_observation_id"] == "B"
    assert np.isclose(suppression["minimum_support_fraction"], 0.8)


def test_single_view_masks_are_audited_but_never_published() -> None:
    vertices = _compact_vertices(120)
    observations = [
        _observation("A", 4, np.arange(0, 100), vertices),
        _observation("B", 4, np.arange(20, 120), vertices),
    ]
    groups, edges, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == []
    assert edges == []
    assert audit["published_component_count"] == 0
    assert audit["unpublished_component_count"] == 1
    assert audit["same_view_3d_suppressed_observation_count"] == 1
    assert audit["unpublished_observation_ids"] == ["A", "B"]


def test_fusion_is_deterministic_for_identical_inputs() -> None:
    vertices = _compact_vertices(140)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(20, 120), vertices),
        _observation("C", 2, np.arange(40, 140), vertices),
    ]
    first = atomic.strict_fuse_observations_with_audit(observations, vertices)
    second = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_final_atom_overlap_above_0p25_deterministically_vetoes_weaker_atom() -> None:
    vertices = _compact_vertices(150)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(0, 100), vertices),
        _observation("C", 2, np.arange(50, 150), vertices),
        _observation("D", 3, np.arange(50, 150), vertices),
    ]
    groups, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == [[0, 1]]
    assert audit["pre_exclusivity_component_count"] == 2
    assert audit["published_component_count"] == 1
    assert audit["exclusivity_vetoed_component_count"] == 1
    veto = audit["exclusivity_vetoed_components"][0]
    assert veto["vetoed_member_observation_ids"] == ["C", "D"]
    assert veto["reason"] == "final_atom_consensus_support_overlap_exceeds_0p25"
    assert veto["conflicts"][0]["kept_member_observation_ids"] == ["A", "B"]
    assert np.isclose(veto["conflicts"][0]["overlap_minimum_support_fraction"], 0.5)
    assert audit["maximum_published_pair_overlap_minimum_support_fraction"] <= 0.25
    vetoed_component = next(
        row
        for row in audit["unpublished_components"]
        if row["member_observation_ids"] == ["C", "D"]
    )
    assert vetoed_component["component_gate_pass"] is True
    assert vetoed_component["publication_gate_pass"] is False
    assert vetoed_component["publication_veto_reasons"] == [
        "final_atom_consensus_support_overlap_exceeds_0p25"
    ]


def test_two_published_atoms_can_share_at_most_0p25_of_smaller_consensus() -> None:
    vertices = _compact_vertices(180)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(0, 100), vertices),
        _observation("C", 2, np.arange(80, 180), vertices),
        _observation("D", 3, np.arange(80, 180), vertices),
    ]
    groups, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == [[0, 1], [2, 3]]
    assert audit["published_component_count"] == 2
    assert audit["exclusivity_vetoed_component_count"] == 0
    assert len(audit["published_component_pair_support_overlaps"]) == 1
    overlap = audit["published_component_pair_support_overlaps"][0]
    assert np.isclose(overlap["overlap_minimum_support_fraction"], 0.2)
    assert np.isclose(
        audit["maximum_published_pair_overlap_minimum_support_fraction"], 0.2
    )


def test_chair_like_component_never_publishes_empty_consensus_support() -> None:
    vertices = _compact_vertices(180)
    observations = [
        _observation("CHAIR_FRONT", 3, np.arange(0, 150), vertices),
        _observation("CHAIR_SIDE", 8, np.arange(50, 180), vertices),
    ]
    groups, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    assert groups == [[0, 1]]
    component = audit["published_components"][0]
    assert component["consensus_support_vertex_count"] == 100
    assert component["consensus_support_vertex_count"] >= 80
    support = atomic._component_supports(tuple(groups[0]), observations)[1]
    assert np.array_equal(support, np.arange(50, 150))


def test_receipt_decoration_rewrites_union_npz_to_consensus_and_uses_strict_status(
    tmp_path: Path,
) -> None:
    vertices = _compact_vertices(120)
    observations = [
        _observation("A", 0, np.arange(0, 100), vertices),
        _observation("B", 1, np.arange(20, 120), vertices),
    ]
    _, _, audit = atomic.strict_fuse_observations_with_audit(observations, vertices)
    component = audit["published_components"][0]
    union_support = np.arange(0, 120, dtype=np.int64)
    support_path = tmp_path / "instance_support_vertices.npz"
    np.savez_compressed(support_path, SCENE_INSTANCE_000=union_support)
    instance = {
        "instance_id": "SCENE_INSTANCE_000",
        "cross_view_gate_pass": True,
        "visible_view_ids": [0, 1],
        "observation_count": 2,
        "support_vertex_count": component["union_support_vertex_count"],
        "support_vertices_file_key": "SCENE_INSTANCE_000",
        "centroid_world_zup_m": component["union_centroid_world_zup_m"],
        "bounds_world_zup_m": component["union_bounds_world_zup_m"],
        "classifier_images": [{"path": "one"}, {"path": "two"}, {"path": "three"}],
        "source_masks": [],
        "semantic_class": None,
    }
    receipt = {
        "schema": "p508.lingo_sam31_multiview_instances.v1",
        "status": "scene_instances_ready",
        "instances": [instance],
        "runtime": {},
        "query_contract": {},
        "source": {"path": "frozen-p508-source"},
        "support_vertices": atomic.sam.artifact(support_path),
    }
    decorated = atomic.decorate_receipt(receipt, audit, [atomic._component_supports((0, 1), observations)[1]])
    row = decorated["instances"][0]
    assert decorated["schema"] == atomic.OUTPUT_SCHEMA
    assert decorated["status"] == atomic.READY_STATUS
    assert row["support_vertices_file_key"] == "SCENE_INSTANCE_000"
    assert row["support_vertex_count"] == 80
    assert row["centroid_world_zup_m"] == component["consensus_centroid_world_zup_m"]
    assert row["bounds_world_zup_m"] == component["consensus_bounds_world_zup_m"]
    assert row["classifier_images"] == [
        {"path": "one"}, {"path": "two"}, {"path": "three"}
    ]
    assert row["atomic_source_observation_ids"] == ["A", "B"]
    assert row["atomic_fusion_audit"]["union_support_vertex_count"] == 120
    assert row["atomic_fusion_audit"]["consensus_support_vertex_count"] == 80
    with np.load(support_path, allow_pickle=False) as archive:
        assert np.array_equal(archive["SCENE_INSTANCE_000"], np.arange(20, 100))
    assert decorated["support_vertices"] == atomic.sam.artifact(support_path)
    assert decorated["query_contract"]["single_link_union_find_used"] is False
    assert decorated["query_contract"][
        "published_support_requires_two_distinct_views_per_vertex"
    ] is True
    assert decorated["stage_gate_pass"] is True
    assert decorated["publish_gate_pass"] is False
    assert decorated["next_action"]["type"] == (
        "run_native_multiview_classification_on_atomic_instances"
    )


