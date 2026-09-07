"""Query-time adapter; reads immutable Stage1A geometry, never reclassifies it."""

import os

os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"

import math

from pathlib import Path

import sys

import time

from hsi.common.artifacts import artifact, digest, verified, read_sealed, write_once

import numpy as np

from .region_geometry import construct, RegionPolicy

class RegionBuilder:
    def __init__(self, kernel, geometry, start, root, policy=RegionPolicy()):
        self.kernel, self.geometry = kernel, geometry
        self.start, self.root, self.policy = start, Path(root), policy
        self.records = {}

    def navigation(self, world, fields, config):
        """Compute height data and verify agreement with the fixed collision map."""
        lower = np.asarray(config.grid_lower_world_xyz_m)
        heights = lower[2] + (np.arange(world.shape[2]) + .5) * config.voxel_size_m
        body = (heights >= config.body_height_range_m[0]) & (heights <= config.body_height_range_m[1])
        body_voxels = world[:, :, body]
        obstacles = body_voxels.any(axis=2)
        if not np.array_equal(obstacles, fields["obstacles"]):
            raise ValueError("Height-slab projection disagrees with the fixed Stage1A obstacle map")
        if np.any(fields["free"] & (obstacles | (fields["clearance"] < config.navigation_clearance_m - 1e-9))):
            raise ValueError("Cached free space violates the unchanged human clearance")
        self.world, self.body = world, body
        self.height = np.max(np.where(body_voxels, heights[body].astype(np.float32), 0), axis=2)
        snapped = self.kernel.nearest_true(fields["free"], np.asarray(self.start), config.maximum_start_snap_m, config)
        if snapped is None:
            raise ValueError("No human-free cell near the supplied start")
        self.start_cell, self.start_snap_distance = snapped
        self.distances = self.kernel.flood_distances(fields["free"], self.start_cell)
        self.reachable = self.distances >= 0
        return fields

    def build(self, row, triangles, target_points, target_bounds, navigation, config):
        started = time.monotonic()
        target_mask, target_audit = self.kernel.target_mask_from_support(self.world, target_points, target_bounds, config)
        target_footprint = target_mask[:, :, self.body].any(axis=2)
        candidates, masks, audit = construct(triangles, row["direction_hypotheses"]["candidates"], navigation,
            target_footprint, self.reachable, config.grid_lower_world_xyz_m[:2], config.voxel_size_m,
            row["surface"]["centre_world_xyz_zup_m"], self.policy)
        root = self.root / row["instance_id"]
        root.mkdir(parents=True, exist_ok=False)
        arrays = root / "region_arrays.npz"
        with arrays.open("xb") as stream:
            np.savez_compressed(stream, **masks, body_height_map_m=self.height, clearance=navigation["clearance"],
                contact_triangles=triangles, grid_lower_world_xy_m=config.grid_lower_world_xyz_m[:2],
                voxel_size_m=config.voxel_size_m)
        options = []
        centre = np.asarray(row["surface"]["centre_world_xyz_zup_m"])
        for candidate in candidates:
            delta = np.asarray(candidate["approach_world_xy_m"]) - centre[:2]
            angle = math.atan2(delta[1], delta[0])
            front = candidate["outward_world_xy"]
            yaw = math.atan2(front[1], front[0])
            difference = abs(math.atan2(math.sin(angle-yaw), math.cos(angle-yaw)))
            option_id = f"APPROACH_OPTION_{len(options):02d}"
            options.append({"option_id": option_id, "source_option_index": len(options),
                "approach_world_xy_m": candidate["approach_world_xy_m"], "contact_forward_yaw_rad": yaw,
                "approach_direction_from_support_rad": angle, "front_orientation_difference_rad": difference,
                "p550_axis_reference_difference_rad": difference, "radius_from_support_m": float(np.linalg.norm(delta)),
                "approach_to_support_edge_min_clearance_m": candidate["front_gap_m"],
                "raw_occupancy_clearance_m": candidate["clearance_m"], "requires_fresh_route_planning": True,
                "source_route_copied_to_final_route": False, "route_planning_status": "fresh_route_required",
                "p550_direction_hypothesis_id": candidate["direction_id"],
                "p550_body_facing_decoupled_from_approach_position": True,
                "p550_candidate_source": "p552_actual_contact_mesh_front_region",
                "p552_region_member": {**candidate, "region_arrays": artifact(arrays)}})
        record = {"schema": "p552.directional_contact_endpoint_region.v1", "scene_id": self.geometry["scene_id"],
            "target_instance_id": row["instance_id"], "target_category_unchanged": row["category"],
            "stable_surface_sha256": row["stable_surface_sha256"], "fixed_surface_arrays": row["surface_arrays"],
            "source_occupancy": self.geometry["source_occupancy"], "source_mesh": self.geometry["source_mesh"],
            "fixed_navigation": self.geometry["navigation_fields"], "fixed_direction_hypotheses": row["direction_hypotheses"],
            "surface": row["surface"], "start_world_xy_m": list(self.start), "start_cell": list(self.start_cell),
            "snapped_start_world_xy_m": self.kernel.cell_center(self.start_cell, config).tolist(),
            "navigation_clearance_m": config.navigation_clearance_m, "body_height_range_m": list(config.body_height_range_m),
            "height_map_matches_fixed_collision_projection": True, "target_mask_audit": target_audit,
            "region_arrays": artifact(arrays), "approach_options": options, "audit": audit,
            "source_codes": [artifact(__file__), artifact(Path(__file__).with_name("region_geometry.py"))],
            "elapsed_seconds": time.monotonic() - started}
        path = root / "receipt.json"
        write_once(path, record, seal=True)
        self.records[row["instance_id"]] = artifact(path)
        front = row["direction_hypotheses"]["candidates"][0]["outward_world_xy"]
        return options, {**row["front_evidence"], "front_direction_world_xy": front,
            "front_yaw_rad": math.atan2(front[1], front[0]),
            "p550_direction_hypotheses": row["direction_hypotheses"],
            "p550_body_facing_decoupled_from_approach": True, "p552_contact_region": artifact(path),
            "p552_empty_region_is_not_category_error": True, "p552_no_radius_ring_or_off_region_snap": True}

