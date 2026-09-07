"""Explicit fixed-scene geometry dependencies, replacing global monkey-patches."""
from pathlib import Path
import numpy as np

from hsi.common.artifacts import digest, verified
from . import _surface as kernel
from .region_scene import RegionBuilder


class SceneGeometryContext:
    def __init__(self, geometry, start, output):
        self.geometry = geometry
        self.rows = {r["instance_id"]: r for r in geometry["objects"]}
        with np.load(verified(geometry["navigation_fields"]), allow_pickle=False) as data:
            self.fields = {k: data[k] for k in data.files}
        self.builder = RegionBuilder(kernel, geometry, start, Path(output) / "contact_regions")
        self.active = None

    def load_and_validate_inputs(self, *args):
        return kernel.load_and_validate_inputs(*args)

    def build_navigation_fields(self, world, config):
        if digest(dict(vars(config))) != digest(self.geometry["navigation_config"]):
            raise ValueError("Navigation cache config drift")
        return self.builder.navigation(world, self.fields, config)

    def extract_support_surface(self, scene_id, key, category, support_ids, vertices, faces,
                                mesh_sha, graph_sha, config):
        row = self.rows[key]
        if not row["query_usable_for_sit"] or row["category"] != category:
            raise ValueError("Cached geometry cannot override semantic eligibility")
        if mesh_sha != self.geometry["source_mesh"]["sha256"] or scene_id != self.geometry["scene_id"]:
            raise ValueError("Cached geometry scene binding drift")
        with np.load(verified(row["surface_arrays"]), allow_pickle=False) as archive:
            face_ids, vertex_ids, cached_support, bounds = [archive[k] for k in
                ("face_ids", "vertex_ids", "support_ids", "target_bounds")]
        if not np.array_equal(cached_support, support_ids):
            raise ValueError("Cached SAM support drift")
        identity = {"schema": "p523.target_bound_surface_identity.v1", "scene_id": scene_id,
            "target_instance_id": key, "target_class": category, "mesh_sha256": mesh_sha,
            "graph_payload_sha256": graph_sha,
            "face_ids_little_endian_int64_sha256": kernel.int64_sha256(face_ids),
            "vertex_ids_little_endian_int64_sha256": kernel.int64_sha256(vertex_ids)}
        surface_sha = kernel.canonical_hash(identity)
        self.active = (row, vertices[faces[face_ids]], bounds)
        return face_ids, vertex_ids, bounds, row["surface"], surface_sha, \
            "INTERACTION_SURFACE_" + surface_sha[:16].upper(), {
            **row["extraction_audit"], "p550_cached_scene_geometry": True,
            "stable_scene_surface_sha256": row["stable_surface_sha256"],
            "query_only_identity_rebinding": True}

    def approach_options(self, surface, target_points, surface_points, navigation, config):
        if self.active is None:
            raise ValueError("Contact region requested before fixed surface binding")
        row, triangles, bounds = self.active
        return self.builder.build(row, triangles, target_points, bounds, navigation, config)
