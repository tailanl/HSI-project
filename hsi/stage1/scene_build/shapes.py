from ._common import *
from .geometry import geometry_kernel

def run(geometry_path, output):
    started = time.monotonic()
    geometry = read_sealed(geometry_path)
    semantics = read_sealed(verified(geometry["source_semantics"]))
    lookup = {r["instance_id"]: r for r in semantics["objects"]}
    kernel = geometry_kernel()
    vertices, faces = kernel.load_obj_world_zup(verified(geometry["source_mesh"]))
    triangles = vertices[faces]
    crosses = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    lengths = np.linalg.norm(crosses, axis=1)
    normals = crosses/np.maximum(lengths[:,None], 1e-12)
    areas = lengths/2
    centres = triangles.mean(1)
    output.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(geometry)
    directions = [("upward", [0.,0.,1.]), ("downward", [0.,0.,-1.]),
        ("vertical_positive_x", [1.,0.,0.]), ("vertical_negative_x", [-1.,0.,0.]),
        ("vertical_positive_y", [0.,1.,0.]), ("vertical_negative_y", [0.,-1.,0.])]
    all_patches = 0
    with np.load(verified(geometry["source_support"]), allow_pickle=False) as supports:
        for row in result["objects"]:
            key = row["instance_id"]
            obj = lookup[key]
            ids = np.unique(supports[obj["geometry_source"]["support_vertices_file_key"]])
            membership = np.zeros(len(vertices), dtype=bool)
            membership[ids] = True
            owned = (membership[faces].sum(1) >= 2) & (areas > 1e-10)
            bounds = np.asarray(obj["geometry_source"]["bounds_world_zup_m"])
            owned &= np.all((triangles >= bounds[0]-1e-9) & (triangles <= bounds[1]+1e-9), axis=(1,2))
            candidates = []
            for name, axis in directions:
                mask = owned & ((normals @ np.asarray(axis)) >= .88)
                positions = np.flatnonzero(mask)
                if len(positions) < 2: continue
                # Separate parallel shelves and other spatially distinct planes.
                plane_bin = np.floor((centres[positions] @ np.asarray(axis))/.06).astype(int)
                for bin_id in np.unique(plane_bin):
                    selection = positions[plane_bin == bin_id]
                    if float(areas[selection].sum()) < .01: continue
                    for component in kernel.connected_face_components(faces, selection):
                        area = float(areas[component].sum())
                        if len(component) < 2 or area < .01: continue
                        vertices_in_patch = vertices[np.unique(faces[component])]
                        extents = np.ptp(vertices_in_patch, axis=0)
                        if sorted(extents)[-2] < .08: continue
                        normal = np.average(normals[component], axis=0, weights=areas[component])
                        normal /= np.linalg.norm(normal)
                        candidates.append((area, name, component, normal, vertices_in_patch))
            candidates.sort(key=lambda r: (-r[0], r[1], int(r[2].min())))
            patches, arrays = [], {}
            for area, name, component, normal, points in candidates[:8]:
                identity = digest({"scene": geometry["scene_id"], "instance": key, "mesh": geometry["source_mesh"]["sha256"],
                    "face_ids_sha256": kernel.int64_sha256(component), "source_support": geometry["source_support"]["sha256"]})
                patch_id = "FACE_"+identity[:14].upper()
                arrays[patch_id] = np.sort(component)
                patches.append({"surface_candidate_id": patch_id, "orientation_family": name, "area_m2": area,
                    "centre_world_zup_m": np.average(centres[component], axis=0, weights=areas[component]).tolist(),
                    "bounds_world_zup_m": np.stack((points.min(0),points.max(0))).tolist(),
                    "outward_normal_world_zup": normal.tolist(), "contact_approach_direction_world_zup": (-normal).tolist(),
                    "semantic_interaction_role": "unassigned_geometric_candidate", "category_override_allowed": False,
                    "object_identity_verified": obj["decision"]["query_usable"], "face_count": len(component)})
            row["generic_face_candidates"] = patches
            row["generic_geometry_not_an_action_feasibility_claim"] = True
            if arrays:
                folder = output / "objects" / key
                folder.mkdir(parents=True)
                path = folder / "generic_faces.npz"
                np.savez_compressed(path, **arrays)
                row["generic_faces_archive"] = artifact(path)
            all_patches += len(patches)
    result.update(schema="p550.fixed_scene_interaction_geometry.v2", source_sitting_geometry=artifact(geometry_path),
        generic_shape_source_code=artifact(__file__), generic_shape_elapsed_seconds=time.monotonic()-started,
        generic_face_candidate_count=all_patches, generic_faces_are_geometric_proposals_not_affordance_memory=True,
        generic_face_sampling_policy={"minimum_area_m2": .01, "minimum_second_extent_m": .08,
            "normal_alignment_min": .88, "parallel_plane_bin_m": .06, "maximum_patches_per_instance": 8})
    write_once(output / "receipt.json", result, seal=True)
    print({"scene": result["scene_id"], "generic_patches": all_patches, "seconds": result["generic_shape_elapsed_seconds"]}, flush=True)
