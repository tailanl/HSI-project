"""CPU-only surface descriptors; coordinates are measured, never model-generated.

Public API: ``describe_surface(V, F, face_ids, origin, outward, size=32)``
returns JSON metadata and NPZ-ready arrays. X follows an audited outward XY
direction, Y=Z cross X, Z is gravity-up. Array axes are [local X, local Y].
Raster cells are sampled at their centres on *actual triangles*, never their
convex hull. Missing cells contain zero plus a false validity mask, NOT known
free space. Holes smaller than the stated raster resolution may be unresolved.
"""
from pathlib import Path

import numpy as np

from memory_common import require

SCHEMA = "p555.surface_local_descriptor.v1"


def _finite(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), f"Invalid finite {label}")
    return result


def _indices(value, upper, label, *, unique=False):
    array = np.asarray(value)
    require(array.ndim == 1 and len(array) > 0 and array.dtype.kind in "iu", f"Invalid {label} indices")
    require(np.all(array >= 0) and np.all(array < upper), f"Out-of-range {label} indices")
    require(not unique or len(np.unique(array)) == len(array), f"Duplicate {label} indices")
    return array.astype(np.int64)


def validate_mesh(vertices, faces):
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 3
            and np.isfinite(vertices).all(), "Invalid finite mesh vertices")
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and faces.dtype.kind in "iu", "Invalid triangle mesh indices")
    require(np.all(faces >= 0) and np.all(faces < len(vertices)), "Out-of-range mesh indices")
    return vertices, faces.astype(np.int64)


def load_obj_world_zup(path):
    """Match P523 OBJ fan triangulation and native (x,y,z)->(x,-z,y)."""
    vertices, faces = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                require(len(fields) >= 4, f"Malformed OBJ vertex {number}")
                vertices.append(tuple(map(float, fields[1:4])))
            elif fields[0] == "f":
                require(len(fields) >= 4, f"Malformed OBJ face {number}")
                ids = []
                for token in fields[1:]:
                    index = int(token.split("/", 1)[0])
                    require(index != 0, "OBJ zero vertex index")
                    ids.append(index - 1 if index > 0 else len(vertices) + index)
                faces.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1))
    native, faces = validate_mesh(vertices, faces)
    return np.column_stack((native[:, 0], -native[:, 2], native[:, 1])), faces


def local_frame(origin, outward):
    origin = _finite(origin, (3,), "surface origin")
    direction = _finite(outward, (2,), "audited outward direction")
    length = float(np.linalg.norm(direction))
    require(length > 1e-10, "Zero outward direction")
    x = np.array([direction[0] / length, direction[1] / length, 0.])
    z = np.array([0., 0., 1.])
    rotation = np.column_stack((x, np.cross(z, x), z))
    return {"origin_world_zup_m": origin.tolist(), "local_to_world_rotation": rotation.tolist(),
            "axis_convention": "X=audited_outward,Y=Z_cross_X,Z=world_up", "units": "metres"}


def describe_surface(vertices, faces, face_ids, origin, outward, size=32):
    vertices, faces = validate_mesh(vertices, faces)
    ids = _indices(face_ids, len(faces), "surface face", unique=True)
    require(type(size) is int and 2 <= size <= 256, "Invalid raster size")
    frame = local_frame(origin, outward)
    rotation = np.asarray(frame["local_to_world_rotation"])
    triangles = (vertices[faces[ids]] - np.asarray(origin)) @ rotation
    points = triangles.reshape(-1, 3)
    lower, upper = points[:, :2].min(0), points[:, :2].max(0)
    extent = upper - lower
    require(np.all(extent > 1e-8), "Contact surface has no two-dimensional support")
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(crosses, axis=1) / 2
    require(float(areas.sum()) > 1e-10, "Degenerate contact triangles")
    spacing = extent / size
    x = lower[0] + (np.arange(size) + .5) * spacing[0]
    y = lower[1] + (np.arange(size) + .5) * spacing[1]
    height = np.full((size, size), -np.inf, dtype=np.float64)
    # Bounded per-triangle subgrids keep whole-scene extraction CPU-friendly.
    for triangle in triangles:
        a, b, c = triangle
        ab, ac = b[:2] - a[:2], c[:2] - a[:2]
        determinant = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(determinant) < 1e-14:
            continue
        lo = np.maximum(0, np.ceil((triangle[:, :2].min(0) - lower) / spacing - .5 - 1e-8).astype(int))
        hi = np.minimum(size - 1, np.floor((triangle[:, :2].max(0) - lower) / spacing - .5 + 1e-8).astype(int))
        if np.any(lo > hi):
            continue
        ix, iy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        dx, dy = x[ix] - a[0], y[iy] - a[1]
        u = (dx * ac[1] - dy * ac[0]) / determinant
        v = (ab[0] * dy - ab[1] * dx) / determinant
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
        old = height[ix, iy]
        height[ix, iy] = np.where(inside, np.maximum(old, z), old)
    valid = np.isfinite(height)
    require(bool(valid.any()), "No contact triangle intersects the raster cell centres")
    raster = np.where(valid, height, 0).astype(np.float32)
    values = raster[valid].astype(np.float64)
    features = {"surface_area_m2": float(areas.sum()), "projected_triangle_area_sum_m2": float(np.abs(crosses[:, 2]).sum() / 2),
                "extent_local_xy_m": extent.tolist(), "height_span_m": float(np.ptp(points[:, 2])),
                "raster_coverage_fraction": float(valid.mean()), "sampled_height_mean_m": float(values.mean()),
                "sampled_height_std_m": float(values.std()), "sampled_height_range_m": [float(values.min()), float(values.max())]}
    metadata = {"schema": SCHEMA, "frame": frame, "raster_size": size,
                "raster_axis_order": "local_x_local_y", "raster_bounds_local_xy_m": [lower.tolist(), upper.tolist()],
                "cell_size_xy_m": spacing.tolist(), "features": features,
                "raster_method": "actual_triangle_barycentric_cell_centres_topmost_z",
                "missing_cells_are_unknown_not_free": True, "unobserved_holes_filled": False,
                "metric_scale_preserved": True, "source_triangle_count": len(ids),
                "evidence_kind": "scene_geometry_only", "positive_credit": 0}
    arrays = {"contact_mask": valid.copy(), "valid_mask": valid.copy(), "heightmap_m": raster}
    validate_descriptor(metadata, arrays)
    return metadata, arrays


def validate_descriptor(metadata, arrays):
    """Validate decoded JSON/NPZ structure; the caller verifies artifact hashes."""
    require(metadata.get("schema") == SCHEMA, "Unknown surface descriptor schema")
    require(metadata.get("evidence_kind") == "scene_geometry_only" and type(metadata.get("positive_credit")) is int
            and metadata["positive_credit"] == 0, "Scene geometry cannot create positive credit")
    require(metadata.get("missing_cells_are_unknown_not_free") is True
            and metadata.get("unobserved_holes_filled") is False and metadata.get("metric_scale_preserved") is True,
            "Descriptor masks or metric contract drift")
    size = metadata["raster_size"]
    require(type(size) is int and 2 <= size <= 256, "Invalid descriptor raster size")
    require(set(arrays) == {"contact_mask", "valid_mask", "heightmap_m"}, "Descriptor array keys drift")
    for name, array in arrays.items():
        require(isinstance(array, np.ndarray) and array.shape == (size, size), f"Invalid {name} shape")
    contact, valid, height = (arrays[k] for k in ("contact_mask", "valid_mask", "heightmap_m"))
    require(contact.dtype == np.bool_ and valid.dtype == np.bool_ and np.array_equal(contact, valid)
            and valid.any(), "Invalid contact validity mask")
    require(height.dtype.kind == "f" and np.isfinite(height).all() and np.all(height[~valid] == 0), "Invalid masked heightmap")
    frame = metadata["frame"]
    _finite(frame["origin_world_zup_m"], (3,), "descriptor origin")
    rotation = _finite(frame["local_to_world_rotation"], (3, 3), "descriptor frame")
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            and abs(np.linalg.det(rotation) - 1) < 1e-8
            and np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-8, rtol=0), "Invalid right-handed Z-up frame")
    bounds = _finite(metadata["raster_bounds_local_xy_m"], (2, 2), "raster bounds")
    spacing = _finite(metadata["cell_size_xy_m"], (2,), "raster spacing")
    require(np.all(spacing > 0) and np.allclose(bounds[1] - bounds[0], spacing * size, atol=1e-8, rtol=0), "Metric raster bounds drift")
    features = metadata["features"]
    for key in ("surface_area_m2", "projected_triangle_area_sum_m2", "height_span_m", "sampled_height_mean_m", "sampled_height_std_m", "raster_coverage_fraction"):
        require(type(features[key]) in (int, float) and np.isfinite(features[key]), "Non-finite descriptor feature")
    require(features["surface_area_m2"] > 0 and features["projected_triangle_area_sum_m2"] > 0,
            "Non-positive surface area")
    require(features["projected_triangle_area_sum_m2"] <= features["surface_area_m2"] + 1e-8
            and features["height_span_m"] >= 0 and features["sampled_height_std_m"] >= 0,
            "Invalid metric geometry feature")
    require(np.allclose(_finite(features["extent_local_xy_m"], (2,), "metric extent"), spacing * size, atol=1e-8, rtol=0), "Metric extent drift")
    require(abs(features["raster_coverage_fraction"] - float(valid.mean())) < 1e-10, "Raster coverage drift")
    require(np.allclose(_finite(features["sampled_height_range_m"], (2,), "height range"),
                        [height[valid].min(), height[valid].max()], atol=1e-7, rtol=0), "Height range drift")
    require(abs(features["sampled_height_mean_m"] - float(height[valid].astype(np.float64).mean())) <= 1e-7
            and abs(features["sampled_height_std_m"] - float(height[valid].astype(np.float64).std())) <= 1e-7,
            "Height summary drift")
    require(type(metadata.get("source_triangle_count")) is int and metadata["source_triangle_count"] > 0,
            "Invalid source triangle count")
    return metadata
