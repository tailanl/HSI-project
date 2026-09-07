import os
os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from hsi.stage1.region_geometry import construct, contact_footprint, grid_points, clear_connectors, RegionPolicy

LOWER = [-2., -2.]
RES = .02
SHAPE = (200, 200)


def rectangle(x0=-.35, x1=.35, y0=-.7, y1=.7):
    v = np.array([[x0,y0,.45],[x1,y0,.45],[x1,y1,.45],[x0,y1,.45]])
    return v[[[0,1,2],[0,2,3]]]


def fixture(triangles, extra=None):
    target = contact_footprint(triangles, SHAPE, LOWER, RES)
    obstacles = target.copy()
    if extra is not None:
        obstacles |= extra
    clearance = distance_transform_edt(~obstacles) * RES
    free = ~obstacles & (clearance >= .28 - 1e-9)
    return {"free":free, "obstacles":obstacles, "clearance":clearance}, target


def run_region(triangles, extra=None, reachable=None, directions=None):
    nav, target = fixture(triangles, extra)
    return construct(triangles, directions or [{"candidate_id":"DIRECTION_0","outward_world_xy":[1.,0.]}], nav,
        target, nav["free"] if reachable is None else reachable, LOWER, RES, triangles.mean(axis=(0,1)))


def test_rectangle_endpoint_is_on_actual_front_band():
    rows, masks, audit = run_region(rectangle())
    assert rows
    for row in rows:
        x,y = row["approach_world_xy_m"]
        assert .63 - 1e-9 <= x <= .69 + 1e-9
        assert abs(y) <= .68
        assert abs(row["contact_anchor_world_xyz_m"][0] - .35) < 1e-8
        assert masks["terminal_frontier"][tuple(row["cell"])]


def test_rotated_long_surface_stays_in_front_not_radial_sides():
    angle = .63
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    triangles = rectangle(y0=-1.2,y1=1.2)
    triangles[...,:2] = triangles[...,:2] @ rotation.T
    direction = rotation[:,0]
    rows, _, _ = run_region(triangles, directions=[{"candidate_id":"DIRECTION_0","outward_world_xy":direction}])
    assert rows
    for row in rows:
        p = np.asarray(row["approach_world_xy_m"])
        a = np.asarray(row["contact_anchor_world_xyz_m"])
        assert .28 - 1e-8 <= (p-a[:2]) @ direction <= .55 + 1e-8
        assert np.linalg.norm(p-a[:2]) < .551


def test_nearby_object_clips_the_region():
    xy = grid_points(SHAPE, LOWER, RES)
    extra = (xy[...,0] > .65) & (xy[...,0] < 1.1) & (xy[...,1] > -.2) & (xy[...,1] < .2)
    rows,masks,audit = run_region(rectangle(), extra)
    assert rows
    assert audit["clipping_counts"][0]["human_free_cells"] < audit["clipping_counts"][0]["raw_cells"]
    assert all(abs(r["approach_world_xy_m"][1]) > .45 for r in rows)


def test_intervening_obstacle_rejects_even_when_endpoint_clear():
    obstacles = np.zeros(SHAPE, bool)
    obstacles[120,100] = True
    result = clear_connectors(np.array([[.85,.01],[.85,.51]]), np.array([[.35,.01],[.35,.51]]), obstacles, LOWER, RES)
    assert result.tolist() == [False, True]


def test_disconnected_region_does_not_snap_to_far_reachable_space():
    triangles = rectangle()
    nav,_ = fixture(triangles)
    xy = grid_points(SHAPE,LOWER,RES)
    reachable = nav["free"] & (xy[...,0] < -.8)
    rows,masks,audit = run_region(triangles, reachable=reachable)
    assert not rows and not masks["reachable_region"].any()
    assert audit["status"] == "no_reachable_contact_front_region"


def test_missing_contact_lanes_are_not_filled_by_a_bounding_box():
    triangles = np.concatenate([rectangle(y0=-.9,y1=-.25),rectangle(y0=.25,y1=.9)])
    rows,masks,_ = run_region(triangles)
    xy = grid_points(SHAPE,LOWER,RES)
    assert rows and not np.any(masks["raw_region"] & (abs(xy[...,1]) < .2))


def test_both_fixed_front_hypotheses_are_retained():
    rows,_,audit = run_region(rectangle(), directions=[
        {"candidate_id":"DIRECTION_0","outward_world_xy":[1,0]},
        {"candidate_id":"DIRECTION_1","outward_world_xy":[-1,0]}])
    assert {r["direction_id"] for r in rows} == {"DIRECTION_0","DIRECTION_1"}
    assert len(audit["directions"]) == 2

