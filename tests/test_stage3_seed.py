"""CPU neutral seed contracts; no body checkpoint, motion row, model or GPU run."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from hsi.common.artifacts import MemoryContractError, artifact, write_once
from hsi.stage3 import seed, static_seed, frame


def arrays():
    value = {key: np.zeros(shape, dtype=np.int64 if key == "faces" else np.float32)
             for key, shape in seed.ARRAY_SHAPES.items()}
    value["global_orient_world_rotmat"] = np.eye(3, dtype=np.float32)
    value["joints_world_zup"][1] = [0., .1, 0.]
    value["joints_world_zup"][2] = [0., -.1, 0.]
    return value


def test_registered_neutral_identity_and_nine_array_shapes_are_unchanged():
    assert seed.NEUTRAL_SHA256 == "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
    assert seed.BUFFER_KEYS == {"v_template", "J_regressor", "lbs_weights", "shapedirs", "posedirs"}
    assert seed.ARRAY_SHAPES == {"betas": (10,), "body_pose_axis_angle": (63,),
        "global_orient_world_rotmat": (3, 3), "smplx_transl_world_zup": (3,),
        "pelvis_delta_native": (3,), "joints_world_zup": (22, 3),
        "root_xyz_yaw_world_zup": (4,), "vertices_world_zup": (10475, 3), "faces": (20908, 3)}
    value = arrays()
    assert seed.validate_arrays(value) is value


@pytest.mark.parametrize("key", ["betas", "body_pose_axis_angle"])
def test_nonzero_shape_or_pose_is_not_current_static_neutral(key):
    value = arrays()
    value[key][0] = .1
    with pytest.raises(MemoryContractError, match="zero-shape and zero-pose"):
        seed.validate_arrays(value)


@pytest.mark.parametrize("key", list(seed.ARRAY_SHAPES))
def test_every_seed_array_is_required_and_shape_bound(key):
    value = arrays()
    value.pop(key)
    with pytest.raises(MemoryContractError, match="Unexpected/missing"):
        seed.validate_arrays(value)
    value = arrays()
    value[key] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(MemoryContractError, match="Invalid neutral seed array"):
        seed.validate_arrays(value)


@pytest.mark.parametrize("case", ["extra", "reflection", "root", "pelvis", "yaw", "hips", "face", "nan", "object"])
def test_invalid_geometry_and_extra_arrays_fail_closed(case):
    value = arrays()
    if case == "extra": value["old_pose"] = np.zeros(63)
    if case == "reflection": value["global_orient_world_rotmat"][0, 0] = -1
    if case == "root": value["root_xyz_yaw_world_zup"][0] = .1
    if case == "pelvis": value["pelvis_delta_native"][0] = .1
    if case == "yaw": value["root_xyz_yaw_world_zup"][3] = 1.
    if case == "hips": value["joints_world_zup"][2] = value["joints_world_zup"][1]
    if case == "face": value["faces"][0, 0] = 10475
    if case == "nan": value["vertices_world_zup"][0, 0] = np.nan
    if case == "object": value["vertices_world_zup"] = value["vertices_world_zup"].astype(object)
    with pytest.raises(MemoryContractError):
        seed.validate_arrays(value)


def test_numeric_fingerprint_binds_dtype_shape_content_not_memory_layout():
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert seed.array_hash(value) == seed.array_hash(np.asfortranarray(value))
    assert seed.array_hash(value) != seed.array_hash(value.reshape(4, 3))
    assert seed.array_hash(value) != seed.array_hash(value.astype(np.float64))
    changed = value.copy()
    changed[0, 0] = 100.
    assert seed.array_hash(value) != seed.array_hash(changed)
    for invalid in (np.array([np.inf]), np.array(["model"]), np.array([object()])):
        with pytest.raises(MemoryContractError):
            seed.array_hash(invalid)


@pytest.mark.parametrize("actual,expected", [(True, 1), (1, True), (2., 2), ("male", "neutral"), ([2], 2)])
def test_scalar_types_and_shapes_are_exact(actual, expected):
    with pytest.raises(MemoryContractError):
        seed.validate_scalar({"x": np.array(actual)}, "x", expected)


@pytest.mark.parametrize("value", [True, 2, "neutral"])
def test_exact_scalar_is_valid(value):
    seed.validate_scalar({"x": np.array(value)}, "x", value)


def model_fixture(path):
    """Tiny numerical buffer layout, explicitly not a registered body asset."""
    source = {"v_template": np.arange(6, dtype=np.float64).reshape(2, 3),
              "J_regressor": np.arange(4, dtype=np.float64).reshape(2, 2),
              "weights": np.arange(4, dtype=np.float64).reshape(2, 2),
              "shapedirs": np.arange(72, dtype=np.float64).reshape(2, 3, 12),
              "posedirs": np.arange(24, dtype=np.float64).reshape(2, 3, 4)}
    np.savez(path, **source)
    return source


def test_five_model_buffer_conversions_match_original_runtime_layout(tmp_path):
    path = tmp_path / "synthetic_layout.npz"
    value = model_fixture(path)
    actual = seed.model_array_hashes(path)
    expected = {"v_template": value["v_template"], "J_regressor": value["J_regressor"],
                "lbs_weights": value["weights"], "shapedirs": value["shapedirs"][..., :10],
                "posedirs": value["posedirs"].reshape(-1, 4).T}
    assert actual == {key: seed.array_hash(np.ascontiguousarray(row, dtype=np.float32))
                      for key, row in expected.items()}
    assert set(actual) == seed.BUFFER_KEYS
    assert seed.array_hash(value["shapedirs"].astype(np.float32)) != actual["shapedirs"]


@pytest.fixture
def buffers(monkeypatch):
    expected = {name: str(index) * 64 for index, name in enumerate(sorted(seed.BUFFER_KEYS))}
    # Only this fixture's asset I/O is stubbed; exact set/recomputation checks remain real.
    monkeypatch.setattr(seed, "verified", lambda value: Path(value["path"]))
    monkeypatch.setattr(seed, "model_array_hashes", lambda path: expected)
    receipt = {"runtime_model_array_hashes": copy.deepcopy(expected), "body_assets": {
        "candidate_body": {"path": "/fixture/neutral.npz", "bytes": 1, "sha256": seed.NEUTRAL_SHA256}}}
    return receipt, expected


def test_exact_five_buffers_must_reproduce_the_registered_asset(buffers):
    receipt, expected = buffers
    assert seed.validate_buffer_identity(receipt) == expected


@pytest.mark.parametrize("bad", [{}, None, [], {"v_template": "a" * 64}])
def test_partial_or_empty_buffer_dictionary_cannot_disable_runtime_checks(buffers, bad):
    receipt, _ = buffers
    receipt["runtime_model_array_hashes"] = bad
    with pytest.raises(MemoryContractError, match="exactly five keys"):
        seed.validate_buffer_identity(receipt)


@pytest.mark.parametrize("case", ["extra", "recomputed_mismatch", "other_asset"])
def test_five_buffer_identity_rejects_extra_changed_or_non_neutral_assets(buffers, case):
    receipt, _ = buffers
    if case == "extra": receipt["runtime_model_array_hashes"]["unknown"] = "f" * 64
    if case == "recomputed_mismatch": receipt["runtime_model_array_hashes"]["v_template"] = "f" * 64
    if case == "other_asset": receipt["body_assets"]["candidate_body"]["sha256"] = "f" * 64
    with pytest.raises(MemoryContractError):
        seed.validate_buffer_identity(receipt)


def test_real_file_identity_rejects_unregistered_numeric_body(tmp_path):
    path = tmp_path / "not_registered.npz"
    model_fixture(path)
    receipt = {"runtime_model_array_hashes": seed.model_array_hashes(path),
               "body_assets": {"candidate_body": artifact(path)}}
    with pytest.raises(MemoryContractError, match="registered neutral asset"):
        seed.validate_buffer_identity(receipt)
    with pytest.raises(MemoryContractError, match="identity differs"):
        seed.verify_neutral_assets(candidate_body=path, runtime_body=path, render_body=path)


def test_empty_runtime_fingerprints_fail_before_dataset_or_feature_math():
    fake = SimpleNamespace(receipt={"runtime_model_array_hashes": {}})
    with pytest.raises(MemoryContractError, match="exactly five keys"):
        seed.NeutralSeed.build_p360_sample(fake, None, "cpu", None)


def test_wrong_query_rejected_before_external_dataset():
    fake = SimpleNamespace(receipt={"scene_id": "s", "instruction": "Sit."})
    with pytest.raises(MemoryContractError, match="different queries"):
        seed.NeutralSeed.build_runtime_dataset(fake, None, "cpu", SimpleNamespace(scene_name="other", text="Sit."), None)


def runtime_seed_fixture(monkeypatch):
    """Numerical runtime tensor doubles; real five-key/hash logic is exercised."""
    class Buffer:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value
    body = SimpleNamespace(**{name: Buffer(np.zeros((2, 3), np.float32)) for name in seed.BUFFER_KEYS})
    expected = {name: seed.array_hash(getattr(body, name).value) for name in seed.BUFFER_KEYS}
    monkeypatch.setattr(seed, "verified", lambda row: Path(row["path"]))
    monkeypatch.setattr(seed, "model_array_hashes", lambda path: expected)
    value = arrays()
    current = seed.NeutralSeed(Path("/fixture/seed.npz"), "0" * 64, "neutral", value["betas"],
        value["body_pose_axis_angle"], value["global_orient_world_rotmat"], value["smplx_transl_world_zup"],
        value["pelvis_delta_native"], value["joints_world_zup"], value["root_xyz_yaw_world_zup"],
        {"p555_seed_receipt": {"test_fixture_only": True}},
        {"runtime_model_array_hashes": expected, "body_assets": {"candidate_body": {
            "path": "/fixture/body.npz", "bytes": 0, "sha256": seed.NEUTRAL_SHA256}},
         "scene_id": "synthetic", "instruction": "Sit."})
    dataset = SimpleNamespace(primitive_utility=SimpleNamespace(get_smpl_model=lambda gender: body))
    return current, dataset, body


@pytest.mark.parametrize("changed", sorted(seed.BUFFER_KEYS))
def test_each_actual_runtime_buffer_is_checked_before_feature_math(monkeypatch, changed):
    current, dataset, body = runtime_seed_fixture(monkeypatch)
    getattr(body, changed).value[0, 0] = 1.
    monkeypatch.setattr(static_seed.QueryStaticSeed, "build_p360_sample",
                        lambda *args: pytest.fail("Mismatched runtime body must reject before retained numerical method"))
    with pytest.raises(MemoryContractError, match="arrays differ"):
        current.build_p360_sample(dataset, "cpu", None)


def test_numerical_history_method_receives_current_neutral_object_not_old_carrier(monkeypatch):
    current, dataset, _ = runtime_seed_fixture(monkeypatch)
    seen = []
    def numerical(self, selected_dataset, device, plan):
        seen.append((self, selected_dataset, device, plan))
        return "synthetic_sample", {"gender": "neutral", "smplx_carrier_j22_max_abs_m": 0.,
                                    "world_j22_roundtrip_max_abs_m": 0.}
    monkeypatch.setattr(static_seed.QueryStaticSeed, "build_p360_sample", numerical)
    sample, audit = current.build_p360_sample(dataset, "cpu", None)
    assert seen == [(current, dataset, "cpu", None)]
    assert sample == "synthetic_sample"
    assert audit["gender"] == "neutral"
    assert audit["legacy_P366_male_lineage_claimed"] is False
    assert set(audit["runtime_body_array_hashes"]) == seed.BUFFER_KEYS


@pytest.mark.parametrize("npz_input", [False, True])
def test_compatibility_loader_delegates_to_current_neutral_authority(tmp_path, monkeypatch, npz_input):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    path = tmp_path / "static_seed.npz"
    path.write_bytes(b"not opened by compatibility adapter")
    current = SimpleNamespace(path=path, gender="neutral", authoritative_fixture=True)
    seen = []
    def current_loader(p):
        seen.append(p)
        return current
    monkeypatch.setattr(seed, "load_seed", current_loader)
    assert static_seed.QueryStaticSeed.load(path if npz_input else receipt) is current
    assert seen == [receipt]


def test_npz_compatibility_does_not_accept_another_file_from_sibling_receipt(tmp_path, monkeypatch):
    path = tmp_path / "static_seed.npz"
    path.write_bytes(b"first")
    other = tmp_path / "other.npz"
    other.write_bytes(b"second")
    monkeypatch.setattr(seed, "load_seed", lambda p: SimpleNamespace(path=other))
    with pytest.raises(static_seed.QueryStaticSeedError, match="not the file bound"):
        static_seed.QueryStaticSeed.load(path)


def test_old_receipt_rejected_before_body_hashing(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    write_once(path, {"schema": "p478.query_static_history_seed_receipt.v1", "gender": "male"})
    monkeypatch.setattr(seed, "validate_buffer_identity", lambda value: pytest.fail("must reject old schema before asset I/O"))
    with pytest.raises(MemoryContractError, match="current neutral"):
        static_seed.QueryStaticSeed.load(path)


@pytest.mark.parametrize("yaw", [0., np.pi / 2, -np.pi / 2, np.pi, -np.pi + 1e-7])
def test_coordinate_and_pelvis_carrier_algebra_matches_neutral_frame(yaw, monkeypatch):
    pelvis = np.array([.3, .9, -.2])
    native_vertices = np.array([[.1, 0., -.2], [.5, 0., -.2], [.3, 1.8, -.2], [.3, .9, .1]])
    joints = np.tile(pelvis, (22, 1))
    joints[1] += [-.1, 0., 0.]
    joints[2] += [.1, 0., 0.]
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    monkeypatch.setattr(frame, "_model_forward", lambda *args: (native_vertices.copy(), joints.copy(), faces.copy()))
    carrier = frame._build_at_explicit_xy([1.2, -2.3], Path("/fixture/not_opened"), yaw, .01)
    rotation = carrier["global_orient"]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)
    np.testing.assert_allclose(carrier["root"][:2], [1.2, -2.3], atol=1e-12)
    assert abs(static_seed.wrapped_angle(carrier["root"][3] - yaw)) < 1e-12
    np.testing.assert_allclose(carrier["vertices"].min(axis=0)[2], .01, atol=1e-12)
    np.testing.assert_allclose(carrier["root"][:3], pelvis + carrier["translation"], atol=1e-12)
    np.testing.assert_allclose(carrier["vertices"], (native_vertices - pelvis) @ rotation.T + pelvis + carrier["translation"], atol=1e-12)
    world_orientation, translation, offset = frame._carrier_from_model(Path("/fixture/not_opened"), np.zeros(10), np.zeros(63), carrier["root"])
    np.testing.assert_allclose(world_orientation, rotation, atol=1e-12)
    np.testing.assert_allclose(translation, carrier["translation"], atol=1e-12)
    np.testing.assert_array_equal(offset, pelvis)
    with pytest.raises(MemoryContractError, match="query root"):
        frame._require_runtime_pelvis_translation(carrier["root"], carrier["translation"] + [.001, 0., 0.], pelvis)


@pytest.mark.parametrize("xy,yaw,clearance", [([1.], 0., .01), ([1., np.nan], 0., .01),
                                             ([1., 2.], np.inf, .01), ([1., 2.], 0., -.01),
                                             ([1., 2.], 0., .11)])
def test_invalid_frame_coordinates_reject_before_model(xy, yaw, clearance, monkeypatch):
    monkeypatch.setattr(frame, "_model_forward", lambda *args: pytest.fail("invalid coordinates must reject before body execution"))
    with pytest.raises(MemoryContractError):
        frame._build_at_explicit_xy(xy, Path("/fixture/not_opened"), yaw, clearance)


def test_native_yup_world_zup_rotation_and_angle_contract():
    np.testing.assert_array_equal(np.array([1., 2., 3.]) @ static_seed.WORLD_CONVERSION_YUP_TO_ZUP.T,
                                  [1., -3., 2.])
    np.testing.assert_allclose(static_seed.world_global_orientation(np.pi / 2),
                              static_seed.rotation_z(np.pi / 2) @ seed.YUP_TO_ZUP)
    assert abs(static_seed.wrapped_angle(3 * np.pi) - np.pi) < 1e-12


def test_pure_seed_import_has_no_torch_or_smplx_dependency_and_exports_are_real():
    subprocess.run([sys.executable, "-B", "-c",
        "from hsi.stage3 import seed, static_seed, frame; import sys; "
        "assert 'torch' not in sys.modules and 'smplx' not in sys.modules; "
        "assert all(hasattr(static_seed, name) for name in static_seed.__all__)"], check=True)
    text = Path(static_seed.__file__).read_text()
    assert "_validate_frame_receipt" not in text and "_validate_grounding" not in text
