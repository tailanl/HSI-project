"""Actual calibrated projection and sealed IO; no model/device doubles needed."""
import numpy as np
import pytest
from PIL import Image

from hsi.common.artifacts import read_sealed, verified, write_once
from hsi.stage2 import projection


def test_current_image_projection_writes_sealed_receipt_and_nearest_depth(tmp_path):
    camera = tmp_path / 'camera.json'
    K = np.array([[320., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    write_once(camera, {'width': 640, 'height': 480, 'K': K.tolist(),
        'world_to_camera': np.eye(4).tolist(), 'extrinsic_convention': 'opencv_world_to_camera'})
    derivation = tmp_path / 'derivation.json'
    write_once(derivation, {'crop_xyxy': [64, 48, 576, 432], 'output_size_wh': [512, 512]})
    depth_path = tmp_path / 'depth.npy'
    depth = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
    np.save(depth_path, depth)
    image = tmp_path / 'image.png'
    Image.new('RGB', (512, 512), (12, 23, 34)).save(image)
    output = tmp_path / 'projection'
    result = projection.run(camera, derivation, depth_path, image, output)
    sealed = read_sealed(output / 'projection_observation.json')
    assert result == sealed
    affine = np.array([[1., 0., -64.], [0., 512. / 384., -64.], [0., 0., 1.]])
    np.testing.assert_array_equal(result['camera']['observation_intrinsics'], affine @ K)
    record = {key: result['depth_m'][key] for key in ('path', 'bytes', 'sha256')}
    aligned = np.load(verified(record), allow_pickle=False)
    y = np.floor(np.arange(512) * 384. / 512.).astype(int) + 48
    expected = depth[y[:, None], np.arange(64, 576)[None, :]]
    np.testing.assert_array_equal(aligned, expected)
    assert result['transform_contract']['condition_to_image_transform_is_identity'] is True
    assert result['inputs']['h3_image']['path'] == str(image)
    with pytest.raises(projection.contract.P533ContractError, match='overwrite'):
        projection.run(camera, derivation, depth_path, image, output)
