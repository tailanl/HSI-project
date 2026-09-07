"""Pose-independent distance-transform preparation and current-mask revalidation."""
from pathlib import Path
import time
import numpy as np
from ._common import artifact, read_sealed, verified, write_once, require
from . import bound_sdf as kernel

SCHEMA = 'hsi.stage2.bound_sdf_cache.v1'
SETTINGS = {'floor_ignore_height_m': .08, 'maximum_target_floor_fraction': .10,
            'maximum_target_fraction': .25}


def source_closure():
    from . import refine_contract, refine_physics
    return {name: artifact(path) for name, path in (
        ('cache', __file__), ('kernel', kernel.__file__),
        ('contract', refine_contract.__file__), ('physics', refine_physics.__file__))}


def prepare(stage1_path, output):
    from hsi.stage1.binding import bind, checked
    started = time.monotonic()
    stage1 = read_sealed(stage1_path)
    bind(stage1)
    target = checked(stage1['target'])
    records = {name: target['artifacts'][key] for name, key in
               (('occupancy', 'scene_occupancy'), ('target_mask', 'target_occupancy_mask'))}
    occupancy, mask = (verified(records[name]) for name in ('occupancy', 'target_mask'))
    masks = kernel.load_bound_masks(occupancy, mask,
        expected_occupancy_file_sha256=records['occupancy']['sha256'],
        expected_target_mask_file_sha256=records['target_mask']['sha256'],
        expected_target_world_sha256=target['target_occupancy_mask_world_xyz_sha256'], **SETTINGS)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name, mask in (('target_sdf', masks.target_world_xyz), ('collision_sdf', masks.collision_world_xyz)):
        path = output / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, kernel._signed_distance(mask), allow_pickle=False)
        arrays[name] = artifact(path)
    return write_once(output / 'receipt.json', {'schema': SCHEMA,
        'source_kind': 'current_new_scene', 'source_binding': {'stage1_execution': artifact(stage1_path),
            'target': stage1['target']}, 'inputs': records, 'settings': SETTINGS,
        'mask_receipt': masks.receipt, 'arrays': arrays, 'source_closure': source_closure(),
        'elapsed_seconds': time.monotonic() - started, 'pose_image_or_future_motion_used': False})


class BoundSDFCache:
    def __init__(self, path, stage1_path=None):
        self.path = Path(path).resolve(strict=True)
        self.receipt = read_sealed(self.path)
        require(self.receipt['schema'] == SCHEMA, 'Wrong integrated SDF cache schema')
        require(self.receipt['source_closure'] == source_closure(), 'SDF computation source changed')
        require(self.receipt['settings'] == SETTINGS, 'SDF policy changed')
        for record in self.receipt['inputs'].values():
            verified(record)
        if stage1_path is not None:
            require(self.receipt['source_binding']['stage1_execution'] == artifact(stage1_path),
                    'SDF cache belongs to another Stage1 plan')
        self.arrays = {name: np.load(verified(record), allow_pickle=False)
                       for name, record in self.receipt['arrays'].items()}
        require(set(self.arrays) == {'target_sdf', 'collision_sdf'}, 'Incomplete SDF cache')
        for array in self.arrays.values():
            require(array.dtype == np.float32 and array.shape == kernel.contract.WORLD_OCCUPANCY_SHAPE
                    and np.isfinite(array).all(), 'Invalid cached SDF tensor')
        self.calls = []

    def __call__(self, occupancy, target_mask, device, **kwargs):
        started = time.monotonic()
        masks = kernel.load_bound_masks(occupancy, target_mask, **kwargs)
        require(all(float(kwargs.get(k, v)) == v for k, v in SETTINGS.items()), 'Cached SDF policy differs')
        require(masks.receipt == self.receipt['mask_receipt'], 'Cached SDF does not match current masks')
        import torch
        lower = torch.tensor(kernel.contract.GRID_LOWER_XYZ, dtype=torch.float32, device=device)
        upper = torch.tensor(kernel.contract.GRID_UPPER_XYZ, dtype=torch.float32, device=device)
        fields = kernel.p478.TargetAwareFields(
            target=kernel.p478.WorldGridSDF(torch.as_tensor(self.arrays['target_sdf'], device=device), lower, upper),
            non_target=kernel.p478.WorldGridSDF(torch.as_tensor(self.arrays['collision_sdf'], device=device), lower, upper),
            target_mask=masks.target_world_xyz, collision_mask=masks.collision_world_xyz,
            component_receipt=masks.receipt)
        self.calls.append({'device': str(device), 'current_mask_gates_rechecked': True,
                          'elapsed_seconds': time.monotonic() - started})
        return fields
