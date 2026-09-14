"""Export verified historical Stage1/2 inputs without pretending to retarget bodies.

Run with the existing Python3.10 SMPL-X environment; the physics runtime is3.8.
Full-body keypose and permissions are retained even when a controller cannot
consume them. No teacher motion, internal timestamps, or new Stage1/2 calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'closd_unihsi_sdf_stage3_20260912' / 'code'
sys.path.insert(0, str(OLD))
from stage12_conditions import load_case


def artifact(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = load_case(args.receipt, device='cpu')
    args.output.mkdir(parents=True, exist_ok=False)
    names = ('initial_pose', 'target_pose', 'betas', 'initial_joints', 'target_joints',
             'route_xy', 'contact_triangles', 'contact_goal', 'contact_normal',
             'contact_joint_ids', 'contact_vertex_ids', 'target_root_xyz')
    arrays = {name: getattr(case, name).detach().cpu().numpy() for name in names}
    arrays['terminal_yaw'] = np.asarray(case.terminal_yaw)
    arrays['arrival_yaw'] = np.asarray(case.arrival_yaw)
    target_path = args.output / 'stage12_arrays.npz'
    with target_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    manifest = dict(schema='agent9.paper_structure.stage12_packet.v1', scene_id=case.scene_id,
                    instruction=case.instruction, target_id=case.target_id,
                    target_class=case.target_class, surface_id=case.surface_id,
                    action_family=case.action_family, arrays=artifact(target_path),
                    scene_mesh=case.scene_mesh, sdf=case.sdf, permissions=case.permissions,
                    keypoint_descriptions=case.keypoint_descriptions, provenance=case.provenance,
                    source=artifact(Path(__file__)), loader_source=artifact(OLD/'stage12_conditions.py'),
                    coordinate_system='world_zup_m', new_stage1_or_stage2_inference=False,
                    simulation_run=False, humanoid_retargeting_validated=False,
                    fullbody_keypose_consumption_by_controller_validated=False,
                    original_smplx_shape_and_pose_preserved=True)
    with (args.output/'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(scene=case.scene_id, manifest=str(args.output/'manifest.json'),
                         retargeting_validated=False)))


if __name__ == '__main__':
    main()
