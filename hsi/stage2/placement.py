"""Current-plan placement of actual HybrIK articulation, followed by current refine.

The old transient prior/provider aliases are not needed here: only the reviewed
numeric placement (fixed gluteal subset, yaw rotation, derived root) is retained.
"""
from pathlib import Path
import math
import shutil
import time
import numpy as np
import torch
from ._common import artifact, read, read_sealed, verified, write_once, require, source_closure
from . import body_geometry as body, refine_target, refine_contract, recovery_contract, refine
from .sdf import BoundSDFCache
from hsi.stage1.binding import bind, checked
from hsi.stage1.navigation import check_stage1_handoff

SEARCH = {'same_surface_xy_search_radius_m': .08, 'same_surface_xy_search_step_m': .02,
          'same_surface_max_clearance_m': .04, 'same_surface_clearance_step_m': .005,
          'same_surface_edge_margin_m': .03}
GATES = {'maximum_non_target_scene_penetration_m': .02,
         'maximum_target_forbidden_body_penetration_m': .02,
         'maximum_target_allowed_contact_penetration_m': .03,
         'minimum_target_allowed_contact_vertices': 16, 'maximum_ground_penetration_m': .02,
         'minimum_left_foot_ground_vertices': 6, 'minimum_right_foot_ground_vertices': 6,
         'minimum_hips_support_inside_surface_fraction': .8, 'maximum_sdf_outside_fraction': .01,
         'maximum_terminal_facing_error_rad': math.radians(5)}


def normalized_target(stage1_path):
    stage1 = read_sealed(stage1_path)
    check_stage1_handoff(stage1_path)
    bound = bind(stage1)
    target, bundle = checked(stage1['target']), checked(stage1['bundle'])
    matches = [row for row in target['candidate_surfaces'] if row['candidate_id'] == bound['selected_surface_id']]
    require(len(matches) == 1, 'Current surface must be unique')
    candidate = matches[0]
    options = [row for row in candidate['approach_options'] if row['option_id'] == bound['selected_approach_option_id']]
    require(len(options) == 1, 'Current selected approach must be unique')
    route = read(verified(bundle['artifacts']['navmesh_route']))
    points = np.asarray(route['route_world_xy_m'], dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 and np.isfinite(points).all(), 'Invalid route')
    arrival = None
    for index in range(len(points) - 2, -1, -1):
        delta = points[-1] - points[index]
        if np.linalg.norm(delta) >= 1e-6:
            arrival = math.atan2(float(delta[1]), float(delta[0]))
            break
    require(arrival is not None, 'Route has no terminal tangent')
    relation = refine_contract.normalize_relation(candidate.get('reference_relation', target.get('reference_relation')))
    normalized = {'schema': body.TARGET_SCHEMA, 'status': 'published_current_stage1_target_for_keypose',
        'scene_id': stage1['scene_id'], 'instruction': stage1['instruction'], 'action_family': 'sit',
        'current_only_stage1_verified': True, 'target_instance_id': target['target_instance_id'],
        'target_class': target['target_class'], 'selected_candidate_id': target['selected_surface_id'],
        'support_component_sha256': target['selected_surface_sha256'],
        'target_occupancy_mask_world_xyz_sha256': target['target_occupancy_mask_world_xyz_sha256'],
        'support_class': target['target_class'], 'reference_relation': relation,
        'surface': candidate['surface'], 'approach': options[0], 'root_chain_world_xy_m': points.tolist(),
        'source_stage1': artifact(stage1_path), 'source_target': stage1['target'],
        'source_route': bundle['artifacts']['navmesh_route']}
    for key in ('producer_variant_schema', 'p530_action_space_audit', 'filled_contact_and_shape_filtered_approach_mask',
                'p530_support_mask_artifact', 'stage2_contact_support_mask'):
        if key in target:
            normalized[key] = target[key]
    refine_target._validate_target(normalized, artifact_base_dir=Path(stage1['target']['path']).parent)
    return stage1, target, normalized, arrival


def run(stage1_path, recovery_path, projection_path, sdf_path, output, *, smplx_model, segmentation, steps=180):
    started = time.monotonic()
    stage1, target, normalized, arrival_yaw = normalized_target(stage1_path)
    recovery = read_sealed(recovery_path)
    prior_path = verified(recovery['outputs']['articulation_prior'])
    provider = recovery_contract.validate_h3_hybrikx_prior(prior_path, Path(recovery_path),
        scene_id=stage1['scene_id'], instruction=stage1['instruction'], target_instance_id=target['target_instance_id'],
        target_surface_id=target['selected_surface_id'])
    require(recovery['inputs']['stage1_artifact'] == stage1['bundle'], 'Recovery belongs to another Stage1')
    require(recovery['execution_source']['recovery'] == artifact(Path(__file__).with_name('recovery.py')),
            'Recovery did not execute the current integrated source')
    projection = read_sealed(projection_path)
    require(projection['inputs']['camera'] == provider['artifacts']['stage1_camera']
            and projection['inputs']['h3_image'] == provider['artifacts']['h3_image'], 'Projection evidence drift')
    smplx_model, segmentation = Path(smplx_model).resolve(strict=True), Path(segmentation).resolve(strict=True)
    require(artifact(smplx_model) == recovery['models']['neutral_smplx'], 'Refine must use recovery neutral carrier')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    normalized_path = output / 'normalized_target.json'
    write_once(normalized_path, normalized)
    _, _, centre, contact_xyz, bounds, terminal_yaw, resolved = refine_target._validate_target(normalized,
        artifact_base_dir=normalized_path.parent)
    cache = BoundSDFCache(sdf_path, stage1_path)
    occupancy, mask = (verified(target['artifacts'][key]) for key in ('scene_occupancy', 'target_occupancy_mask'))
    with np.load(prior_path, allow_pickle=False) as data:
        prior = {key: np.asarray(data[key]) for key in data.files}
    source_root = prior['root_xyz_yaw'].astype(np.float64).reshape(4)
    source_pose = prior['body_pose_axis_angle'].astype(np.float64).reshape(21, 3)
    source_betas = prior['betas'].astype(np.float64).reshape(10)
    source_frame = prior['pose_frame_to_world_rotation'].astype(np.float64).reshape(3, 3)
    subset_np = body.hips_support_subset(prior['vertices_world_zup'].astype(np.float64), read(segmentation))
    delta_yaw = math.atan2(math.sin(terminal_yaw-source_root[3]), math.cos(terminal_yaw-source_root[3]))
    frame_np = body.rotation_z(delta_yaw) @ source_frame
    device = torch.device('cpu')
    model = body.body_util.load_fixed_smplx(smplx_model, device)
    pose = torch.as_tensor(source_pose, dtype=torch.float32)
    betas = torch.as_tensor(source_betas, dtype=torch.float32)
    contact = torch.as_tensor(contact_xyz, dtype=torch.float32)
    frame = torch.as_tensor(frame_np, dtype=torch.float32)
    subset = torch.as_tensor(subset_np, dtype=torch.long)
    with torch.no_grad():
        state = body.materialize(model, pose, betas, frame, contact, subset)
    turn = refine_contract.angle_difference(arrival_yaw, terminal_yaw)
    lineage = {
        'source_h3_model_manifest_sha256': str(prior['source_h3_model_manifest_sha256']),
        'source_h3_generation_receipt_sha256': provider['h3_generation_receipt_sha256'],
        'source_h3_frame_sha256': provider['source_h3_frame_sha256'],
        'source_hybrikx_model_sha256': str(prior['source_hybrikx_model_sha256']),
        'source_hybrikx_raw_recovery_sha256': provider['source_raw_recovery_sha256'],
        'source_hybrikx_recovery_receipt_sha256': provider['hybrikx_recovery_receipt_sha256']}
    arrays = {'schema': refine.SOURCE_SCHEMA, 'status': 'published_current_stage1_conditioned_before_refine',
        'selected_candidate_id': target['selected_surface_id'], 'support_component_sha256': target['selected_surface_sha256'],
        'target_occupancy_mask_world_xyz_sha256': target['target_occupancy_mask_world_xyz_sha256'],
        'body_pose_axis_angle': source_pose.astype(np.float32), 'betas': source_betas.astype(np.float32),
        'root_xyz_yaw': np.asarray([*state.root.numpy(), terminal_yaw], dtype=np.float32),
        'pose_frame_to_world_rotation': frame_np.astype(np.float32), 'vertices_world_zup': state.vertices.numpy().astype(np.float32),
        'faces': prior['faces'], 'route_arrival_tangent_yaw_rad': np.float32(arrival_yaw),
        'terminal_facing_yaw_rad': np.float32(terminal_yaw), 'arrival_to_terminal_turn_rad': np.float32(turn),
        'expected_arrival_to_terminal_turn_rad': np.float32(turn), 'route_arrival_tangent_used_as_terminal_facing': False,
        'legacy_fixed_pi_expectation_used': False, 'h3_model_used': True, 'hybrikx_model_used': True,
        'camera_world_placement_discarded': True, 'old_world_placement_consumed': False, **lineage}
    generated = output / 'stage1_conditioned_keypose.npz'
    with generated.open('xb') as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in arrays.items()})
    trial = refine.run(normalized_path, generated, occupancy, mask, smplx_model, segmentation, output / 'refined_seed0',
        steps=steps, device_name='cpu', initialization_seed=0, search=SEARCH, publication_thresholds=GATES,
        expected_occupancy_file_sha256=target['artifacts']['scene_occupancy']['sha256'],
        expected_target_mask_file_sha256=target['artifacts']['target_occupancy_mask']['sha256'],
        expected_target_world_sha256=target['target_occupancy_mask_world_xyz_sha256'], ablation_mode='hybrid',
        projection_observation_path=Path(projection_path), projection_weight=20., depth_weight=4.,
        lower_limb_tail_weight=500., field_builder=cache)
    trial_path = output / 'refined_seed0/receipt.json'
    passed = len(trial['gates']) == 18 and all(value is True for value in trial['gates'].values())
    keypose = None
    if passed:
        final = output / 'published_keypose.npz'
        shutil.copyfile(verified(trial['keypose']), final)
        keypose = artifact(final)
    selection = output / 'selection.json'
    write_once(selection, {'schema': 'hsi.stage2.physical_selection.v1', 'trials': [artifact(trial_path)],
        'selected_refine_receipt': artifact(trial_path) if passed else None, 'published_keypose': keypose})
    return write_once(output / 'receipt.json', {'schema': 'hsi.stage2.physical_keypose_candidate.v1',
        'status': 'physical_candidate' if passed else 'vetoed_physical_candidate', 'scene_id': stage1['scene_id'],
        'instruction': stage1['instruction'], 'source_stage1': artifact(stage1_path), 'source_recovery': artifact(recovery_path),
        'source_projection': artifact(projection_path), 'source_sdf': artifact(sdf_path), 'source_closure': source_closure(),
        'target_binding': trial['target_binding'], 'selected_gates': trial['gates'], 'selected_metrics': trial['metrics'],
        'outputs': {'published_keypose': keypose, 'selection': artifact(selection)},
        'stage3_handoff_allowed': passed, 'numeric_internal_handoff_only': True,
        'verified_stage2_keypose': False, 'semantic_publication_granted': False,
        'sdf_current_mask_rechecks': cache.calls, 'elapsed_seconds': time.monotonic() - started})
