"""Independent final full-mesh physical recheck; no caller-provided pass authority."""
from pathlib import Path
import numpy as np
import torch
from ._common import artifact, read, read_sealed, require, verified, source_closure, verify_artifact_tree
from . import placement, refine, refine_target, refine_contract, body_geometry as body
from .sdf import BoundSDFCache


def validate(bundle_path):
    bundle = read_sealed(bundle_path)
    require(bundle['schema'] == 'hsi.stage2.physical_keypose_candidate.v1', 'Wrong physical candidate schema')
    require(bundle['source_closure'] == source_closure(), 'Physical source closure changed')
    verify_artifact_tree(bundle)
    stage1_path = verified(bundle['source_stage1'])
    _, target, expected_target, _ = placement.normalized_target(stage1_path)
    selection = read_sealed(verified(bundle['outputs']['selection']))
    require(len(selection['trials']) == 1 and selection['selected_refine_receipt'] == selection['trials'][0],
            'Current policy requires one actual seed-0 trial')
    trial_path = verified(selection['selected_refine_receipt'])
    trial = read(trial_path)
    keypose_path = verified(bundle['outputs']['published_keypose'])
    require(selection['published_keypose'] == bundle['outputs']['published_keypose'], 'Selection/keypose mismatch')
    require(trial['keypose']['sha256'] == artifact(keypose_path)['sha256'], 'Published mesh differs from selected trial')
    verify_artifact_tree(trial)
    normalized_path = verified(trial['inputs']['stage1_target'])
    observed_target = read_sealed(normalized_path)
    require({k: v for k, v in observed_target.items() if k != 'receipt_payload_sha256'} == expected_target,
            'Normalized current target differs')
    require(trial['publication_thresholds'] == placement.GATES and trial['optimizer_initialization_seed'] == 0
            and trial['ablation_mode'] == 'hybrid', 'Current physical policy changed')
    _, _, _, contact_seed, bounds, yaw, resolved = refine_target._validate_target(expected_target,
        artifact_base_dir=normalized_path.parent)
    with np.load(keypose_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    require(str(arrays['schema']) == refine.KEYPOSE_SCHEMA, 'Wrong full-mesh archive schema')
    require(str(arrays['target_instance_id']) == target['target_instance_id']
            and str(arrays['selected_candidate_id']) == target['selected_surface_id']
            and str(arrays['support_component_sha256']) == target['selected_surface_sha256'], 'Mesh target identity differs')
    selected_contact = np.asarray(trial['target_binding']['selected_contact_world_xyz_zup_m'], dtype=float)
    require(np.allclose(selected_contact, arrays['contact_goal_world_xyz_zup_m'], atol=1e-7, rtol=0),
            'Archived contact differs from exact selected trial contact')
    delta = selected_contact - contact_seed
    require(np.max(np.abs(delta[:2])) <= placement.SEARCH['same_surface_xy_search_radius_m'] + 1e-12,
            'Selected contact exceeds current search radius')
    edge = placement.SEARCH['same_surface_edge_margin_m']
    inside = bool(np.all(selected_contact[:2] >= bounds[0, :2] + edge)
                  and np.all(selected_contact[:2] <= bounds[1, :2] - edge))
    model = body.body_util.load_fixed_smplx(verified(trial['inputs']['fixed_neutral_smplx']), torch.device('cpu'))
    pose = torch.as_tensor(arrays['body_pose_axis_angle'], dtype=torch.float32)
    betas = torch.as_tensor(arrays['betas'], dtype=torch.float32)
    frame = torch.as_tensor(arrays['pose_frame_to_world_rotation'], dtype=torch.float32)
    subset = torch.as_tensor(arrays['fixed_generated_butt_vertex_indices'], dtype=torch.long)
    with torch.no_grad():
        state = body.materialize(model, pose, betas, frame, torch.as_tensor(selected_contact, dtype=torch.float32), subset)
        require(np.max(np.abs(state.vertices.numpy() - arrays['vertices_world_zup'])) <= 2e-5
                and np.max(np.abs(state.joints.numpy() - arrays['joints_world_zup'])) <= 2e-5,
                'Saved mesh does not materialize from its actual SMPL-X pose')
        # Evaluate the saved geometry exactly, avoiding a second float32 round trip.
        vertices = torch.as_tensor(arrays['vertices_world_zup'], dtype=torch.float32)
        config = body.target.RefineConfig(steps=1, bilateral_foot_support=True,
            target_allowed_tail_m=placement.GATES['maximum_target_allowed_contact_penetration_m'],
            ground_tail_m=placement.GATES['maximum_ground_penetration_m'], minimum_allowed_contact_vertices=16,
            minimum_foot_ground_vertices=6, maximum_target_forbidden_penetration_m=.02,
            maximum_non_target_penetration_m=.02, maximum_outside_fraction=.01).validate()
        cache = BoundSDFCache(verified(bundle['source_sdf']), stage1_path)
        fields = cache(verified(trial['inputs']['scene_occupancy']), verified(trial['inputs']['target_occupancy_mask']),
            torch.device('cpu'), expected_occupancy_file_sha256=trial['inputs']['scene_occupancy']['sha256'],
            expected_target_mask_file_sha256=trial['inputs']['target_occupancy_mask']['sha256'],
            expected_target_world_sha256=target['target_occupancy_mask_world_xyz_sha256'],
            floor_ignore_height_m=.08, maximum_target_floor_fraction=.1, maximum_target_fraction=.25)
        anatomy = body.target.build_anatomy_masks(model, 10475, config)
        metrics = body._float_diagnostics(body.target.branch_terms(vertices, fields, anatomy, config))
        support = vertices.index_select(0, subset)
        count = min(len(subset), max(8, int(np.ceil(len(subset) * body.SUPPORT_ENVELOPE_FRACTION))))
        anchor = torch.cat((support[:, :2].mean(0), torch.topk(support[:, 2], count, largest=False).values.mean().reshape(1)))
        error = float(torch.linalg.vector_norm(anchor - torch.as_tensor(selected_contact, dtype=torch.float32)))
        metrics['hips_support_inside_surface_fraction'] = body.support_inside_surface_fraction(support.numpy(), bounds)
    for key, value in metrics.items():
        require(abs(float(value) - float(trial['metrics'][key])) <= 2e-6, 'Fresh physical metric drift: ' + key)
    gates = refine_contract.publication_gates(metrics, placement.GATES,
        facing_error_rad=refine_contract.angle_difference(float(arrays['terminal_facing_yaw_rad']), yaw),
        selected_surface_match=str(arrays['support_component_sha256']) == target['selected_surface_sha256'],
        contact_lock_error_m=error)
    gates.update(current_stage1_target_mask_hash_bound=fields.component_receipt['target_component_sha256'] == target['target_occupancy_mask_world_xyz_sha256'],
        direct_target_relation_valid=expected_target['reference_relation'] in ('none', 'beside', 'associated_with_reference_setup'),
        known_action_family_sit=str(arrays['action_family']) == 'sit', contact_point_inside_bound_surface=inside,
        contact_point_on_true_support_mask=resolved.support_mask is None or resolved.support_mask.contains_world_xy(selected_contact[:2]),
        contact_anchor_clearance_bounded=0.0 <= float(delta[2]) <= placement.SEARCH['same_surface_max_clearance_m'] + 1e-12)
    require(len(gates) == 18 and gates == trial['gates'] == bundle['selected_gates'] and all(v is True for v in gates.values()),
            'Fresh recomputation failed one of the exact eighteen physical gates')
    return {'physical_gate_count': 18, 'all_gates_recomputed': True, 'full_mesh_rematerialization_verified': True,
            'keypose': artifact(keypose_path), 'selected_refine_receipt': artifact(trial_path)}
