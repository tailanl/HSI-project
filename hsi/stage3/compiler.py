"""Strict current H3/HybrIK keypose admission to the existing motion network.

This compiler does not synthesize motion or authorize release. It materializes
the existing single-sit route/reorientation/contact contract with fresh neutral
frame0 and actual body-derived proxies. No historical pose provider is renamed.
"""
from __future__ import annotations
import math
from pathlib import Path
import time
import numpy as np
import torch
from hsi.common.artifacts import artifact, digest, read_json, read_sealed, require, verified, write_once
from hsi.stage1.binding import bind, checked
from hsi.stage1.navigation import check_stage1_handoff
from . import guidance, seed as neutral
from .packet import PACKET_SCHEMA, PROVENANCE, build_packet, route_contact_check
from .route_math import extract_route_roots

SCHEMA = 'hsi.stage3.verified_keypose_adapter.v1'


def sources():
    return {p.name: artifact(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _record(value):
    return {k: value[k] for k in ('path', 'bytes', 'sha256')}


def admit(stage2_path, body_assets):
    """Re-read actual final Stage2 evidence, current route and physical inputs."""
    from hsi.stage2.pipeline import validate_success
    from hsi.stage2 import body_geometry as body
    from hsi.stage2.sdf import BoundSDFCache, SETTINGS
    value = validate_success(stage2_path)
    stage1_path = verified(value['source_stage1'])
    check_stage1_handoff(stage1_path)
    stage1 = read_sealed(stage1_path)
    binding = bind(stage1)
    target, bundle = checked(stage1['target']), checked(stage1['bundle'])
    steps = stage1['interaction_plan']['steps']
    require(len(steps) == 1 and steps[0]['action'] == 'sit' and len(steps[0]['target_ids']) == 1
            and target['action_family'] == 'sit', 'Current motion adapter supports one feet-grounded sit')
    physical = checked(value['physical_refine'])
    selection = checked(physical['outputs']['selection'])
    trial = read_json(verified(selection['selected_refine_receipt']))
    candidate_body = verified(trial['inputs']['fixed_neutral_smplx'])
    require(set(body_assets) == neutral.ASSET_KEYS, 'Explicit neutral candidate/runtime/render assets required')
    actual_assets = neutral.verify_neutral_assets(**{k: verified(v) for k, v in body_assets.items()})
    require(actual_assets == body_assets and artifact(candidate_body) == body_assets['candidate_body'],
            'Stage2 and motion neutral body assets differ')
    with np.load(verified(value['keypose']), allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    require(arrays['betas'].shape == (10,) and np.array_equal(arrays['betas'], np.zeros(10)),
            'Current motion network requires neutral zero shape; shape conversion is not implicit')
    arrays['gluteal_support_vertex_ids'] = arrays['fixed_generated_butt_vertex_indices']
    require(arrays['joints_world_zup'].shape == (22, 3) and arrays['vertices_world_zup'].shape == (10475, 3),
            'Full current SMPL-X geometry required')
    surface = next(r for r in target['candidate_surfaces'] if r['candidate_id'] == target['selected_surface_id'])
    option = next(r for r in surface['approach_options'] if r['option_id'] == binding['selected_approach_option_id'])
    with np.load(verified(option['p552_region_member']['region_arrays']), allow_pickle=False) as data:
        triangles = data['contact_triangles'].copy()
    records = {'stage2': artifact(stage2_path), 'candidate': value['keypose'],
        'qwen_review': value['post_refine_qwen'], 'physical_refine': value['physical_refine'],
        'stage1_execution': value['source_stage1'], 'stage1_target': stage1['target'], 'stage1_bundle': stage1['bundle'],
        'key_nodes': bundle['artifacts']['key_nodes'], 'navmesh_route': bundle['artifacts']['navmesh_route'],
        'raw_occupancy': _record(target['artifacts']['scene_occupancy']),
        'target_occupancy_mask': _record(target['artifacts']['target_occupancy_mask']),
        'sdf_cache': value['source_sdf'], 'body_model': artifact(candidate_body),
        'original_scene_mesh': _record(target['artifacts']['original_scene_mesh'])}
    cache = BoundSDFCache(verified(value['source_sdf']), stage1_path)
    fields = cache(verified(records['raw_occupancy']), verified(records['target_occupancy_mask']), torch.device('cpu'),
        expected_occupancy_file_sha256=records['raw_occupancy']['sha256'],
        expected_target_mask_file_sha256=records['target_occupancy_mask']['sha256'],
        expected_target_world_sha256=target['target_occupancy_mask_world_xyz_sha256'], **SETTINGS)
    model = body.body_util.load_fixed_smplx(candidate_body, torch.device('cpu'))
    anatomy = body.target.build_anatomy_masks(model, 10475,
        body.target.RefineConfig(steps=1, bilateral_foot_support=True).validate())
    keys = checked(records['key_nodes'])
    route = read_json(verified(records['navmesh_route']))
    require(keys['inputs']['navmesh_route'] == records['navmesh_route']
        and keys['navmesh_execution_route']['all_segment_samples_human_free'] is True
        and keys['collision_safe_control_polyline']['direct_linear_interpolation_allowed'] is True,
        'Current route/control lineage invalid')
    controls = np.asarray(keys['collision_safe_control_polyline']['nodes_world_xy_m'], dtype=float)
    dense = np.asarray(route['route_world_xy_m'], dtype=float)
    require(controls.ndim == dense.ndim == 2 and controls.shape[1:] == dense.shape[1:] == (2,)
        and len(controls) >= 2 and len(dense) >= 2 and np.isfinite(controls).all() and np.isfinite(dense).all()
        and np.linalg.norm(controls[0]-dense[0]) <= 2e-4 and np.linalg.norm(controls[-1]-dense[-1]) <= 2e-4,
        'Control polyline changed current route endpoints')
    root = np.asarray(keys['start']['source']['root_xyz_yaw_world_zup'], dtype=float)
    require(root.shape == (4,) and np.isfinite(root).all(), 'No bound current start root')
    route_contact_check(controls[-1], float(root[2]), arrays, fields.non_target.sample)
    return dict(stage1=stage1, target=target, binding=binding, surface=surface, option=option,
        triangles=triangles, arrays=arrays, anatomy=anatomy, fields=fields, records=records,
        keys=keys, source_root=root, value=value)


def _guidance_products(output, current, query_id, body_assets):
    arrays, records = current['arrays'], current['records']
    yaw = current['binding']['contact_forward_yaw_rad']
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
    origin = np.asarray(current['surface']['surface']['centre_world_xyz_zup_m'])
    patch = guidance.SurfacePatch(origin_world_zup_m=origin, local_to_world_rotation=rotation,
        triangles_local_m=(current['triangles']-origin)@rotation)
    body = {'schema': 'hsi.current_body_identity.v1', 'model_type': 'smplx', 'gender': 'neutral',
        'joint_order': list(guidance.J22_NAMES), 'body_model_sha256': neutral.NEUTRAL_SHA256,
        'body_assets': body_assets, 'betas_sha256': neutral.array_hash(arrays['betas']),
        'coordinate_frame': 'world_zup', 'units': 'metres', 'source_candidate': records['candidate'],
        'source_stage2': records['stage2']}
    write_once(output/'body_identity.json', body)
    inputs = {'stage1_execution': records['stage1_execution'], 'fixed_geometry': current['stage1']['source_fixed_geometry'],
        'contact_surface': current['option']['p552_region_member']['region_arrays'], 'raw_occupancy': records['raw_occupancy'],
        'body_identity': artifact(output/'body_identity.json'), 'geometry_producer': artifact(__file__)}
    write_once(output/'guidance_input.json', {'schema': guidance.INPUT_SCHEMA,
        'scene_id': current['stage1']['scene_id'], 'query_id': query_id, 'coordinate_frame': 'world_zup',
        'units': 'metres', 'source_bindings': inputs, 'surface_patch': patch.payload()})
    binding = guidance.GuidanceBinding(output/'guidance_input.json')
    vertices, joints, butt = (arrays[k] for k in ('vertices_world_zup','joints_world_zup','gluteal_support_vertex_ids'))
    count = max(8, math.ceil(len(butt)*.15))
    foot_ids = [np.flatnonzero(mask.detach().cpu().numpy()).tolist()
        for mask in (current['anatomy'].left_foot_support, current['anatomy'].right_foot_support)]
    proxy = {'contact_ids': [0], 'contact_drop_m': [float(joints[0,2])-float(np.sort(vertices[butt,2])[:count].mean())],
        'foot_drop_m': [float(joints[i,2])-float(np.sort(vertices[ids,2])[:32].mean())
            for i, ids in zip((10,11),foot_ids)], 'body_identity_sha256': inputs['body_identity']['sha256']}
    require(all(0 <= v <= .30 for v in [*proxy['contact_drop_m'], *proxy['foot_drop_m']]),
        'Body proxy offsets exceed registered bounds; do not clip')
    proof = {'schema': 'hsi.current_body_proxy_offsets.v1', 'source_candidate': records['candidate'],
        'source_stage2': records['stage2'], 'source_body_identity': inputs['body_identity'], 'parameters': proxy,
        'gluteal_support_vertex_ids': butt.tolist(), 'left_foot_vertex_ids': foot_ids[0], 'right_foot_vertex_ids': foot_ids[1],
        'gluteal_lower_envelope_vertex_count': count, 'foot_lower_envelope_count_per_side': 32,
        'contact_proxy_is_not_actual_mesh_contact_validation': True, 'original_sdf_and_release_gates_unchanged': True}
    write_once(output/'guidance_proxy_receipt.json', proof)
    guidance._validate_proxy(proof, binding)
    return body, proxy, foot_ids


def _surface(current, foot_ids):
    target, arrays, records = current['target'], current['arrays'], current['records']
    return {k: target[k] for k in ('target_instance_id','target_class','selected_surface_id','selected_surface_sha256',
        'target_occupancy_mask_world_xyz_sha256')} | {
        'fixed_generated_butt_vertex_indices': arrays['gluteal_support_vertex_ids'].tolist(),
        'butt_vertex_source': records['candidate'], 'left_foot_vertex_indices': foot_ids[0], 'right_foot_vertex_indices': foot_ids[1],
        'selected_approach_option_id': current['binding']['selected_approach_option_id']}


def compile(stage2_path, output, *, runtime_body, render_body):
    from hsi.stage2.pipeline import validate_success
    value = validate_success(stage2_path)
    selection = checked(checked(value['physical_refine'])['outputs']['selection'])
    trial = read_json(verified(selection['selected_refine_receipt']))
    assets = neutral.verify_neutral_assets(candidate_body=verified(trial['inputs']['fixed_neutral_smplx']),
        runtime_body=runtime_body, render_body=render_body)
    started = time.monotonic()
    current = admit(stage2_path, assets)
    source_before = sources()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage1, records = current['stage1'], current['records']
    root = current['source_root']
    seed = neutral.build_seed(output/'neutral_seed', scene_id=stage1['scene_id'], instruction=stage1['instruction'],
        start_xy=root[:2], yaw=root[3], body_assets=assets,
        source_bindings={k: records[k] for k in ('stage1_execution','key_nodes','navmesh_route')})
    seed_path = output/'neutral_seed/receipt.json'
    seed_value = neutral.load_seed(seed_path)
    roots = extract_route_roots(current['keys'], seed_value.root_xyz_yaw_world_zup)
    transition = route_contact_check(roots[-1][:2], float(seed_value.root_xyz_yaw_world_zup[2]), current['arrays'], current['fields'].non_target.sample)
    query_id = digest({'stage1': records['stage1_execution'], 'stage2': records['stage2']})
    body, proxy, feet = _guidance_products(output, current, query_id, assets)
    surface = _surface(current, feet)
    packet = build_packet(scene_id=stage1['scene_id'], instruction=stage1['instruction'], roots=roots,
        arrays=current['arrays'], source_artifacts=records, target_contract=surface)
    write_once(output/'packet.json', packet)
    for row in [*source_before.values(), *records.values()]:
        verified(row)
    result = {'schema': SCHEMA, 'status': 'compiled_verified_keypose_not_motion_release', 'producer': PROVENANCE,
        'scene_id': stage1['scene_id'], 'instruction': stage1['instruction'], 'query_id': query_id,
        'inputs': records, 'packet': artifact(output/'packet.json'), 'static_seed': seed['seed'],
        'static_seed_receipt': artifact(seed_path), 'body_identity': body, 'surface_contract': surface,
        'route_contact_validation': transition, 'guidance_input': artifact(output/'guidance_input.json'),
        'guidance_proxy_map': proxy, 'guidance_proxy_receipt': artifact(output/'guidance_proxy_receipt.json'),
        'implementation_sources': source_before, 'motion_generated': False, 'motion_release_authorized': False,
        'real_positive_credit_allowed': False, 'elapsed_seconds': time.monotonic()-started}
    return write_once(output/'adapter.json', result)


def load_adapter(path):
    value = read_sealed(path)
    require(value['schema'] == SCHEMA and value['producer'] == PROVENANCE
        and value['implementation_sources'] == sources(), 'Unregistered or changed motion adapter')
    current = admit(verified(value['inputs']['stage2']), value['body_identity']['body_assets'])
    require(current['records'] == value['inputs'], 'Adapter input lineage changed')
    seed_receipt = checked(value['static_seed_receipt'])
    seed = neutral.load_seed(verified(value['static_seed_receipt']))
    require(seed_receipt['seed'] == value['static_seed'] and seed_receipt['source_bindings'] ==
        {k: value['inputs'][k] for k in ('stage1_execution','key_nodes','navmesh_route')}, 'Static seed bound to another route')
    roots = extract_route_roots(current['keys'], seed.root_xyz_yaw_world_zup)
    actual = route_contact_check(roots[-1][:2], float(seed.root_xyz_yaw_world_zup[2]), current['arrays'], current['fields'].non_target.sample)
    require(actual == value['route_contact_validation'], 'Route contact proof changed')
    binding = guidance.GuidanceBinding(verified(value['guidance_input']))
    proof = checked(value['guidance_proxy_receipt'])
    guidance._validate_proxy(proof, binding)
    require(proof['parameters'] == value['guidance_proxy_map'], 'Proxy declaration differs')
    feet = [proof['left_foot_vertex_ids'], proof['right_foot_vertex_ids']]
    require(_surface(current, feet) == value['surface_contract'], 'Target/body surface contract differs')
    expected = build_packet(scene_id=current['stage1']['scene_id'], instruction=current['stage1']['instruction'], roots=roots,
        arrays=current['arrays'], source_artifacts=current['records'], target_contract=value['surface_contract'])
    packet = checked(value['packet'])
    require({k:v for k,v in packet.items() if k != 'receipt_payload_sha256'} == expected,
        'Packet is not an exact current route plus verified terminal keypose')
    return value, packet, seed_receipt
