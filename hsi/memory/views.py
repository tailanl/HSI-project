"""Persistent pre-keypose camera hints, NOT successful Affordance Memory.

save_selection(facts_path, view_receipt_path, store_root) returns a sealed entry
Path. query_views(facts_path, stage1_path, store_root) returns a JSON-compatible
lookup; absent stores are read-only misses. The native selector consumes hints only after fresh current-scene checks.
All historical gates are observations, never authority for a new task. The
native route binder is called only to verify the current target/direction; no
NavMesh object, route array, pose, motion or Qwen answer is saved in this bank.

Atomic rename publishes complete immutable entries; interrupted staging folders
are not queried. SHA protects against accidental local drift, not a malicious
same-UID process capable of rewriting both source evidence and its hashes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from hsi.common.artifacts import (artifact, canonical_bytes, digest,
                           fsync_directory, read_json, read_sealed, require,
                           verified, write_once)
from .facts import load_scene_facts
from hsi.stage1 import binding as stage1_binding

SOURCE = Path(__file__).resolve()
HERE = SOURCE.parent
SCHEMA = 'p555.pre_keypose_view_hint.v1'
STORE_SCHEMA = 'p555.geometry_view_hint_store.v1'
LOOKUP_SCHEMA = 'p555.geometry_view_hint_lookup.v1'
POLICY = dict(surface_area_in_crop_min=.95, surface_mask_visible_fraction_min=.85,
    surface_mask_pixels_min=300, owner_bbox_retained_fraction_min=.98,
    owner_mask_visible_fraction_min=.90, head_probe_visible_fraction_min=.95,
    each_foot_probe_visible_fraction_min=.85, depth_tolerance_m=.03,
    surface_area_samples=5000, sample_seed=549, body_probe_is_not_a_pose_prediction=True,
    hypothetical_body_probes_are_ranking_only=True, camera_contact_facing_is_ranking_only=True,
    generated_body_visibility_checked_after_H3_by_actual_Qwen=True,
    downstream_full_mesh_physical_gates_unchanged=True)
FRESH = dict(current_target_and_direction_binding=True, crop_and_intrinsics=True,
    current_target_owner_surface_masks=True, actual_new_body_depth_and_visibility_masks=True,
    actual_new_images_qwen_review=True, unchanged_full_body_physical_gates=True)
CAMERA_KEYS = {'K','intrinsics','anchor_id','camera_id','coordinate_system','depth_units',
    'depth_zero_semantics','extrinsic_convention','height','width','intrinsics_convention',
    'position_world_zup_m','target_world_zup_m','schema','world_to_camera','yaw_degrees','yaw_index'}


def sources():
    """Actual consolidated sources, never the historical selector's source pins."""
    from hsi.stage2 import views as selector
    from hsi.common import artifacts
    rows = [artifact(SOURCE), artifact(HERE/'facts.py'), artifact(HERE/'surface.py'),
            artifact(artifacts.__file__), artifact(stage1_binding.__file__), *selector.source_closure()]
    return [record for _, record in sorted({row['path']: row for row in rows}.items())]


def _bind(stage1):
    return stage1_binding.bind(stage1)


def _context(facts_path, stage1_path):
    facts_binding, stage_binding = artifact(facts_path), artifact(stage1_path)
    facts = load_scene_facts(facts_path)
    stage1 = read_sealed(stage1_path)
    require(stage1.get('schema') == 'p550.stage1_query_execution.v1', 'Unsupported current Stage1 schema')
    require(stage1['source_fixed_geometry'] == facts['sources']['fixed_geometry'], 'Stale facts/current geometry')
    require(stage1['scene_id'] == facts['scene_id'], 'Cross-scene facts')
    bound = _bind(stage1)
    target = read_sealed(verified(stage1['target']))
    require(target['scene_id'] == facts['scene_id'] == bound['scene_id'], 'Cross-scene target')
    require(target['action_family'] == 'sit', 'Only pre-keypose sit generation hints are registered')
    require(stage1['instruction'] == target['instruction'] == bound['instruction'], 'Current instruction binding drift')
    rows = [r for r in facts['objects'] if r['instance_id'] == bound['target_instance_id']]
    require(len(rows) == 1 and rows[0]['descriptor_eligible'] is True, 'No audited current target surface')
    row = rows[0]
    surfaces = [r for r in target['candidate_surfaces'] if r['candidate_id'] == bound['selected_surface_id']]
    require(len(surfaces) == 1, 'Selected surface not unique')
    surface = surfaces[0]
    owner = surface['furniture_instance_binding']
    require(owner['verified'] is True and owner['surface_inside_target_furniture_verified'] is True
        and owner['graph_target_instance_id'] == row['instance_id']
        and owner['resolved_target_class'] == row['category'], 'Current owner binding mismatch')
    require(surface['surface'] == row['surface'], 'Current surface geometry differs from facts')
    with np.load(verified(surface['surface_face_archive']), allow_pickle=False) as a, \
            np.load(verified(row['surface_arrays']), allow_pickle=False) as b:
        require(str(a['target_instance_id'].item()) == row['instance_id'], 'Surface archive target mismatch')
        for left, right in (('mesh_face_indices','face_ids'), ('surface_vertex_ids','vertex_ids'),
                            ('target_support_vertex_ids','support_ids')):
            require(np.array_equal(a[left], b[right]), 'Current owner/surface arrays differ: '+left)
    directions = row['direction_hypotheses']['candidates']
    selected = [d for d in directions if d['candidate_id'] == bound['direction_hypothesis_id']]
    require(len(selected) == 1, 'No audited current direction')
    direction = selected[0]
    forward = np.asarray(direction['outward_world_xy'], dtype=float)
    require(forward.shape == (2,) and np.isfinite(forward).all() and abs(np.linalg.norm(forward)-1)<1e-8
        and np.allclose(forward,[math.cos(bound['contact_forward_yaw_rad']),math.sin(bound['contact_forward_yaw_rad'])],
                        atol=1e-8,rtol=0), 'Current direction mismatch')
    for name, src in (('original_scene_mesh','source_mesh'), ('scene_occupancy','source_occupancy')):
        require({k:target['artifacts'][name][k] for k in ('path','bytes','sha256')} == facts['sources'][src],
                'Current target/scene asset mismatch')
    key = dict(schema='p555.pre_keypose_geometry_identity.v1', scene_id=facts['scene_id'],
        facts_sha256=facts_binding['sha256'], fixed_geometry_sha256=facts['sources']['fixed_geometry']['sha256'],
        fixed_semantics_sha256=facts['sources']['fixed_semantics']['sha256'],
        mesh_sha256=facts['sources']['source_mesh']['sha256'], occupancy_sha256=facts['sources']['source_occupancy']['sha256'],
        target_id=row['instance_id'], target_semantic=row['category'], action='sit',
        role='pre_keypose_generation', stable_surface_sha256=row['stable_surface_sha256'],
        surface_arrays_sha256=row['surface_arrays']['sha256'], surface_geometry_sha256=digest(row['surface']),
        owner_mask_sha256=owner['target_occupancy_mask_world_xyz_sha256'],
        direction_id=direction['candidate_id'], outward_world_xy=forward.tolist())
    verified(facts_binding); verified(stage_binding)
    return dict(key=key, context_key=digest(key), facts=facts, facts_binding=facts_binding,
                stage1=stage1, stage_binding=stage_binding, target=target, surface=surface)


def validate_camera(camera, crop):
    require(set(camera) <= CAMERA_KEYS and camera.get('schema') == 'p508.lingo_fullscene_camera.v1',
            'Unknown/non-camera fields or camera schema')
    require(camera.get('coordinate_system') == 'world_zup_metric'
        and camera.get('extrinsic_convention') == 'opencv_world_to_camera'
        and camera.get('intrinsics_convention') == 'pixels_fx_fy_cx_cy_origin_top_left', 'Camera conventions differ')
    w,h = camera.get('width'),camera.get('height')
    require(type(w) is int and type(h) is int and 0<w<=8192 and 0<h<=8192, 'Invalid camera dimensions')
    require(type(camera.get('camera_id')) is int and camera['camera_id']>=0, 'Invalid camera ID')
    K,W = np.asarray(camera.get('K'), dtype=float), np.asarray(camera.get('world_to_camera'), dtype=float)
    require(K.shape == (3,3) and W.shape == (4,4) and np.isfinite(K).all() and np.isfinite(W).all(), 'Invalid camera matrices')
    require(K[0,0]>0 and K[1,1]>0 and 0<=K[0,2]<=w and 0<=K[1,2]<=h
        and K[0,1]==K[1,0]==0 and np.array_equal(K[2],[0,0,1])
        and np.array_equal(W[3],[0,0,0,1]), 'Invalid camera projection')
    require(np.allclose(W[:3,:3] @ W[:3,:3].T,np.eye(3),atol=1e-6,rtol=0)
        and abs(np.linalg.det(W[:3,:3])-1)<1e-6, 'Camera rotation not rigid/right-handed')
    if 'intrinsics' in camera: require(np.array_equal(K,np.asarray(camera['intrinsics'])), 'Duplicate intrinsics differ')
    for field in ('position_world_zup_m','target_world_zup_m'):
        point = np.asarray(camera[field],dtype=float)
        require(point.shape == (3,) and np.isfinite(point).all(), 'Invalid camera '+field)
    require(np.allclose(W[:3,:3] @ camera['position_world_zup_m']+W[:3,3],0,atol=1e-6,rtol=0),
            'Camera position/extrinsic mismatch')
    require(isinstance(crop,list) and len(crop)==4 and all(type(x) is int for x in crop), 'Invalid crop type')
    x0,y0,x1,y1=crop
    require(0<=x0<x1<=w and 0<=y0<y1<=h, 'Crop outside camera')
    canonical_bytes(camera)


def _selection(facts_path, view_path):
    from hsi.stage2 import views as selector
    closure=sources(); view_binding=artifact(view_path); view=read_sealed(view_path)
    require(view.get('schema') == selector.SCHEMA and view.get('mode') == 'full',
            'Only complete native full-mode view selections may enter this hint bank')
    current_stage1_path=verified(view['source_stage1_execution'])
    require(selector.validate_receipt(view_path,current_stage1_path)==view, 'Native selector validation differs')
    require(view.get('status') == 'geometry_view_ready' and type(view.get('qwen_calls')) is int
        and view['qwen_calls']==0 and view.get('numeric_geometry_computed_by_qwen') is False,
        'Only recorded pre-keypose numeric selections are accepted')
    require(view['policy']==POLICY, 'Selector policy drift')
    context=_context(facts_path,current_stage1_path)
    if view.get('current_geometry_identity') is not None:
        require(view['current_geometry_identity']==context['key'], 'Selector/facts geometry identity differs')
    require(view['target']==context['stage1']['target']
        and view['surface_faces']==context['surface']['surface_face_archive'], 'Selection target/surface mismatch')
    selected=view['selection']; camera=read_json(verified(selected['selected_camera']))
    crop=selected['selected_crop_xyxy']; validate_camera(camera,crop)
    require(type(selected['selected_view_index']) is int and selected['selected_view_index']>=0
        and camera['camera_id']==selected['selected_view_index'], 'Selected camera ID mismatch')
    render=read_sealed(verified(view['source_render']))
    require(render['scene_id']==context['key']['scene_id'], 'Cross-scene render')
    for name,src in (('original_scene_mesh','source_mesh'),('query_occupancy','source_occupancy')):
        require(render['inputs'][name]==context['facts']['sources'][src], 'Render geometry drift')
    rows=[r for r in render['views'] if r['view_index']==selected['selected_view_index']]
    require(len(rows)==1 and rows[0]['camera']==selected['selected_camera']
        and rows[0]['rgb']==selected['selected_source_rgb'], 'Selected camera not in original render')
    for field in ('selected_source_rgb','selected_stage2_crop','selected_stage2_crop_target_overlay'): verified(selected[field])
    verified(rows[0]['depth'])
    audits=view['geometry_audit']
    require(len(audits)==render['view_count'] and len({r['view_index'] for r in audits})==len(audits), 'Candidate inventory differs')
    winner=[r for r in audits if r['candidate_id']==selected['selected_candidate_id']]
    require(len(winner)==1 and winner[0]['view_index']==selected['selected_view_index']
        and winner[0]['derived_crop_xyxy']==crop and winner[0]['all_geometry_gates_passed'] is True,
        'Historical selected candidate binding differs')
    eligible=[r for r in audits if r['all_geometry_gates_passed'] is True]
    eligible.sort(key=lambda r:(-r['mask_metrics']['surface_mask_visible_fraction'],
        -min(r['mask_metrics']['probe_visibility'].values()),-r['mask_metrics']['owner_mask_visible_fraction'],
        -float(r['camera_contact_facing_cosine']) if r['camera_contact_facing_cosine'] is not None else 1.,
        -r['mask_metrics']['surface_visible_mask_pixels'],r['view_index']))
    require(eligible[0]['candidate_id']==selected['selected_candidate_id'], 'Historical deterministic winner differs')
    history=dict(schema='p555.historical_view_audit_only.v1', geometry_audit=audits,policy=view['policy'],
                 historical_only=True, grants_new_query_pass=False)
    payload=dict(schema=SCHEMA,namespace='geometry_view_hints',role='pre_keypose_generation',
        context_key=context['context_key'],geometry_identity=context['key'],sources=closure,
        source_facts=context['facts_binding'],source_view_selection=view_binding,
        source_render=view['source_render'],original_camera=selected['selected_camera'],
        camera_parameters=camera,camera_parameters_sha256=digest(camera),
        crop_hint_xyxy=crop,original_selected_candidate_id=selected['selected_candidate_id'],
        original_selected_view_index=selected['selected_view_index'],
        historical_audit_sha256=digest(history),historical_candidate_count=len(audits),
        selector_policy_sha256=digest(view['policy']),must_revalidate=FRESH,
        grants_new_query_pass=False,automatic_runtime_consumption=True,acceleration_claimed=False,
        learned_experience=False,positive_credit=0,navmesh_or_route_objects_saved=False,
        pose_motion_or_qwen_answers_saved=False)
    for item in [*closure,view_binding,context['facts_binding']]: verified(item)
    return payload,history


def _safe_path(path):
    path=Path(os.path.abspath(path))
    for p in (path,*path.parents): require(not p.is_symlink(), 'Symlink store paths rejected')
    return path


def _manifest():
    return dict(schema=STORE_SCHEMA,namespace='geometry_view_hints',role='pre_keypose_generation',
        source_codes=sources(),successful_memory_store=False,positive_credit=0,
        navmesh_or_route_objects_saved=False,automatic_runtime_consumption=True)


def _check_store(root):
    m=read_sealed(root/'manifest.json')
    require({k:v for k,v in m.items() if k!='receipt_payload_sha256'}==_manifest(), 'Foreign/drifted view hint store')


def _read_entry(path, expected=None):
    path=_safe_path(path); saved=read_sealed(path/'receipt.json')
    require(saved.get('schema')==SCHEMA and re.fullmatch('[0-9a-f]{64}',saved.get('entry_id','')),
            'Invalid view hint entry')
    payload,history=_selection(verified(saved['source_facts']),verified(saved['source_view_selection']))
    require(saved['entry_id']==digest(payload)==path.name and path.parent.name==payload['context_key'], 'Entry identity/path drift')
    if expected is not None: require(payload==expected,'Idempotent entry content conflict')
    local_camera=verified(saved['camera_copy']);local_audit=verified(saved['historical_audit_copy'])
    require(local_camera==path/'camera.json' and local_audit==path/'historical_audit.json', 'Entry copy path escapes')
    require(read_json(local_camera)==payload['camera_parameters'], 'Copied camera differs from actual source')
    require(read_sealed(local_audit)==dict(history,receipt_payload_sha256=digest(history)), 'Copied historical audit differs')
    extra=dict(entry_id=digest(payload),camera_copy=artifact(local_camera),historical_audit_copy=artifact(local_audit))
    require(saved==dict(payload,**extra,receipt_payload_sha256=digest(dict(payload,**extra))), 'Entry content drift')
    return saved


@contextmanager
def _write_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    fd=os.open(root/'.writer.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        if not (root/'manifest.json').exists():
            require(not any(p.name not in {'.writer.lock'} for p in root.iterdir()), 'Refuse foreign nonempty store')
            write_once(root/'manifest.json',_manifest())
        _check_store(root)
        yield
    finally: os.close(fd)


def save_selection(facts_path, view_receipt_path, store_root):
    """Save one actual historical camera hint; replay is immutable/idempotent."""
    payload,history=_selection(facts_path,view_receipt_path)
    root=_safe_path(store_root); entry_id=digest(payload)
    destination=root/'entries'/payload['context_key']/entry_id
    with _write_lock(root):
        _safe_path(destination)
        if destination.exists():
            _read_entry(destination,payload)
            return destination/'receipt.json'
        staging=root/'.staging';staging.mkdir(exist_ok=True);_safe_path(staging)
        temporary=Path(tempfile.mkdtemp(prefix='entry_',dir=staging))
        write_once(temporary/'camera.json',payload['camera_parameters'],seal=False)
        write_once(temporary/'historical_audit.json',history)
        copies={name:dict(artifact(temporary/file),path=str(destination/file)) for name,file in
                (('camera_copy','camera.json'),('historical_audit_copy','historical_audit.json'))}
        write_once(temporary/'receipt.json',dict(payload,entry_id=entry_id,**copies))
        for row in [*payload['sources'],payload['source_facts'],payload['source_view_selection']]:verified(row)
        destination.parent.mkdir(parents=True,exist_ok=True);_safe_path(destination.parent)
        os.rename(temporary,destination)
        fsync_directory(destination.parent);fsync_directory(staging);fsync_directory(root)
    _read_entry(destination,payload)
    return destination/'receipt.json'


def query_views(facts_path, stage1_path, store_root):
    """Return proposal hints only. A missing bank never creates directories."""
    closure=sources(); context=_context(facts_path,stage1_path);root=_safe_path(store_root)
    candidates=[];exists=root.exists()
    if exists:
        _check_store(root); parent=_safe_path(root/'entries'/context['context_key'])
        if parent.exists():
            for path in sorted(parent.iterdir()):
                require(path.is_dir() and re.fullmatch('[0-9a-f]{64}',path.name), 'Unexpected entry path')
                row=_read_entry(path)
                require(row['geometry_identity']==context['key'], 'Cross-context entry')
                candidates.append(dict(entry=artifact(path/'receipt.json'),camera= row['camera_copy'],
                    camera_parameters=row['camera_parameters'],crop_hint_xyxy=row['crop_hint_xyxy'],
                    original_selected_candidate_id=row['original_selected_candidate_id'],
                    historical_view_selection=row['source_view_selection'],historical_only=True))
    require(sources()==closure, 'View index source changed while querying')
    verified(context['facts_binding']);verified(context['stage_binding'])
    final=_context(facts_path,stage1_path)
    require(final==context, 'Current facts/target/direction dependencies changed during lookup')
    return dict(schema=LOOKUP_SCHEMA,namespace='geometry_view_hints',role='pre_keypose_generation',
        status='hints_available_require_fresh_checks' if candidates else 'miss',
        reason='matching_geometry_hints' if candidates else ('no_matching_geometry' if exists else 'store_absent'),
        source_facts=context['facts_binding'],current_stage1_execution=context['stage_binding'],
        context_key=context['context_key'],geometry_identity=context['key'],store_root=str(root),
        candidates=candidates,candidate_count=len(candidates),sources=closure,must_revalidate=FRESH,
        grants_new_query_pass=False,automatic_runtime_consumption=True,acceleration_claimed=False,
        positive_credit=0,learned_experience=False,navmesh_or_route_objects_saved=False,
        query_does_not_create_store=True)



def context_identity(facts_path,stage1_path):
    """Full current owner/surface/direction identity; this does not grant a pass."""
    return _context(facts_path,stage1_path)['key']


def validate_lookup(value_or_path,facts_path,stage1_path,store_root):
    """Re-query immutable evidence/current binding immediately before consumption."""
    binding=None
    if isinstance(value_or_path,(str,Path)):
        binding=artifact(value_or_path)
        value=read_sealed(verified(binding))
        value={k:v for k,v in value.items() if k!='receipt_payload_sha256'}
    else:
        value=value_or_path
    require(isinstance(value,dict) and value.get('schema')==LOOKUP_SCHEMA, 'Invalid view lookup')
    expected=query_views(facts_path,stage1_path,store_root)
    require(value==expected, 'View lookup differs from current evidence or sources')
    if binding is not None: verified(binding)
    return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('save','query'):
        p=sub.add_parser(name);p.add_argument('--facts',type=Path,required=True)
        p.add_argument('--store',type=Path,required=True)
        p.add_argument('--view' if name=='save' else '--stage1',type=Path,required=True)
        if name=='query':p.add_argument('--output',type=Path,help='Optional NEW sealed lookup JSON')
    args=parser.parse_args()
    if args.command=='save':print(save_selection(args.facts,args.view,args.store))
    else:
        value=query_views(args.facts,args.stage1,args.store)
        if args.output:write_once(args.output,value)
        print(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
