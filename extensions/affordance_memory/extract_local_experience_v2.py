"""Deterministic Stage2-only local extraction, never an admission authority.

Pure extraction may be tested on isolated fixtures. Only episode_evidence_v2
can allow a production result after all actual motion gates and semantics.
"""
from pathlib import Path
import numpy as np

from memory_common import artifact, digest, read_sealed, require, verified, write_once
from memory_retrieval import FRAME_CONVENTION
from memory_schema import make_record, validate_record
from motion_semantic_contract_v1 import canonical_episode

SCHEMA='p555.extracted_local_experience.v1'  # Existing store ABI, versioned kernel.


def derive_records(context, motion, tail, triangles, *, scopes=('invariant','scene_local')):
    """Use one actually observed terminal medoid; no full pose in the record."""
    require(context['frame_convention']==FRAME_CONVENTION and context['stage']=='stage2_contact','Wrong local frame/stage')
    invariant=context['invariant']
    require(invariant['action']=='sit' and invariant['surface_role']=='interaction_contact'
        and invariant['effector']=='pelvis_glute' and invariant['body_shape_bin']=='neutral','Wrong typed local role')
    rows=tail['frames']; indices=tail['actual_source_frame_indices'];count=len(motion['joints'])
    require(indices==list(range(count-4,count)) and [r['frame_index'] for r in rows]==indices,'Extraction requires actual final four frames')
    anchors=np.asarray([r['metrics']['measured_gluteal_anchor_world_xyz_m'] for r in rows],dtype=float)
    require(anchors.shape==(4,3) and np.isfinite(anchors).all(),'Invalid observed contact anchors')
    chosen=int(np.argmin(np.linalg.norm(anchors[:,None]-anchors[None,:],axis=-1).sum(axis=1)))
    frame=indices[chosen];contact=anchors[chosen];joints=np.asarray(motion['joints'][frame])
    require(joints.shape==(22,3) and np.isfinite(joints).all(),'Invalid observed joints')
    rotation=np.asarray(context['frame']['local_to_world_rotation'],dtype=float)
    origin=np.asarray(context['frame']['origin_world_zup_m'],dtype=float)
    require(rotation.shape==(3,3) and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8)
        and np.linalg.det(rotation)>0,'Invalid handed local frame')
    from fast_keypose import triangle_heights
    triangles=np.asarray(triangles,dtype=float);chosen_tri=None
    for triangle in triangles:
        height,inside=triangle_heights(contact[None,:2],triangle[None])
        if inside[0] and -1e-6<=contact[2]-height[0]<=.040001:
            if chosen_tri is None or height[0]>chosen_tri[0]: chosen_tri=(height[0],triangle)
    require(chosen_tri is not None,'Measured contact lacks current true triangle support')
    triangle=chosen_tri[1];normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
    require(np.linalg.norm(normal)>1e-10,'Degenerate support normal');normal/=np.linalg.norm(normal)
    if normal[2]<0:normal=-normal
    axis=joints[2]-joints[1];facing=np.array([-axis[1],axis[0],0.]);facing=rotation.T@facing
    require(np.linalg.norm(facing[:2])>1e-8,'No observed facing');facing=facing[:2]/np.linalg.norm(facing[:2])
    payload=dict(contact_point_local_xyz_m=(rotation.T@(contact-origin)).tolist(),
        surface_normal_local_xyz=(rotation.T@normal).tolist(),root_offset_local_xyz_m=(rotation.T@(joints[0]-contact)).tolist(),
        # Stage2 terminal contact phase is local to this typed endpoint, not a
        # fraction of the entire route's variable duration.
        facing_local_xy=facing.tolist(),contact_phase=1.0)
    require(isinstance(scopes,(tuple,list)) and scopes and len(set(scopes))==len(scopes)
        and set(scopes)<= {'invariant','scene_local'},'Unknown extraction scope')
    records=[make_record(stage='stage2_contact',scope=scope,
        source_scene_fingerprint_sha256=context['scene_fingerprint_sha256'],invariant=invariant,
        shape=context['shape'],payload=payload) for scope in scopes]
    return records,dict(frame_convention=FRAME_CONVENTION,observed_frame_index=frame,
        source_terminal_frame_indices=indices,selection='deterministic_observed_anchor_medoid',
        contact_phase_convention='typed_stage2_terminal_endpoint_not_global_motion_time',
        normal_from_actual_current_triangle=True,pose_or_motion_stored=False,world_payload_stored=False)


def check_receipt(binding, *, episode_id, record, source_episode, source_geometry, extraction):
    value=read_sealed(verified(binding));validate_record(record)
    require(value.get('schema')==SCHEMA and value.get('episode_id')==episode_id
        and value.get('record_id')==record['record_id'] and value.get('record')==record,'Extraction record/episode drift')
    require(value.get('source_episode_artifact')==source_episode and value.get('source_geometry_artifact')==source_geometry
        and value.get('extractor_source_artifact')==artifact(__file__) and value.get('extraction')==extraction,
        'Extraction source or measured local payload drift')
    for key in ('source_episode_artifact','source_geometry_artifact','extractor_source_artifact'):verified(value[key])
    return value


def write_receipts(output, *, episode_id, records, source_episode, source_geometry, extraction):
    """Writes proposed extraction evidence, not a positive store or success flag."""
    root=Path(output);root.mkdir(parents=True,exist_ok=False);result=[]
    for record in records:
        validate_record(record);path=root/(record['record_id']+'.json')
        write_once(path,dict(schema=SCHEMA,episode_id=episode_id,record_id=record['record_id'],record=record,
            source_episode_artifact=source_episode,source_geometry_artifact=source_geometry,
            extractor_source_artifact=artifact(__file__),extraction=extraction))
        result.append(artifact(path))
    return result
