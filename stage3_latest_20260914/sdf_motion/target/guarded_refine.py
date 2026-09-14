"""Bounded generated-motion contact refinement with explicit non-regression.

The prior experiment showed that whole-mesh mean SDF can underweight feet, and
free root corrections can leave the planned approach. Here every generated
root XYZ/orientation stays EXACTLY fixed, the existing initial/event poses are
locked, and only intermediate leg rotations may change. Full-scene SDF remains
present; a separate actual anatomical-foot floor term prevents dilution by the
10k non-foot vertices. Nothing is changed during rendering.

This is kinematic refinement of THIS episode, not historical-motion retrieval,
training, physics simulation or an admitted UniHSI controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

AGENT=Path(__file__).resolve().parents[3]
for directory in (AGENT/'HSI-project',AGENT/'methods/affordance_memory_dip_20260912/code',
                  AGENT/'methods/closd_unihsi_sdf_stage3_20260912/code'):
    if str(directory) not in sys.path:sys.path.append(str(directory))
from hsi.common.artifacts import require
from hsi.stage2.refine_physics import build_anatomy_masks,RefineConfig
from current_contact_energy import SameBodySoles,CurrentFloorSupport
from contact_phase_guidance import ContactConfig,ContactPhaseGuidance,build_contact_phase

LOWER_JOINTS=(1,2,4,5,7,8,10,11)


@dataclass(frozen=True)
class Config:
    steps:int=100
    lr:float=.003
    rotation_component_radius:float=.08
    contact_weight:float=2.
    anatomical_floor_weight:float=1000.
    trust_joint_weight:float=50.
    velocity_match_weight:float=.02
    trace_every:int=10

    def validate(self):
        require(type(self.steps) is int and 1<=self.steps<=200,'Bounded refinement steps required')
        require(type(self.trace_every) is int and 1<=self.trace_every<=50,'Invalid trace interval')
        for key,value in asdict(self).items():
            require(type(value) in (int,float) and math.isfinite(value) and value>0,'Invalid '+key)
        require(self.rotation_component_radius<=.15 and self.lr<=.02 and self.contact_weight<=10,
                'Conservative bounded local rotations/step/weight required')
        return self


def editable_components(motion,anchors):
    require(motion.ndim==2 and motion.shape[1]==135,'Expected same-body [T,135] motion')
    mask=torch.zeros_like(motion,dtype=torch.bool)
    for joint in LOWER_JOINTS:
        mask[:,3+joint*6:3+(joint+1)*6]=True
    mask[0]=False
    require(all(type(i) is int and 0<=i<len(motion) for i in anchors),'Invalid current event indices')
    mask[anchors]=False
    return mask


def materialize(reference,delta,mask,radius):
    # Multiplication cannot make a locked NaN safe; check upstream finiteness.
    require(reference.shape==delta.shape==mask.shape and mask.dtype==torch.bool,
            'Reference, delta and bool permission mask must have identical shape')
    require(torch.isfinite(reference).all() and torch.isfinite(delta).all(),'Nonfinite pose correction')
    bounded=delta.clamp(-radius,radius)
    return torch.where(mask,reference+bounded,reference)


@torch.no_grad()
def measures(mesh,scene,foot_ids,pair_mask,fps):
    vertices=mesh.vertices[0]
    require(torch.isfinite(vertices).all() and torch.isfinite(mesh.joints).all(),'Nonfinite actual mesh')
    query=scene.query(vertices[None])
    d=query.distances[0]
    require(torch.isfinite(d).all() and query.outside.dtype==torch.bool
            and query.outside.shape==query.distances.shape,'Invalid/nonfinite scene query')
    inside=~query.outside[0]
    speed=[]
    contacts=[]
    lows=[]
    for foot_index,ids in enumerate(foot_ids):
        foot=vertices.index_select(1,ids)
        centre=foot.mean(1)
        v=(centre[1:,:2]-centre[:-1,:2]).norm(dim=-1)*fps
        chosen=pair_mask[:,foot_index]
        speed.append(v[chosen])
        contacts.append((foot[...,2].abs()<=.03).sum(-1)>=6)
        lows.append(foot[...,2].min())
    values=torch.cat(speed)
    contact=torch.stack(contacts,-1)
    still_contact=contact[:-1]&contact[1:]
    lost=int((pair_mask&~still_contact).sum())
    steps=(mesh.joints[0,1:]-mesh.joints[0,:-1]).norm(dim=-1)
    return dict(fixed_reference_contact_speed_mean_m_s=float(values.mean()) if len(values) else None,
        fixed_reference_contact_pair_count=len(values),
        lost_reference_contact_pair_count=lost,
        foot_floor_contact_fraction=float(contact.float().mean()),
        foot_floor_minimum_m=float(torch.stack(lows).min()),
        known_negative_sdf_vertex_fraction=float(((d<-.003)&inside).float().mean()),
        outside_unknown_fraction=float((~inside).float().mean()),
        deepest_union_sdf_m=float(d.min()),max_joint_step_m=float(steps.max()),
        pair_mask_held_fixed_to_prevent_raising_feet_to_game_speed=True,
        fullmesh_frame_vertex_count=int(d.numel()))


def acceptable(candidate,baseline):
    """No threshold relaxation or weighted score hiding a worse collision."""
    a,b=candidate,baseline
    reasons=[]
    for row in (a,b):
        for key,value in row.items():
            if type(value) in (int,float) and not math.isfinite(value):
                return False,['nonfinite_'+key]
    counts=[row.get('fixed_reference_contact_pair_count') for row in (a,b)]
    if any(type(count) is not int or count<=0 for count in counts) or counts[0]!=counts[1]:
        return False,['invalid_or_changed_reference_contact_pair_count']
    if a.get('lost_reference_contact_pair_count',1)!=0:
        reasons.append('lost_original_reference_contact_pairs')
    if a['fixed_reference_contact_speed_mean_m_s'] is None or b['fixed_reference_contact_speed_mean_m_s'] is None:
        reasons.append('no_reference_contact_pairs')
    elif a['fixed_reference_contact_speed_mean_m_s']>=b['fixed_reference_contact_speed_mean_m_s']-1e-5:
        reasons.append('contact_speed_not_improved')
    # Explicit tiny tolerances: ~one vertex-frame for known-SDF fraction,
    # below one vertex-frame for unknown, and 2mm kinematic step margin.
    limits=(('known_negative_sdf_vertex_fraction',1e-6),('outside_unknown_fraction',1e-7),
            ('max_joint_step_m',.002))
    for name,margin in limits:
        if a[name]>b[name]+margin:reasons.append(name+'_worsened')
    if a['deepest_union_sdf_m']<b['deepest_union_sdf_m']-1e-4:reasons.append('deepest_sdf_worsened')
    if a['foot_floor_minimum_m']<b['foot_floor_minimum_m']-1e-4:reasons.append('floor_depth_worsened')
    if a['foot_floor_contact_fraction']<b['foot_floor_contact_fraction']-.02:reasons.append('lost_floor_contact')
    return not reasons,reasons


def refine(reference,case,scene,schedule,config=None,*,fps=20.):
    config=(config or Config()).validate()
    require(fps==20. and reference.shape==(120,135),'Initial controlled experiment is 120 frames at20Hz')
    require(torch.isfinite(reference).all(),'Nonfinite original motion')
    original=reference.detach().clone()
    mask=editable_components(original,list(schedule.anchor_frames))
    soles=SameBodySoles(case)
    anatomy=build_anatomy_masks(case.body.model,10475,RefineConfig(steps=1,bilateral_foot_support=True).validate())
    ids=[torch.nonzero(m.to(reference.device),as_tuple=False).flatten()
         for m in (anatomy.left_foot_support,anatomy.right_foot_support)]
    started=time.monotonic()
    with torch.no_grad():
        base_mesh=case.body.mesh(original[None])
        protected=~mask.any(-1)
        locomotion=torch.arange(len(original),device=original.device)<schedule.description['walk_frames']
        phase=build_contact_phase(soles.from_vertices(base_mesh.vertices),CurrentFloorSupport(scene),fps=fps,
            locomotion_mask=locomotion[None],protected_frames=protected[None],
            config=ContactConfig(use_root_relative_intent=True),root_points=base_mesh.joints[...,0,:])
        objective=ContactPhaseGuidance(phase)
        contact=torch.stack([((base_mesh.vertices[0].index_select(1,x)[...,2].abs()<=.03).sum(-1)>=6)
                             for x in ids],-1)
        pair_mask=contact[:-1]&contact[1:]
        baseline=measures(base_mesh,scene,ids,pair_mask,fps)
    delta=torch.zeros_like(original,requires_grad=True)
    optimizer=torch.optim.Adam([delta],lr=config.lr)
    best=original.clone();best_metrics=baseline;best_iteration=None;trace=[]
    base_joints=base_mesh.joints.detach()
    for iteration in range(config.steps):
        motion=materialize(original,delta,mask,config.rotation_component_radius)
        mesh=case.body.mesh(motion[None])
        correction=mesh.joints-base_joints
        trust=correction.square().mean()
        velocity=((correction[:,1:]-correction[:,:-1])*fps).square().mean()
        contact_loss=objective(soles.from_vertices(mesh.vertices))['loss']
        feet=torch.cat([mesh.vertices.index_select(-2,x) for x in ids],-2)
        floor=torch.relu(scene.floor_z-feet[...,2]-.003).square().mean()
        collision=scene.mesh_energy(mesh,motion[None])['loss']
        loss=(collision+config.anatomical_floor_weight*floor+config.trust_joint_weight*trust+
              config.velocity_match_weight*velocity+config.contact_weight*contact_loss)
        require(torch.isfinite(loss),'Nonfinite fullmesh refinement loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        require(delta.grad is not None and torch.isfinite(delta.grad).all(),'Missing/nonfinite refinement gradient')
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-config.rotation_component_radius,config.rotation_component_radius)
            delta.masked_fill_(~mask,0.)
        if iteration%config.trace_every==0 or iteration==config.steps-1:
            with torch.no_grad():
                candidate=materialize(original,delta,mask,config.rotation_component_radius)
                current_mesh=case.body.mesh(candidate[None])
                numbers=measures(current_mesh,scene,ids,pair_mask,fps)
                allowed,reasons=acceptable(numbers,baseline)
                selected=allowed and (best_iteration is None or
                    numbers['fixed_reference_contact_speed_mean_m_s']<best_metrics['fixed_reference_contact_speed_mean_m_s'])
                if selected:
                    best=candidate.detach().clone();best_metrics=numbers;best_iteration=iteration
                trace.append(dict(iteration=iteration,loss=float(loss),contact_loss=float(contact_loss),
                    floor_m2=float(floor),trust_m2=float(trust),measures=numbers,
                    nonregression_passed=allowed,rejection_reasons=reasons,selected=selected))
    with torch.no_grad():
        output=case.body.mesh(best[None])
    require(torch.equal(best[~mask],original[~mask]),'Locked root/yaw/initial/event or upper body changed')
    return best,output,dict(schema='agent9.closd_unihsi_sdf.guarded_contact_refine.v1',
        config=asdict(config),baseline=baseline,result=best_metrics,selected_iteration=best_iteration,
        refinement_accepted=best_iteration is not None,trace=trace,elapsed_seconds=time.monotonic()-started,
        all_root_xyz_and_orientation_exactly_preserved=True,initial_and_event_poses_exactly_preserved=True,
        only_intermediate_leg_rotations_editable=True,body_shape_and_hands_unchanged=True,
        known_and_unknown_sdf_checked_all_vertices_all_frames=True,
        nonregression_tolerances=dict(known_sdf_fraction=1e-6,outside_fraction=1e-7,
            deepest_sdf_m=1e-4,floor_minimum_m=1e-4,max_joint_step_m=.002,
            lost_original_contact_pairs=0),
        unihsi_actor_or_discriminator_used=False,physics_simulation=False,
        learned_memory_used=False,motion_quality_accepted=False)
