"""Explicit ablations of constructed initial posture and static-pose timing.

Stage1/2 does not supply a moving initial history in these three cases. The
old adapter constructed a neutral T-pose. A relaxed-standing construction is
therefore an INPUT change, not an output repair or a same-initial ablation.
Never use it to overwrite an actual user-supplied initial pose/history.
"""
from dataclasses import replace
import math

import torch

from guarded_refine import AGENT
from hsi.common.artifacts import require
from hsi.stage3_sequence.data import axis_angle_to_matrix
from hsi.stage3_sequence.constraints import matrix_to_rotation_6d
from sdf_guidance import EventSchedule


def constructed_relaxed_stand(case, *, source_is_adapter_constructed=False):
    require(source_is_adapter_constructed is True,
            'Never replace a supplied initial pose/history with a constructed stand')
    require(case.provenance.get('initialization_kind')=='new_same_shape_neutral_standing_from_stage1_xy_yaw'
            and case.provenance.get('initial_pose_is_observed_motion') is False
            and case.provenance.get('future_gt_pose_motion_or_times_used') is False,
            'Only a source-bound adapter-constructed initial pose may change')
    pose=case.initial_pose.detach().clone()
    require(pose.shape==(135,) and torch.isfinite(pose).all(),'Invalid initial pose')
    identity=matrix_to_rotation_6d(torch.eye(3,device=pose.device,dtype=pose.dtype))
    require(torch.equal(pose[9:].reshape(21,6),identity.expand(21,6)),
            'Only the explicitly constructed all-identity local T-pose is eligible')
    for joint,sign in ((16,-1.),(17,1.)):
        aa=pose.new_tensor([0.,0.,sign*math.radians(75.)])
        pose[3+6*joint:3+6*(joint+1)]=matrix_to_rotation_6d(axis_angle_to_matrix(aa))
    with torch.no_grad(): mesh=case.body.mesh(pose[None,None])
    require(torch.equal(pose[:9],case.initial_pose[:9]),'Constructed root position/orientation changed')
    report=dict(kind='constructed_relaxed_stand_NOT_observed_history',
        changed_joints=[16,17],local_axis='SMPLX_native_Z',shoulder_degrees=[-75.,75.],
        root_xyz_global_orientation_betas_hands_unchanged=True,
        original_pose_source='adapter_constructed_identity_local_pose',
        initial_posture_is_different_from_old_T_pose=True,
        retrieved_reference_or_future_motion_used=False)
    return replace(case,initial_pose=pose,initial_joints=mesh.joints[0,0],
                   initial_vertices=mesh.vertices[0,0]),report


def target_condition_schedule(schedule, mode, *, window=20):
    """Delay static whole-body attraction, preserving original path/timetable.

    Only shape-target strength is changed. Root path and separate root-height
    conditions keep the original schedule. This does not learn event time.
    """
    require(mode in ('progressive','late_keypose'),'Unregistered target condition mode')
    require(type(window) is int and 1<=window<=40,'Bounded static-target window required')
    require(schedule.target_strength.ndim==1 and schedule.target_strength.is_floating_point()
            and torch.isfinite(schedule.target_strength).all(),'Finite floating target strength required')
    frames=len(schedule.target_strength)
    require(schedule.route_xy.shape==(frames,2) and schedule.route_xy.is_floating_point()
            and torch.isfinite(schedule.route_xy).all(),'Finite matching root path required')
    anchors=schedule.anchor_frames
    require(isinstance(anchors,list) and anchors and all(type(i) is int and 0<i<frames for i in anchors)
            and anchors==sorted(set(anchors)),'Ordered unique integer event anchors required')
    event=schedule.description.get('interaction_anchor_frame_0based')
    require(type(event) is int and window<=event<frames and anchors[0]==event,'Invalid event schedule')
    if mode=='progressive': return schedule
    phase=((torch.arange(frames,device=schedule.target_strength.device)-event+window).float()/window).clamp(0,1)
    weight=phase.square()*(3-2*phase)
    description={**schedule.description,'whole_body_target_mode':mode,
                 'whole_body_attraction_start_frame_0based':event-window,
                 'root_route_and_height_schedule_changed':False}
    return EventSchedule(schedule.route_xy.clone(),weight,list(schedule.anchor_frames),description)
