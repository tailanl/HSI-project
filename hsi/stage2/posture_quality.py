"""Explicit neutral sitting checks for the current prompt's grounded forward feet."""
import math
import numpy as np


def metrics(joints,yaw):
    joints=np.asarray(joints,dtype=float)
    if joints.shape!=(22,3) or not np.isfinite(joints).all() or not math.isfinite(yaw):
        raise ValueError("Need actual finite SMPL-X joints and bound numeric yaw")
    forward=np.array([math.cos(yaw),math.sin(yaw)])
    hips, knees, ankles, toes=[joints[list(ids)] for ids in ((1,2),(4,5),(7,8),(10,11))]
    foot=toes[:,:2]-ankles[:,:2]
    cosine=foot@forward/np.maximum(np.linalg.norm(foot,axis=1),1e-8)
    ankle_forward=(ankles[:,:2]-hips[:,:2])@forward
    shin=np.linalg.norm(knees[:,:2]-ankles[:,:2],axis=1)
    gates={"ankles_near_floor":bool(np.all((ankles[:,2]>=.02)&(ankles[:,2]<=.14))),
        "ankles_forward_of_hips":bool(np.all(ankle_forward>=.05)),
        "feet_face_open_seat_side":bool(np.all(cosine>=.30)),
        "shins_not_strongly_folded_back":bool(np.all(shin<=.30))}
    return {"ankle_z_m":ankles[:,2].tolist(),"ankle_forward_from_hip_m":ankle_forward.tolist(),
        "toe_forward_cosines":cosine.tolist(),"shin_horizontal_span_m":shin.tolist(),
        "neutral_sit_quality_gates":gates,"all_neutral_sit_quality_gates_passed":all(gates.values()),
        "scope":"neutral sitting with both feet naturally grounded as required by the current H3 prompt",
        "not_a_universal_gate_for_other_pose_semantics":True}


def loss(state,yaw):
    import torch
    joints=state.joints[:22]
    forward=joints.new_tensor([math.cos(yaw),math.sin(yaw)])
    hips,knees,ankles,toes=[joints[list(ids)] for ids in ((1,2),(4,5),(7,8),(10,11))]
    foot=toes[:,:2]-ankles[:,:2]
    cosine=(foot*forward).sum(1)/torch.linalg.vector_norm(foot,dim=1).clamp_min(1e-6)
    advance=((ankles[:,:2]-hips[:,:2])*forward).sum(1)
    shin=torch.linalg.vector_norm(knees[:,:2]-ankles[:,:2],dim=1)
    return (1000*(ankles[:,2]-.085).square().mean()
        +300*torch.relu(.14-advance).square().mean()
        +2*torch.relu(.60-cosine).square().mean()
        +150*torch.relu(shin-.20).square().mean())
