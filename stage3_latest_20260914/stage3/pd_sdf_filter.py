"""Experimental SDF action filter, not an original UniHSI paper component.

The trained actor and actual PhysX transition are retained. A refreshed physical
Jacobian predicts SMALL articulation changes only; this is not a dynamics model
or a collision-free guarantee. Root/observed states are NEVER written here.
The velocity check disables filtering if the simulator Jacobian convention does
not agree with the actual body velocities. All events are logged.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from reference_sdf import SceneSDF


def body_surface_samples(xml, body_names):
    path=Path(__file__).with_name('comparison')/'physics_rollout_tools.py'
    spec=importlib.util.spec_from_file_location('_sdf_physical_xml',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    geoms=module.mjcf_geometries(xml,body_names)
    # Deterministic actual primitive surface samples; not a full mesh guarantee.
    directions=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)
    points=[];indices=[]
    for g in geoms:
        if g['kind']=='sphere':local=g['pos']+directions*g['size'][0]
        elif g['kind']=='box':
            signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
            local=g['pos']+signs*g['size']
        else:
            a,b=g['endpoints'];axis=(b-a)/np.linalg.norm(b-a)
            tangent=np.cross(axis,[1,0,0] if abs(axis[0])<.8 else [0,1,0]);tangent/=np.linalg.norm(tangent)
            side=np.cross(axis,tangent)
            ring=np.stack([np.cos(v)*tangent+np.sin(v)*side for v in np.linspace(0,2*np.pi,8,endpoint=False)])
            local=np.concatenate([a+ring*g['size'][0],b+ring*g['size'][0],
                                  ((a+b)/2)+ring*g['size'][0],
                                  (a-axis*g['size'][0])[None],(b+axis*g['size'][0])[None]])
        points.extend(local);indices.extend([g['body_index']]*len(local))
    return np.asarray(points,np.float32),np.asarray(indices,np.int64)


def rotate_xyzw(q,v):
    uv=torch.cross(q[:,:3],v,dim=-1)
    return v+2*(q[:,3:]*uv+torch.cross(q[:,:3],uv,dim=-1))


class PDSDFFilter:
    def __init__(self, task, packet, xml, body_names, iterations=2, trust_radius=.03):
        from isaacgym import gymtorch
        if task.num_envs!=1 or task.num_actions!=28:
            raise ValueError('Only validated single native AMP28 actor supported')
        if type(iterations) is not int or not 1<=iterations<=8 or not np.isfinite(trust_radius) or not 0<trust_radius<=.05:
            raise ValueError('Bounded action filter settings required')
        self.task=task
        self.scene=SceneSDF.from_packet(packet).to(task.device)
        local,ids=body_surface_samples(xml,body_names)
        self.local=torch.as_tensor(local,device=task.device)
        self.ids=torch.as_tensor(ids,device=task.device)
        raw=task.gym.acquire_jacobian_tensor(task.sim,'humanoid')
        if raw is None:raise RuntimeError('No real physical humanoid Jacobian')
        self.jacobian=gymtorch.wrap_tensor(raw)
        if self.jacobian.shape!=(1,15,6,34):raise ValueError('Wrong free-root Jacobian layout')
        self.iterations=iterations;self.trust_radius=trust_radius;self.trace=[]

    def __call__(self, action):
        task=self.task
        if action.shape!=(1,28) or not bool(torch.isfinite(action).all()) or float(action.abs().max())>1.000001:
            raise ValueError('Finite bounded native PD action required')
        task.gym.refresh_jacobian_tensors(task.sim)
        jac=self.jacobian.detach().clone()[0]
        generalized=torch.cat((task._humanoid_root_states[0,7:13],task._dof_vel[0]))
        measured=torch.cat((task._rigid_body_vel[0],task._rigid_body_ang_vel[0]),dim=-1)
        jac_velocity=torch.einsum('bck,k->bc',jac,generalized)
        residual=float((jac_velocity-measured).square().mean().sqrt())
        record=dict(jacobian_velocity_rmse=residual,surface_sample_count=len(self.ids),
                    actual_physical_transition=True,root_state_modified=False,
                    learned_policy_weights_modified=False,full_surface_guarantee=False)
        if not np.isfinite(residual) or residual>.05:
            self.trace.append(dict(record,applied=False,reason='jacobian_velocity_convention_check_failed'))
            return action
        local_world=rotate_xyzw(task._rigid_body_rot[0,self.ids],self.local)
        points=task._rigid_body_pos[0,self.ids]+local_world
        linear=jac[self.ids,:3,6:]
        angular=jac[self.ids,3:,6:]
        point_jac=linear+torch.cross(angular.transpose(1,2),local_world[:,None,:].expand(-1,28,-1),dim=-1).transpose(1,2)
        base=action.detach().clone()
        q=task._dof_pos.detach().clone()[0]
        # A bounded response proxy, deliberately NOT an assumed next PhysX state.
        def energy(candidate):
            target=task._pd_action_offset+task._pd_action_scale*candidate[0]
            dq=(.1*(target-q)).clamp(-.04,.04)
            predicted=points+torch.einsum('vck,k->vc',point_jac,dq)
            return self.scene.loss(predicted)
        with torch.enable_grad():
            x=base.clone().requires_grad_(True)
            with torch.no_grad():initial=float(energy(base))
            opt=torch.optim.Adam([x],lr=.01)
            for _ in range(self.iterations):
                loss=100*energy(x)+(x-base).square().mean()
                opt.zero_grad();loss.backward()
                if x.grad is None or not bool(torch.isfinite(x.grad).all()):
                    self.trace.append(dict(record,applied=False,reason='nonfinite_filter_gradient'))
                    return base
                opt.step()
                with torch.no_grad():x.copy_((base+(x-base).clamp(-self.trust_radius,self.trust_radius)).clamp(-1,1))
            with torch.no_grad():after=float(energy(x));change=float((x-base).abs().max())
        accepted=np.isfinite(after) and after<=initial+1e-12
        self.trace.append(dict(record,applied=bool(accepted and change>1e-8),
            reason='predicted_surface_penetration_nonincrease' if accepted else 'prediction_deteriorated_fallback',
            before_proxy_sdf_m2=initial,after_proxy_sdf_m2=after,max_normalized_action_delta=change if accepted else 0.))
        return x.detach() if accepted else base
