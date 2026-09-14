"""Explicit six-sofa-convex proxy: not full surface or PhysX VHACD collision.

The max outward plane value is exact signed penetration depth inside each
convex solid, and a conservative signed-clearance surrogate outside. Min over
six solids unions them; ground is analytical. J24/bone samples are not a skin.
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch

from run_closd_smoke import ROOT,VENDOR,binding,geometry_sources

PARENTS=(-1,0,1,2,3,0,5,6,7,0,9,10,11,12,11,14,15,16,17,11,19,20,21,22)


def build_hull_planes():
    import trimesh
    geometry=geometry_sources('bench');arrays={};rows=[]
    for i,row in enumerate(geometry['target_meshes']):
        mesh=trimesh.load(row['path'],force='mesh',process=False)
        hull=mesh.convex_hull
        normals=np.asarray(hull.face_normals,dtype=np.float32)
        offsets=-(normals*np.asarray(hull.triangles_center)).sum(-1).astype(np.float32)
        if not hull.is_watertight or not hull.is_volume:raise ValueError('Convex hull must define a closed positive solid')
        planes=np.c_[normals,offsets].astype(np.float32)
        if not np.allclose(np.linalg.norm(planes[:,:3],axis=-1),1,atol=1e-5):raise ValueError('Invalid normalized hull planes')
        arrays['planes_%d'%i]=planes
        rows.append(dict(source=row,original_watertight=bool(mesh.is_watertight),convex_faces=len(hull.faces),
                         original_vertices=len(mesh.vertices),original_faces=len(mesh.faces),convex_volume_m3=float(hull.volume)))
    return arrays,dict(parts=rows,kind='six_original_sofa_asset_convex_halfspace_proxy',not_sdk_vhacd=True,
                       not_full_mesh_sdf=True,body_proxy='24 joints plus 3 interior points per 23 bones',
                       sign='positive outside; negative inside; minimum unions each solid and Z=0 ground')


def rotation_xyzw(q):
    x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def body_points(joints):
    if joints.shape[-2:]!=(24,3):raise ValueError('Expected original world J24')
    start=joints[...,list(PARENTS[1:]),:];end=joints[...,1:,:]
    return torch.cat([joints]+[start+(end-start)*a for a in (.25,.5,.75)],dim=-2)


class ConvexSceneProxy:
    def __init__(self,planes,target_position,target_xyzw):
        self.planes=[torch.as_tensor(p,device=target_position.device,dtype=target_position.dtype) for p in planes]
        self.position=target_position.detach().clone();self.rotation=rotation_xyzw(target_xyzw.detach())
        if len(self.planes)!=6 or self.position.shape!=(3,) or self.rotation.shape!=(3,3):raise ValueError('One exact original sofa with six proxy solids required')
        if not bool(torch.isfinite(self.position).all()):raise ValueError('Invalid target position')
    def query(self,points):
        local=(points-self.position)@self.rotation
        fields=[(local@p[:,:3].T+p[:,3]).amax(-1) for p in self.planes]
        return torch.minimum(torch.stack(fields,-1).amin(-1),points[...,2])
    def loss(self,points):
        return torch.relu(-self.query(points)-.003).square().mean()


def decode_world_future(rep,prefix,suffix,recon_data):
    from closd.utils.rep_util import smpl_2_mujoco
    if prefix.shape!=(1,263,1,20) or suffix.shape!=(1,263,1,40):raise ValueError('Exact 20→40 DiP ABI required')
    full=torch.cat((prefix.detach(),suffix),-1).squeeze(2).permute(0,2,1)
    smpl=rep.hml_to_pose(full,recon_data,sim_at_hml_idx=19)
    # Same official final world conversion as _get_state_from_gen_cache.
    world=smpl[:,:,smpl_2_mujoco].matmul(rep.to_isaac_mat.T)
    if world.shape!=(1,90,24,3):raise ValueError('Original 20→30FPS decoder ABI changed')
    return world[:,30:]


class BoundedGuide:
    def __init__(self,scene,decoder):
        self.scene=scene;self.decoder=decoder;self.trace=[];self.calls=0
    def __call__(self,clean):
        original=clean.detach().clone();self.calls+=1
        with torch.enable_grad():
            initial_points=body_points(self.decoder(original));before=self.scene.loss(initial_points).detach()
            candidate=original.clone().requires_grad_(True)
            optimizer=torch.optim.Adam([candidate],lr=.01)
            gradients=[]
            for _ in range(2):
                collision=self.scene.loss(body_points(self.decoder(candidate)))
                objective=100.*collision+(candidate-original).square().mean()
                gradient=torch.autograd.grad(objective,candidate)[0]
                if not bool(torch.isfinite(gradient).all()):raise ValueError('Nonfinite original decoder gradient')
                gradients.append(float(gradient.abs().max().detach()))
                candidate.grad=gradient;optimizer.step();optimizer.zero_grad()
                with torch.no_grad():candidate.copy_(original+(candidate-original).clamp(-.08,.08))
            after=self.scene.loss(body_points(self.decoder(candidate))).detach()
            accepted=bool(torch.isfinite(after) and after<=before+1e-12)
            result=candidate.detach() if accepted else original
        self.trace.append(dict(call=self.calls,iterations=2,before_collision_m2=float(before),candidate_collision_m2=float(after),
            accepted=accepted,returned_collision_m2=float(after if accepted else before),gradient_max=gradients,
            native_reference_change_max=float((result-original).abs().max()),trust_radius_normalized_features=.08,
            prefix_modified=False,full_surface=False,not_sdk_collision_sdf=True))
        return result


def cpu_rep(mean,std):
    """CPU test fixture for original decoder; only fixed constructor CUDA placement differs."""
    from scipy.spatial.transform import Rotation
    from closd.utils.rep_util import RepresentationHandler
    obj=object.__new__(RepresentationHandler)
    obj.mean=torch.as_tensor(mean);obj.std=torch.as_tensor(std);obj.offset_height=.92;obj.offset=0.
    obj.to_isaac_mat=torch.as_tensor(Rotation.from_euler('xyz',[-np.pi/2,0,0]).as_matrix(),dtype=torch.float32)
    obj.smpl2sim_rot_mat=obj.to_isaac_mat@obj.to_isaac_mat
    obj.y180_rot=torch.as_tensor(Rotation.from_euler('xyz',[0,-np.pi,0]).as_matrix(),dtype=torch.float32)
    obj.smpl_end_effectors_idx=[22,23];obj.time_prints=False
    return obj
