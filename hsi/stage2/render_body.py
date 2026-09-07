"""Two actual-body diagnostic views with explicitly hidden scene occluders."""
from ._common import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hsi.stage1.perception import renderer as loader

def run(bundle_path,output,gpu):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),PYOPENGL_PLATFORM="egl",EGL_DEVICE_ID=str(gpu))
    bundle=read_sealed(bundle_path)
    if bundle["stage3_handoff_allowed"] is not True:
        raise ValueError("Need an actual physics-passed candidate bundle")
    keypose=verified(bundle["outputs"]["published_keypose"])
    with np.load(keypose,allow_pickle=False) as data:
        vertices=np.array(data["vertices_world_zup"],copy=True)
        faces=np.array(data["faces"],copy=True)
        centre=np.array(data["surface_centre_world_xyz_zup_m"],copy=True)
        yaw=float(data["terminal_facing_yaw_rad"])
    if vertices.shape!=(10475,3) or faces.shape!=(20908,3):
        raise ValueError("Not a full actual SMPL-X body")
    import pyrender
    import trimesh
    from PIL import Image,ImageDraw
    output.mkdir(parents=True,exist_ok=False)
    scene=pyrender.Scene(bg_color=[1,1,1,1],ambient_light=[.5,.5,.5])
    body=trimesh.Trimesh(vertices,faces,process=False)
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.12,.65,.75,1],roughnessFactor=.85)
    scene.add(pyrender.Mesh.from_trimesh(body,material=material,smooth=False))
    x,y=centre[:2]
    floor=trimesh.Trimesh(vertices=[[x-1.1,y-1.1,0],[x+1.1,y-1.1,0],[x+1.1,y+1.1,0],[x-1.1,y+1.1,0]],
        faces=[[0,1,2],[0,2,3]],process=False)
    plane=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.84,.86,.88,1],doubleSided=True)
    scene.add(pyrender.Mesh.from_trimesh(floor,material=plane,smooth=False))
    renderer=pyrender.OffscreenRenderer(640,640)
    views=[]
    try:
        for index,offset in enumerate((-50,50)):
            azimuth=yaw+math.radians(offset)
            point=centre[:2]+2.7*np.array([math.cos(azimuth),math.sin(azimuth)])
            cam=loader.make_camera(index,index,0,math.degrees(azimuth+math.pi),point,
                eye_height_m=1.10,look_height_m=.67,look_distance_m=2.7,width=640,height=640,vertical_fov_degrees=44)
            K=cam.intrinsics
            pose=loader.opencv_to_pyrender_pose(cam.world_to_camera)
            camera_node=scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2]),pose=pose)
            light=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.8),pose=pose)
            rgb,_=renderer.render(scene)
            scene.remove_node(camera_node);scene.remove_node(light)
            image=Image.fromarray(rgb)
            draw=ImageDraw.Draw(image)
            draw.rectangle([0,0,640,46],fill="white")
            draw.text((10,6),"ACTUAL BODY / OCCLUDERS HIDDEN / DIAGNOSTIC ONLY",fill="black")
            draw.text((10,24),"Gray plane = world z=0 reference, not an edited scene",fill="black")
            path=output / ("body_view_"+str(index)+".png")
            image.save(path)
            views.append({"image":artifact(path),"position_world_zup_m":[*point.tolist(),1.1],
                "intrinsics":K.tolist(),"world_to_camera":cam.world_to_camera.tolist()})
    finally:
        renderer.delete()
    write_once(output / "receipt.json",{"schema":"p550.actual_body_occluders_hidden_diagnostic.v1",
        "source":artifact(__file__),"source_refine_bundle":artifact(bundle_path),"source_keypose":artifact(keypose),
        "views":views,"full_original_scene_render":False,"scene_occluders_hidden_explicitly":True,
        "gray_floor_is_coordinate_reference_at_z_m":0.0,
        "actual_body_world_vertices_or_pose_modified":False,"stage3_handoff_allowed":False},seal=True)
    print({"scene":bundle["scene_id"],"views":len(views),"diagnostic_only":True},flush=True)
