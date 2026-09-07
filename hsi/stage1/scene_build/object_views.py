from ._common import *
from .geometry import geometry_kernel
from .context_review import crop_bounds,touches_border

def bbox_in_camera(bounds, camera):
    corners=np.asarray(list(itertools.product(*zip(bounds[0],bounds[1]))))
    xyz=np.c_[corners,np.ones(8)] @ camera.world_to_camera.T
    image=xyz[:,:3] @ camera.intrinsics.T
    if np.any(xyz[:,2] <= .05):
        return False
    uv=image[:,:2]/image[:,2:3]
    return bool(np.all(uv[:,0]>=8) and np.all(uv[:,0]<camera.width-8)
        and np.all(uv[:,1]>=8) and np.all(uv[:,1]<camera.height-8))


def run(geometry_path, instance_ids, output, gpu):
    started=time.monotonic()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),PYOPENGL_PLATFORM="egl",EGL_DEVICE_ID="0")
    geo=read_sealed(geometry_path)
    if geo["task_instruction_read"] is not False or geo["start_state_read"] is not False:
        raise ValueError("Only an instruction-independent scene geometry snapshot is allowed")
    snapshot_path=verified(geo["source_semantics"])
    snapshot=read_sealed(snapshot_path)
    objects={r["instance_id"]:r for r in snapshot["objects"]}
    if not 1<=len(instance_ids)<=3 or len(set(instance_ids))!=len(instance_ids):
        raise ValueError("Acquire views of one object or a bounded two/three-part proposal")
    rows=[objects[k] for k in instance_ids]
    kernel=geometry_kernel()
    config=kernel.SurfaceConfig()
    vertices,faces=kernel.load_obj_world_zup(verified(geo["source_mesh"]))
    with np.load(verified(geo["source_support"]),allow_pickle=False) as archive:
        member_ids=[np.asarray(archive[r["geometry_source"]["support_vertices_file_key"]],np.int64) for r in rows]
    support=np.unique(np.concatenate(member_ids))
    bounds=np.stack((vertices[support].min(0),vertices[support].max(0)))
    centre=(bounds[0]+bounds[1])*.5
    with np.load(verified(geo["navigation_fields"]),allow_pickle=False) as archive:
        free=archive["free"]
    from ..perception import renderer as loader
    anchors=[]
    for angle in np.linspace(-math.pi,math.pi,12,endpoint=False):
        for radius in (1.4,2.0,2.6):
            wanted=centre[:2]+radius*np.array([math.cos(angle),math.sin(angle)])
            found=kernel.nearest_true(free,wanted,.25,config)
            if found is None:
                continue
            cell,snap=found
            point=kernel.cell_center(cell,config)
            if any(np.linalg.norm(point-r["point"])<.5 for r in anchors):
                continue
            cameras=[]
            yaw=math.degrees(math.atan2(centre[1]-point[1],centre[0]-point[0]))
            for hi,height in enumerate((max(1.25,bounds[1,2]+.30),max(1.65,bounds[1,2]+.60))):
                cam=loader.make_camera(0,len(anchors),hi,yaw,point,eye_height_m=float(height),
                    look_height_m=float(centre[2]),look_distance_m=float(np.linalg.norm(centre[:2]-point)),
                    width=640,height=480,vertical_fov_degrees=70)
                if bbox_in_camera(bounds,cam):
                    cameras.append((hi,height))
            if cameras:
                anchors.append({"point":point,"yaw":yaw,"heights":cameras,"cell":cell,"snap":snap})
                break
    # Angular diversity first, not an instruction- or class-defined front side.
    if len(anchors)>6:
        anchors=[anchors[i] for i in np.linspace(0,len(anchors)-1,6,dtype=int)]
    cameras=[]
    for ai,anchor in enumerate(anchors):
        for hi,height in anchor["heights"]:
            cameras.append(loader.make_camera(len(cameras),ai,hi,anchor["yaw"],anchor["point"],
                eye_height_m=float(height),look_height_m=float(centre[2]),
                look_distance_m=float(np.linalg.norm(centre[:2]-anchor["point"])),
                width=640,height=480,vertical_fov_degrees=70))
    if len(cameras)<2:
        raise ValueError("No two full-object geometric cameras fit this scene")
    output.mkdir(parents=True,exist_ok=False)
    mesh=loader.load_scene_mesh(verified(geo["source_mesh"]))
    if not np.allclose(mesh.vertices,vertices,atol=1e-7) or not np.array_equal(mesh.faces,faces):
        raise ValueError("Rendering and SAM support mesh indexing differ")
    views,coverage=loader.render_views(mesh,cameras,output,scene_id=geo["scene_id"],
        depth_stride=8,surface_voxel_m=.05,maximum_vertex_samples=4000)
    projected=project_masks(mesh,member_ids,rows,views,loader,output)
    receipt={"schema":"p550.scene_only_object_context_cameras.v1","scene_id":geo["scene_id"],
        "source_geometry":artifact(geometry_path),"source_snapshot":artifact(snapshot_path),
        "member_ids":instance_ids,"source_member_objects":rows,"source_mesh":geo["source_mesh"],
        "source_support":geo["source_support"],"views":views,"coverage":coverage,
        "projected_views":projected,"actual_new_view_count":len(views),
        "new_sam_inference_performed":False,"mask_authority":"visible_original_mesh_faces_with_at_least_two_original_SAM_support_vertices",
        "task_instruction_read":False,"human_start_or_future_motion_read":False,"stage2_outputs_read":False,
        "category_or_function_used_for_camera_coordinates":False,"source":artifact(__file__),
        "elapsed_seconds":time.monotonic()-started}
    write_once(output / "receipt.json",receipt,seal=True)
    print({"scene":geo["scene_id"],"views":len(views),"eligible":sum(r["eligible"] for r in projected),
        "seconds":receipt["elapsed_seconds"]},flush=True)


def project_masks(mesh,member_ids,rows,views,loader,output):
    import pyrender
    import trimesh
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
    scene=pyrender.Scene()
    material=pyrender.MetallicRoughnessMaterial(doubleSided=True)
    nodes=[]
    for ids in member_ids:
        owned=np.zeros(len(vertices),bool)
        owned[ids]=True
        face_ids=np.flatnonzero(owned[faces].sum(1)>=2)
        if not len(face_ids):
            raise ValueError("No directly supported member faces")
        nodes.append(scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices,faces[face_ids],process=False),
            material=material,smooth=False)))
    renderer=pyrender.OffscreenRenderer(640,480)
    results=[]
    try:
        for view in views:
            camera=read(verified(view["camera"]))
            K=np.asarray(camera["K"])
            node=scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2]),
                pose=loader.opencv_to_pyrender_pose(np.asarray(camera["world_to_camera"])))
            observed=np.load(verified(view["depth"]),allow_pickle=False)
            masks,stats=[],[]
            for ni in range(len(nodes)):
                for index,item in enumerate(nodes):
                    item.mesh.is_visible=index==ni
                depth=renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
                expected=np.isfinite(depth)&(depth>0)
                visible=expected & (observed>0) & (np.abs(depth-observed)<=.03)
                masks.append(visible)
                stats.append({"member_id":rows[ni]["instance_id"],"expected_pixels":int(expected.sum()),
                    "visible_pixels":int(visible.sum()),"visible_fraction":float(visible.sum()/max(1,expected.sum())),
                    "projected_mask_touches_border":touches_border(expected)})
            scene.remove_node(node)
            union=np.any(masks,axis=0)
            eligible=all(r["visible_pixels"]>=250 and r["visible_fraction"]>=.80
                and not r["projected_mask_touches_border"] for r in stats)
            if not union.any():
                results.append({"view_index":view["view_index"],"members":stats,"eligible":False})
                continue
            raw=verified(view["rgb"])
            rgb=np.asarray(Image.open(raw).convert("RGB"))
            crop=crop_bounds(union)
            x0,y0,x1,y1=crop
            unmarked=output / f'context_view_{view["view_index"]:02d}_original_crop.png'
            marked=output / f'context_view_{view["view_index"]:02d}_mask_crop.png'
            Image.fromarray(rgb[y0:y1,x0:x1]).save(unmarked)
            pixels=rgb.copy()
            colors=[np.array(c) for c in ((0,230,230),(245,60,210),(255,205,0))]
            for mask,color in zip(masks,colors):
                pixels[mask]=(rgb[mask]*.84+color*.16).astype(np.uint8)
                pixels[mask & ~binary_erosion(mask,iterations=2)]=color
            canvas=Image.new("RGB",(x1-x0,y1-y0+28),"white")
            canvas.paste(Image.fromarray(pixels[y0:y1,x0:x1]),(0,28))
            ImageDraw.Draw(canvas).text((6,7),"Actual SAM-supported mesh projection",fill="black")
            canvas.save(marked)
            results.append({"view_index":view["view_index"],"members":stats,"eligible":eligible,
                "unmodified_rgb":artifact(raw),"original_texture_crop":artifact(unmarked),
                "contour_visualization":artifact(marked),"crop_xyxy":crop,"source_camera":view["camera"]})
    finally:
        renderer.delete()
    return results
