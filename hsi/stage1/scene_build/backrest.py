from ._common import *
from .geometry import geometry_kernel,local_front
from ._common import exclusive_qwen

CHECKS=("pink_part_is_backrest","green_part_is_seat","parts_belong_to_same_target","evidence_sufficient")


def decision(judgments):
    if len(judgments)!=3 or len({r["source_view_index"] for r in judgments})!=3:
        raise ValueError("Need three different actual backrest views")
    accepted=True
    calls=[]
    for item in judgments:
        response,value=item["generation"],item["parsed"]
        call=read(response["call_receipt"])
        if call["status"]!="complete" or call["result"]!=response or json.loads(response["raw_completion"])!=value:
            raise ValueError("Backrest semantics do not match actual Qwen response")
        calls.append(response["call_receipt"])
        if set(value)!=set(CHECKS)|{"confidence","reason"} or any(type(value[k]) is not bool for k in CHECKS):
            raise ValueError("Invalid backrest semantic fields")
        score=value["confidence"]
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError("Invalid backrest confidence")
        accepted=accepted and all(value[k] for k in CHECKS) and score>=.85
    if len(set(calls))!=3:
        raise ValueError("Duplicate backrest call")
    return {"direction_review_accepted":bool(accepted),"selected_direction_ids":["A"] if accepted else [],
        "object_class_changed":False,"coordinate_or_angle_prediction_requested":False}


def run(geometry_path,key,output, *, qwen_client):
    geo=read_sealed(geometry_path)
    if geo["task_instruction_read"] is not False or geo["start_state_read"] is not False:
        raise ValueError("Backrest evidence is scene only")
    row=next(r for r in geo["objects"] if r["instance_id"]==key)
    old=read_sealed(verified(row["direction_review"]))
    kernel=geometry_kernel()
    vertices,_=kernel.load_obj_world_zup(verified(geo["source_mesh"]))
    with np.load(verified(row["surface_arrays"]),allow_pickle=False) as data:
        support=data["support_ids"].copy();seat=data["vertex_ids"].copy()
    z=float(row["surface"]["centre_world_xyz_zup_m"][2])
    front,evidence=local_front(kernel,vertices[support],vertices[seat],z)
    if evidence["method"]!="opposite_high_target_support_backrest_centroid" or evidence["high_support_vertex_count"]<100 \
            or evidence["backrest_offset_m"]<.10:
        raise ValueError("No strong geometric part candidate; do not infer backrest from category")
    bounds=np.stack((vertices[seat].min(0),vertices[seat].max(0)))
    local=np.all((vertices[support,:2]>=bounds[0,:2]-.30)&(vertices[support,:2]<=bounds[1,:2]+.30),axis=1)
    high=support[local & (vertices[support,2]>=z+.16)]
    snapshot=read_sealed(verified(geo["source_semantics"]))
    if old.get("source_object_views"):
        cameras=read_sealed(verified(old["source_object_views"]))["views"]
    else:
        cameras=read_sealed(verified(snapshot["sources"]["render"]))["views"]
    by_index={r["view_index"]:r for r in cameras}
    output.mkdir(parents=True,exist_ok=False)
    rendered=[]
    for record in old["evidence"]:
        view=by_index[record["view_index"]]
        camera=read(verified(view["camera"]))
        rgb=np.asarray(Image.open(verified(view["rgb"])).convert("RGB"))
        depth=np.load(verified(view["depth"]),allow_pickle=False)
        masks=[]
        for ids in (high,seat):
            xyz=np.c_[vertices[ids],np.ones(len(ids))] @ np.asarray(camera["world_to_camera"]).T
            uvz=xyz[:,:3] @ np.asarray(camera["K"]).T
            uv=np.rint(uvz[:,:2]/np.maximum(uvz[:,2:3],1e-8)).astype(int)
            visible=(xyz[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])&(uv[:,1]>=0)&(uv[:,1]<rgb.shape[0])
            ix=np.flatnonzero(visible);xy=uv[ix]
            visible[ix]&=(depth[xy[:,1],xy[:,0]]>0)&(np.abs(depth[xy[:,1],xy[:,0]]-xyz[ix,2])<=.03)
            xy=uv[visible]
            mask=np.zeros(rgb.shape[:2],bool);mask[xy[:,1],xy[:,0]]=True
            masks.append(binary_dilation(mask,iterations=1))
        if any(m.sum()<120 for m in masks):
            continue
        target_points=vertices[support]
        xyz=np.c_[target_points,np.ones(len(target_points))] @ np.asarray(camera["world_to_camera"]).T
        uvz=xyz[:,:3] @ np.asarray(camera["K"]).T
        uv=uvz[:,:2]/np.maximum(uvz[:,2:3],1e-8)
        lo=np.maximum([0,0],np.floor(uv.min(0)-30)).astype(int)
        hi=np.minimum([rgb.shape[1],rgb.shape[0]],np.ceil(uv.max(0)+31)).astype(int)
        x0,y0=lo;x1,y1=hi
        raw=output / f'view_{view["view_index"]:02d}_original.png'
        marked=output / f'view_{view["view_index"]:02d}_parts.png'
        Image.fromarray(rgb[y0:y1,x0:x1]).save(raw)
        painted=rgb.copy()
        for mask,color in zip(masks,([245,30,200],[0,230,95])):
            painted[mask]=(rgb[mask]*.55+np.asarray(color)*.45).astype(np.uint8)
        Image.fromarray(painted[y0:y1,x0:x1]).save(marked)
        rendered.append({"view_index":view["view_index"],"source_rgb":view["rgb"],"source_camera":view["camera"],
            "original_crop":artifact(raw),"part_visualization":artifact(marked),"part_pixel_counts":[int(m.sum()) for m in masks]})
    if len(rendered)!=3:
        raise ValueError("Need three visible, independently rendered backrest/seat part pairs")
    props={**{k:{"type":"boolean"} for k in CHECKS},"confidence":{"type":"number","minimum":0,"maximum":1},
        "reason":{"type":"string","maxLength":100}}
    schema={"type":"object","properties":props,"required":list(props),"additionalProperties":False}
    prompt=("The two images show the same exact target furniture: original texture then two projected mesh parts. "
        "Identify parts only. Is the PINK highlighted part the actual backrest? Is the GREEN part the actual seat? "
        "Do both highlighted parts belong to that same target chair/sofa rather than a nearby table or different furniture? "
        "Use visible shape, not the highlight color alone. If evidence is insufficient, say so. "
        "Do not choose any direction, arrow, front/back side, coordinate or angle. No human task or prior judgement is given. "
        "Return the four independent checks and confidence, plus one short reason of at most 12 words.")
    judgments=[]
    qwen=qwen_client.session(output / "qwen_calls",max_tokens=165)
    for index,record in enumerate(rendered):
        with exclusive_qwen():
            response=qwen.call([verified(record["original_crop"]),verified(record["part_visualization"])],prompt,
                schema=schema,call_id="p550_backrest_part_not_arrow_interpretation")
        judgments.append({"view_id":"V"+str(index),"source_view_index":record["view_index"],
            "generation":response,"parsed":json.loads(response["raw_completion"])})
    value={"schema":"p550.scene_only_semantic_backrest_direction.v1","source":artifact(__file__),"scene_id":geo["scene_id"],
        "instance_id":key,"source_geometry":artifact(geometry_path),"source_snapshot":geo["source_semantics"],"source_shape":row,
        "source_support":geo["source_support"],"source_surface_arrays":row["surface_arrays"],
        "high_part_vertex_count":len(high),"numeric_front_evidence":evidence,"evidence":rendered,"judgments":judgments,
        "decision":decision(judgments),"candidate_vectors_computed_by_geometry":{"A":front.tolist(),"B":(-front).tolist()},
        "human_query_read":False,"stage2_generated_person_read":False,"qwen_received_numeric_geometry":False,
        "qwen_direction_or_arrow_prediction_requested":False,"old_direction_review_not_overwritten":row["direction_review"],
        "direction_authority":"geometry opposite actual Qwen-confirmed backrest support centroid"}
    write_once(output / "receipt.json",value,seal=True)
    print({"scene":geo["scene_id"],"part_review":value["decision"]},flush=True)
    return value
