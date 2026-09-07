"""Native live typed-local retrieval with the unchanged v2 query policy.

Facts, body, direction, true support IDs and actual current SDF are independently
bound. Queries grant no credit. Every consumer must revalidate the frozen
snapshot plus live quarantine immediately before applying a local hypothesis.
"""
from __future__ import annotations
from pathlib import Path
import math
import time
import numpy as np
from hsi.common.artifacts import artifact, digest, read_json, read_sealed, require, verified, write_once
from hsi.stage1.binding import bind as stage1_bind
from .matching import FRAME_CONVENTION, descriptor_to_shape, load_current_surface, retrieve_current
from .schema import shape_family
from .facts import load_descriptors
from .store import MemoryStore, require_authority_sources
from .authority import _family
from .identity import canonical_episode

BODY_SHA256 = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
PREPARATION = 'hsi.memory.runtime_preparation.v1'
LOOKUP = 'hsi.memory.runtime_lookup.v1'


def sources():
    require_authority_sources()
    from hsi.common import artifacts
    from hsi.stage1 import binding
    from hsi.stage2 import sdf
    rows = [artifact(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    rows += [artifact(artifacts.__file__), artifact(binding.__file__), *sdf.source_closure().values()]
    return [record for _,record in sorted({row['path']:row for row in rows}.items())]


def _production_source(path):
    path = Path(path).resolve(strict=True)
    require(not any('fixture' in part.lower() or part.lower() == 'tests' for part in path.parts),
            'Fixture/test namespace is not production evidence')
    value = read_sealed(path)
    require(value.get('purpose', 'production') == 'production'
        and value.get('namespace', 'production') != 'test_fixture', 'Fixture evidence rejected')
    return artifact(path), value


def _current(stage1_path, facts_path, body_model, sdf_path, stage, body_shape_bin):
    require(stage == "stage2_contact", "Only Stage2 typed sit has a current verified role")
    require(body_shape_bin == "neutral", "Only the current zero-beta neutral body is registered")
    verified(body_model)
    require(body_model["sha256"] == BODY_SHA256,
            "Current body model identity mismatch")
    stage_binding, stage1 = _production_source(stage1_path)
    facts_binding, facts = _production_source(facts_path)
    sdf_binding, sdf = _production_source(sdf_path)
    require(stage1.get("schema") == "p550.stage1_query_execution.v1", "Unsupported actual Stage1 execution")
    bound = stage1_bind(stage1)
    target = read_sealed(verified(stage1["target"]))
    require(target["action_family"] == "sit", "Only actual sit actions are registered")
    require(stage1["instruction"] == target["instruction"] == bound["instruction"], "Actual instruction drift")
    require(stage1["source_fixed_geometry"] == facts["sources"]["fixed_geometry"], "Stale facts/fixed geometry")
    require(stage1["scene_id"] == facts["scene_id"] == bound["scene_id"], "Current scene mismatch")
    guard = read_sealed(verified(stage1["p552_contact_region_guard"]))
    require(guard["all_gates_passed"] is True and guard["quality_gates"]
            and all(x is True for x in guard["quality_gates"].values()), "Current route region gates failed")
    require(guard["source_target"] == stage1["target"]
            and guard["target_instance_id"] == bound["target_instance_id"]
            and guard["direction_id"] == bound["direction_hypothesis_id"]
            and guard["selected_approach_option_id"] == bound["selected_approach_option_id"]
            and np.allclose(guard["snapped_goal_world_xy_m"], bound["snapped_goal_world_xy_m"], atol=1e-8, rtol=0),
            "Current route/owner/direction region binding mismatch")
    verified(guard["source_region"])
    verified(guard["source_actual_navmesh_route"])
    triples = load_descriptors(facts_path)
    selected = [t for t in triples if t[0]["instance_id"] == bound["target_instance_id"]
                and t[1]["source_binding"]["direction"]["candidate_id"] == bound["direction_hypothesis_id"]]
    require(len(selected) == 1, "Current selected owner/direction has no unique typed descriptor")
    row, descriptor, _ = selected[0]
    surface = next(x for x in target["candidate_surfaces"] if x["candidate_id"] == bound["selected_surface_id"])
    owner = surface["furniture_instance_binding"]
    require(owner["verified"] is True and owner["surface_inside_target_furniture_verified"] is True
            and owner["graph_target_instance_id"] == row["instance_id"]
            and owner["resolved_target_class"] == row["category"], "Current semantic furniture owner mismatch")
    require(surface["surface"] == row["surface"], "Selected surface geometry differs from current facts")
    # Stable facts and selected target use different hash namespaces. Compare
    # the actual shared mesh face/vertex/support IDs, not those two SHA strings.
    with np.load(verified(surface["surface_face_archive"]), allow_pickle=False) as a, \
            np.load(verified(row["surface_arrays"]), allow_pickle=False) as b:
        for left, right in (("mesh_face_indices", "face_ids"), ("surface_vertex_ids", "vertex_ids"),
                            ("target_support_vertex_ids", "support_ids")):
            require(np.array_equal(a[left], b[right]), "Current surface mesh ownership drift: " + left)
        require(str(a["target_instance_id"].item()) == row["instance_id"], "Surface archive owner mismatch")
    chosen = descriptor["source_binding"]["direction"]
    expected_direction = [math.cos(bound["contact_forward_yaw_rad"]), math.sin(bound["contact_forward_yaw_rad"])]
    require(np.allclose(chosen["outward_world_xy"], expected_direction, atol=1e-8, rtol=0), "Direction evidence drift")
    shape = descriptor_to_shape(selected[0])
    invariant = dict(action="sit", target_semantic=row["category"], surface_role="interaction_contact",
                     effector="pelvis_glute", motion_phase="terminal_contact", body_model_sha256=body_model["sha256"],
                     body_shape_bin=body_shape_bin, shape_family=shape_family(shape))
    context = load_current_surface(facts_path, row["instance_id"], bound["direction_hypothesis_id"],
                                   stage=stage, invariant=invariant, body_model=body_model)
    from hsi.stage2.sdf import BoundSDFCache, SCHEMA as SDF_SCHEMA, SETTINGS
    cache = BoundSDFCache(sdf_path, stage1_path)
    require(sdf["schema"] == SDF_SCHEMA
            and sdf["source_kind"] == "current_new_scene", "Unsupported current SDF cache")
    require(sdf["source_binding"]["stage1_execution"] == stage_binding
            and sdf["source_binding"]["target"] == stage1["target"], "SDF actual Stage1/target drift")
    require(sdf["inputs"]["occupancy"] == facts["sources"]["source_occupancy"]
            and sdf["mask_receipt"]["target_component_sha256"] == owner["target_occupancy_mask_world_xyz_sha256"],
            "Current SDF scene/owner mismatch")
    expected_mask = {k: target["artifacts"]["target_occupancy_mask"][k] for k in ("path", "bytes", "sha256")}
    require(sdf["inputs"]["target_mask"] == expected_mask, "Current SDF target mask drift")
    require(sdf["mask_receipt"]["positive_is_free"] is True
            and sdf["mask_receipt"]["query_local_component_guess_used"] is False
            and sdf["pose_image_or_future_motion_used"] is False, "SDF provenance/semantics mismatch")
    for rec in [*sdf["inputs"].values(), *sdf["arrays"].values(), *sdf["source_closure"].values()]:
        verified(rec)
    # Native cache re-reads actual masks and all original resolution/owner gates.
    import torch
    cache(verified(sdf["inputs"]["occupancy"]), verified(sdf["inputs"]["target_mask"]), torch.device("cpu"),
        expected_occupancy_file_sha256=sdf["inputs"]["occupancy"]["sha256"],
        expected_target_mask_file_sha256=sdf["inputs"]["target_mask"]["sha256"],
        expected_target_world_sha256=target["target_occupancy_mask_world_xyz_sha256"], **SETTINGS)
    bundle = read_sealed(verified(stage1["bundle"]))
    require(guard["source_compile"] == bundle["artifacts"]["roadmap_compile"]
            and guard["source_actual_navmesh_route"] == bundle["artifacts"]["navmesh_route"],
            "Region guard differs from actual compiled/NavMesh route")
    request_binding = bundle["artifacts"]["roadmap_request"]
    request = read_json(verified(request_binding))  # P504 input is intentionally unsealed, artifact-bound.
    start = request["start_world_xy_m"]
    require(isinstance(start, list) and len(start) == 2
            and all(type(x) in (float, int) and math.isfinite(x) for x in start), "Invalid actual query start")
    actions = []
    for step in stage1["interaction_plan"]["steps"]:
        require(step["action"] == "sit" and isinstance(step["target_ids"], list) and step["target_ids"], "Unsupported ordered action plan")
        actions.append({k: step[k] for k in ("action", "target_ids", "reference_ids")})
    require(len(actions) == 1 and actions[0]["target_ids"] == [row["instance_id"]], "Current Stage2 is one selected sit interaction only")
    identity = dict(schema="p555.runtime_full_query_identity.v1", scene_id=context["scene_id"],
        scene_fingerprint_sha256=context["scene_fingerprint_sha256"], fixed_geometry_sha256=context["fixed_geometry"]["sha256"],
        instruction=stage1["instruction"], start_world_xy_m=[float(x) for x in start], ordered_actions=actions,
        body_model_sha256=body_model["sha256"], body_shape_bin=body_shape_bin, stage=stage)
    return dict(current_stage1_execution=stage_binding, facts=facts_binding, body_model=body_model,
                sdf_receipt=sdf_binding, sdf=sdf, context=context, identity=identity,
                route_binding=bound, route_guard=stage1["p552_contact_region_guard"], roadmap_request=request_binding)

class CurrentSDFPointFilter:
    """Conservative minimum of eight neighbouring current SDF voxel centres.

    No interpolation can conceal a negative corner. Outside the centre domain
    is rejected. This is ONLY a two-point filter and does not validate a body.
    """
    def __init__(self, current):
        self.current, self.calls = current, 0

    def __call__(self, projection):
        current, sdf = self.current, self.current["sdf"]
        require(projection["fixed_geometry"] == current["context"]["fixed_geometry"]
                and projection["owner"] == current["context"]["owner"], "SDF projection owner drift")
        points = np.asarray([projection["contact_world_xyz_m"], projection["root_world_xyz_m"]], dtype=float)
        require(points.shape == (2, 3) and np.isfinite(points).all(), "Invalid SDF points")
        grid = np.load(verified(sdf["arrays"]["collision_sdf"]), mmap_mode="r", allow_pickle=False)
        meta = sdf["mask_receipt"]
        require(list(grid.shape) == meta["grid_shape_world_xyz"] and grid.dtype == np.float32, "SDF grid identity/shape mismatch")
        lower, upper = np.asarray(meta["grid_bounds_world_zup_m"], dtype=float)
        spacing = np.asarray(meta["spacing_m"], dtype=float)
        require(np.allclose((upper - lower) / np.asarray(grid.shape), spacing, atol=1e-12, rtol=0), "SDF metric spacing mismatch")
        coords = (points - lower) / spacing - .5
        base = np.floor(coords).astype(int)
        outside = np.any((base < 0) | (base + 1 >= np.asarray(grid.shape)), axis=1)
        values = []
        for index, out in zip(base, outside):
            values.append(-1. if out else float(min(grid[tuple(index + [x, y, z])]
                for x in (0, 1) for y in (0, 1) for z in (0, 1))))
        require(all(math.isfinite(x) for x in values), "Nonfinite current SDF sample")
        self.calls += 1
        return dict(accepted=not outside.any() and all(x >= 0 for x in values), projection_sha256=digest(projection),
            fixed_geometry_sha256=projection["fixed_geometry"]["sha256"], owner_instance_id=projection["owner"]["instance_id"],
            non_target_sdf_m=values, outside=outside.tolist(), sdf_receipt=current["sdf_receipt"],
            sampling="conservative_min_eight_voxel_centres", full_body_validated=False)

def _purpose(purpose,store_root,output):
    require(purpose in ('production','test_fixture'),'Unknown runtime purpose')
    if purpose=='test_fixture':
        for path in (store_root,output):require('test_fixture' in Path(path).resolve().parts,
            'Fixture runtime requires explicitly isolated test_fixture paths')

def original_identity(current,sequence_path):
    binding,sequence=_production_source(sequence_path)
    stage1=read_sealed(verified(current['current_stage1_execution']))
    require(sequence['schema']=='p550.stage1_interaction_sequence_execution.v1'
        and sequence['status']=='complete_ordered_stage1_sequence' and sequence['interaction_count']==1
        and sequence['stage2_sequence_handoff_allowed'] is True and sequence['all_source_actions_retained'] is True
        and sequence['unsupported_actions']==[] and sequence['scene_id']==stage1['scene_id'],'Incomplete original task')
    step=sequence['interaction_steps'][0];target=current['context']['owner']['instance_id']
    require(step['stage1_execution']==current['current_stage1_execution'] and step['action']=='sit'
        and step['status']=='complete' and step['selected_target_id']==target
        and step['requested_target_ids']==[target],'Original/current target mismatch')
    compilation=read_sealed(verified(sequence['semantic_compilation']))
    raw=read_sealed(verified(sequence['semantic_plan']))
    require(compilation['schema']=='p550.ordered_interaction_semantic_compilation.v1'
        and compilation['source_actual_qwen_call']==sequence['semantic_plan']
        and raw['parsed']==compilation['raw_plan']==compilation['mapping']['raw_semantic_plan']
        and compilation['fixed_geometry']==sequence['source_fixed_geometry']==stage1['source_fixed_geometry']
        and compilation['mapping']['all_source_actions_preserved'] is True and compilation['unsupported_actions']==[],
        'Original semantic compilation mismatch')
    # Native producer identity and complete source-plan compilation are checked
    # again; a hash-sealed caller's all-actions-retained flag is not authority.
    from hsi.stage1 import pipeline, planning, qwen
    import json
    require(sequence['source'] == artifact(pipeline.__file__)
        and stage1['source_code'] == artifact(pipeline.__file__)
        and compilation['source'] == artifact(pipeline.__file__)
        and stage1['source_sequence_semantic_compilation'] == sequence['semantic_compilation'],
        'Unregistered original sequence producer')
    require(raw['schema'] == 'p550.qwen_local_text_semantic_call.v1'
        and raw['status'] == 'complete' and raw['source_code'] == artifact(qwen.__file__)
        and raw['runtime']['gpt_called'] is False
        and raw['runtime']['numeric_geometry_in_prompt'] is False
        and raw['runtime']['coordinates_generated_by_qwen'] is False,
        'Original task lacks a native actual semantic-only Qwen call')
    choice = raw['response']['choices'][0]
    require(choice['finish_reason'] == 'stop'
        and raw['response']['model'] == raw['model']['served_model_name']
        and json.loads(choice['message']['content']) == raw['parsed']
        and raw['response_format']['json_schema']['schema'] == planning.PLAN_SCHEMA,
        'Original task raw model response changed')
    geometry = read_sealed(verified(stage1['source_fixed_geometry']))
    semantics = read_sealed(verified(stage1['source_fixed_semantics']))
    compiled, recomputed_mapping, unsupported = planning.compile_supported_sequence(
        raw['parsed'], planning.inventory(semantics, geometry))
    require(compilation['compiled_plan'] == compiled and compilation['mapping'] == recomputed_mapping
        and unsupported == [] and raw['parsed']['needs_clarification'] is False,
        'Original ordered actions were not exactly compiled')
    derived = read_sealed(verified(stage1['semantic_plan']))
    require(derived['schema'] == 'p550.actual_qwen_sequence_step_binding.v1'
        and derived['source_call'] == sequence['semantic_plan']
        and derived['source_compilation'] == sequence['semantic_compilation']
        and derived['compiled_interaction_index'] == stage1['interaction_index'] == 0
        and derived['step'] == compiled['steps'][0] == stage1['interaction_plan']['steps'][0]
        and derived['new_qwen_call_performed'] is False,
        'Compiled child task differs from the actual original sequence')
    mapping=compilation['mapping']['execution_compilation'];steps=compilation['raw_plan']['steps']
    require(len(mapping)==1 and mapping[0]['source_step_indices']==step['source_step_indices']==list(range(len(steps)))
        and mapping[0]['actions_dropped']==[],'Original actions dropped')
    start=current['identity']['start_world_xy_m']
    require(start==sequence['initial_start_world_xy_m']==step['route_start_world_xy_m'],'Original start mismatch')
    identity=canonical_episode(scene_id=stage1['scene_id'],scene_fingerprint=current['context']['scene_fingerprint_sha256'],
        fixed_geometry_sha256=current['context']['fixed_geometry']['sha256'],instruction=sequence['instruction'],
        start_world_xy_m=start,ordered_actions=[{k:r[k] for k in ('action','target_ids','reference_ids')} for r in steps],
        body_model_sha256=current['body_model']['sha256'],betas=[0.]*10)
    require(identity['ordered_actions'][-1]['target_ids']==[target],'Original terminal owner mismatch')
    return identity,binding


def original_identity_from_stage1(stage1_path, sequence_path, body_model):
    """Shared native original-task identity for motion render and authority.

    Scene facts/SDF are not needed to verify the original text and actual body;
    the exact mesh/occupancy fingerprint uses the same namespace as facts.
    """
    from hsi.stage1.navigation import check_stage1_handoff
    verified(body_model)
    require(body_model['sha256'] == BODY_SHA256, 'Only the actual neutral body is registered')
    stage_binding, stage1 = _production_source(stage1_path)
    check_stage1_handoff(stage1_path)
    bound = stage1_bind(stage1)
    geometry = read_sealed(verified(stage1['source_fixed_geometry']))
    for name in ('source_mesh', 'source_occupancy'):
        verified(geometry[name])
    bundle = read_sealed(verified(stage1['bundle']))
    request = read_json(verified(bundle['artifacts']['roadmap_request']))
    fingerprint = digest({'mesh_sha256': geometry['source_mesh']['sha256'],
                          'occupancy_sha256': geometry['source_occupancy']['sha256']})
    current = {'current_stage1_execution': stage_binding,
        'context': {'owner': {'instance_id': bound['target_instance_id']},
            'scene_fingerprint_sha256': fingerprint, 'fixed_geometry': stage1['source_fixed_geometry']},
        'identity': {'start_world_xy_m': request['start_world_xy_m']}, 'body_model': body_model}
    return original_identity(current, sequence_path)

def prepare(current_stage1_execution,facts_receipt,body_model,sdf_receipt,store_root,output,*,
            original_sequence,stage='stage2_contact',body_shape_bin='neutral',purpose='production'):
    started=time.monotonic();_purpose(purpose,store_root,output);closure=sources()
    current=_current(current_stage1_execution,facts_receipt,body_model,sdf_receipt,stage,body_shape_bin)
    identity,sequence=original_identity(current,original_sequence);key=digest(identity)
    store=MemoryStore(store_root,purpose=purpose);state=store.recover()
    cycle=store.open_cycle(dict(query_id=key,episode_identity=key,scene_id=identity['scene_id'],
        scene_fingerprint_sha256=identity['scene_fingerprint'],scene_family=_family(identity['scene_id']),
        canonical_episode_identity=identity))
    require(sources()==closure,'Source changed during preparation')
    result=Path(output)/'receipt.json'
    write_once(result,dict(schema=PREPARATION,purpose=purpose,current=current,original_sequence=sequence,
        canonical_episode_identity=identity,episode_id=key,cycle=cycle,store_root=str(Path(store_root).resolve()),
        store_state=state,stage=stage,body_shape_bin=body_shape_bin,source_codes=closure,
        credit_granted_by_runtime=0,seed_output_and_stage_excluded_from_episode_identity=True,
        elapsed_seconds=time.monotonic()-started))
    return result

def validate_preparation(path,*,purpose='production'):
    value=read_sealed(path);require(value['schema']==PREPARATION and value['purpose']==purpose,'Runtime purpose/schema mismatch')
    _purpose(purpose,value['store_root'],path);require(value['source_codes']==sources(),'Runtime source drift')
    old=value['current'];current=_current(verified(old['current_stage1_execution']),verified(old['facts']),
        old['body_model'],verified(old['sdf_receipt']),value['stage'],value['body_shape_bin'])
    identity,sequence=original_identity(current,verified(value['original_sequence']))
    require(current==old and sequence==value['original_sequence'] and identity==value['canonical_episode_identity']
        and digest(identity)==value['episode_id'],'Current facts/body/original identity drift')
    cycle=value['cycle'];require(cycle['query_binding']==dict(query_id=value['episode_id'],episode_identity=value['episode_id'],
        scene_id=identity['scene_id'],scene_fingerprint_sha256=identity['scene_fingerprint'],
        scene_family=_family(identity['scene_id']),canonical_episode_identity=identity),'Cycle identity drift')
    store=MemoryStore(value['store_root'],purpose=purpose)
    store._cycle(cycle)
    return value,store

def _retrieve(prepared,store,scopes):
    callback=CurrentSDFPointFilter(prepared['current'])
    result=retrieve_current(store,prepared['cycle'],prepared['current']['context'],sdf_check=callback,scopes=scopes)
    require(result['purpose']==store.purpose,'Cross-purpose retrieval')
    return result,callback

def query(preparation_path,output,*,scopes=('scene_local','invariant'),purpose='production'):
    started=time.monotonic();binding=artifact(preparation_path);prepared,store=validate_preparation(preparation_path,purpose=purpose)
    result,callback=_retrieve(prepared,store,scopes);current=prepared['current'];count=len(result['records'])
    require(artifact(preparation_path)==binding and sources()==prepared['source_codes'],'Lookup source changed')
    _purpose(purpose,prepared['store_root'],output);destination=Path(output)/'receipt.json'
    write_once(destination,dict(schema=LOOKUP,purpose=purpose,preparation=binding,cycle=prepared['cycle'],
        canonical_episode_identity=prepared['canonical_episode_identity'],episode_id=prepared['episode_id'],
        retrieval=result,experience_count=count,m0_bypass=not count,credit_granted_by_runtime=0,
        real_experience_count=count if purpose=='production' else 0,
        provenance={'cache':dict(kind='verified_precomputed_scene_facts',artifact=current['facts'],learned_credit=0),
            'geometry_prior':dict(kind='current_typed_metric_surface',descriptor=current['context']['descriptor'],
                frame_convention=FRAME_CONVENTION,applied_to_keypose=False,learned_credit=0),
            'learned_experience':dict(kind='hsi_native_full_motion_authority' if purpose=='production' else 'isolated_test_fixture_only',
                count=count,applied_to_keypose=False,reason=result['m0_reason'])},
        current_sdf_filter=dict(source=current['sdf_receipt'],callback_count=callback.calls,
            scope='contact_and_root_points_not_full_body',full_body_validated=False),
        consumption_requires_live_revalidation=True,original_23_and_qwen_gates_still_required=True,
        stage3_handoff_registered=False,elapsed_seconds=time.monotonic()-started))
    return destination

def validate_lookup(path,*,purpose='production'):
    """Re-run active-only fixed-snapshot query and live quarantine/current veto."""
    binding=artifact(path);lookup=read_sealed(path)
    require(lookup['schema']==LOOKUP and lookup['purpose']==purpose,'Lookup schema/purpose mismatch')
    prepared,store=validate_preparation(verified(lookup['preparation']),purpose=purpose)
    require(lookup['cycle']==prepared['cycle'] and lookup['episode_id']==prepared['episode_id']
        and lookup['canonical_episode_identity']==prepared['canonical_episode_identity'],'Lookup identity mismatch')
    result,_=_retrieve(prepared,store,lookup['retrieval']['scope_filter'])
    require(result==lookup['retrieval'],'Live quarantine, current geometry or retrieval changed since lookup')
    count=len(result['records']);require(lookup['experience_count']==count and lookup['m0_bypass'] is (count==0)
        and lookup['real_experience_count']==(count if purpose=='production' else 0)
        and lookup['credit_granted_by_runtime']==0,'Lookup count/credit tampering')
    require(artifact(path)==binding,'Lookup changed during validation')
    return lookup,prepared,store
