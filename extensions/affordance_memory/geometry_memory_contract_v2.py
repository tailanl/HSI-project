"""Narrow CPU lineage contract for opt-in Stage2 local Memory v2.

No Qwen/GL; current proposal, typed lookup and exact candidate are verified.
The optional CPU numerical replay preserves all retained 23 predicates and
records rejection instead of throwing solely because a numerical gate failed.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap
import types
import numpy as np
from PIL import Image
from memory_common import PROJECT,artifact,digest,read_sealed,read_json,verified,require,write_once
import memory_runtime_v2 as runtime
import fast_keypose_memory_v2 as provider
import compile_geometry_stage3 as numeric
import render_geometry_keypose_v2 as retained_render

HERE=Path(__file__).resolve().parent
SCHEMA='p555.actual_geometric_memory_provider_render.v2'
REVIEW='p555.actual_geometry_memory_keypose_qwen_review.v2'
PROPOSAL='p555.geometry_guided_keypose_proposal.v2'
NUMERIC='p555.memory_v2_exact_candidate_recheck.v1'
PINS={'fast_keypose_memory_v2.py':'c06d8c62e84f40e70db32b8f3b77e23b7d013ef3ee25fbaeb556d56712d216ba',
    'memory_runtime_v2.py':'ab66b9dea3d05544434b5cd7e37441936d39d1b096690713741422f2b9a9949c',
    'compile_geometry_stage3.py':'b9d18293f428f45ef6c7666b7de90c66c4b7f9109cb8be2c439e6458983bb7a7',
    'render_geometry_keypose_v2.py':'fd4ec8d14080857dacfb72ca641bf9ba12890efefa972876363eda919c42b9ac',
    'audit_geometry_keypose.py':'493e0a55281b3182f6a0bbfa79cce1e59a52c6a22f1a885215d50e231ab3bbe7'}


def unique_sources(rows):
    result={}
    for row in rows:
        path=str(Path(row['path']).resolve())
        require(path not in result or result[path]==row,'Conflicting source artifact')
        result[path]=row
    return list(result.values())


def sources():
    rows=[]
    for name,sha in PINS.items():
        row=artifact(HERE/name);require(row['sha256']==sha,'Frozen Stage2 source drift: '+name);rows.append(row)
    for rel,sha in {
        'p548_qwen38_full_pipeline_20260905/code/qwen_api.py':'416021a370aefea00ace569fc22c5f40f7e59c6c4c5d16cba151ebeecff10a8b',
        'p550_scene_understanding_decoupled_20260905/code/compact_audit_contract.py':'009d49bc780caea4a651c61649af5cfaf2f89938319260a4b148c3996a7f0959',
        'p550_scene_understanding_decoupled_20260905/code/qwen_scheduling.py':'6bc57677d415ce08cc51adcbef71384820b909b899ea34eb9fe7ecfcee904d04',
    }.items():
        row=artifact(PROJECT/'agent9/methods'/rel);require(row['sha256']==sha,'Frozen Qwen protocol changed');rows.append(row)
    return unique_sources([artifact(__file__),*rows,*runtime.sources()])


def gates(values):
    require(isinstance(values,dict) and set(values)==set(numeric.NUMERIC_GATES)
        and all(type(v) is bool for v in values.values()),'Exact original 23 Boolean gates required')
    return all(values.values())


def load_candidate(binding):
    source=inspect.getsource(numeric.load_candidate)
    require(source.count('p555.current_geometry_ik_candidate.v1')==1,'Candidate loader changed')
    source=source.replace('p555.current_geometry_ik_candidate.v1','p555.current_geometry_ik_candidate.v2')
    scope=dict(vars(numeric));exec(compile(source,__file__,'exec'),scope)
    return scope['load_candidate'](binding)


class ArrayValue:
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def numpy(self):return self.value


def validate_proposal(path,*,purpose='production'):
    closure=sources();binding=artifact(path);proposal=read_sealed(path)
    require(proposal['schema']==PROPOSAL and proposal['provider']=='declared_analytic_sit_seed_plus_typed_local_memory_current_geometry_ik_v2',
        'Unregistered memory proposal/provider')
    require(proposal['source']['sha256']==PINS['fast_keypose_memory_v2.py'],'Unregistered memory provider source')
    verified(proposal['source'])
    for row in [*proposal['inputs'].values(),*proposal['source_dependencies']]:verified(row)
    passed=gates(proposal['gates']);require(proposal['status']==('numeric_candidate_pass' if passed else 'numeric_candidate_rejected'),
        'Proposal status contradicts exact numerical gates')
    require(proposal['thresholds']==numeric.THRESHOLDS and proposal['source_hashes_rechecked_after_optimization'] is True,
        'Changed physical thresholds or missing source recheck')
    for name in ('h3_model_used','hybrikx_model_used','qwen_checked','historical_full_pose_or_motion_read','stage3_handoff_allowed','semantic_publication_allowed'):
        require(proposal[name] is False,'Unsupported inherited success/provider claim')
    require(proposal['real_positive_credit']==0,'A keypose cannot grant motion credit')
    consumption=proposal['memory_consumption'];require(consumption['schema']=='p555.typed_local_memory_consumption.v2'
        and consumption['purpose']==purpose,'Memory consumption purpose/schema mismatch')
    lookup,prepared,store=runtime.validate_lookup(verified(consumption['lookup']),purpose=purpose)
    require(consumption['source']==artifact(provider.__file__) and consumption['source_codes']==runtime.sources()
        and consumption['retained_provider']==artifact(provider.retained.__file__),'Consumption source closure differs')
    require(prepared['current']['current_stage1_execution']==proposal['inputs']['stage1']
        and prepared['current']['sdf_receipt']==proposal['inputs']['sdf_cache']
        and prepared['current']['body_model']==proposal['inputs']['body_model']
        and prepared['canonical_episode_identity']==consumption['canonical_episode_identity']
        and prepared['episode_id']==consumption['episode_id'],'Memory/provider current identity mismatch')
    current=provider.retained.load_current(verified(proposal['inputs']['stage1']))
    yaw=current['binding']['contact_forward_yaw_rad']
    require(proposal['terminal_facing_yaw_rad']==yaw and proposal['target_instance_id']==current['target']['target_instance_id'],
        'Memory proposal changed audited owner/facing')
    base=provider.retained.contact_candidates(current['option']['p552_region_member']['contact_anchor_world_xyz_m'],
        current['surface']['surface']['centre_world_xyz_zup_m'],np.array([np.cos(yaw),np.sin(yaw)]),current['triangles'],proposal['contact_policy'])
    # Replay the exact provider's pure nested candidate/veto function in private
    # globals, including post-triangle-snap current SDF checks. No IK/model load.
    tree=ast.parse(textwrap.dedent(inspect.getsource(provider.FastKeyposeEngine.propose)))
    node=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='candidates')
    state={};scope={**vars(provider),'lookup':lookup,'prepared':prepared,'state':state}
    exec(compile(ast.Module(body=[node],type_ignores=[]),__file__,'exec'),scope)
    scope['candidates'](base,current,yaw)
    for name,expected in [('accepted_projections',state['selected']),('rejected_projections',state['rejected']),
        ('original_candidate_count',state['original_candidate_count']),('actual_candidate_count',state['actual_candidate_count'])]:
        require(consumption[name]==expected,'Actual memory candidate/ranking lineage mismatch: '+name)
    arrays=load_candidate(proposal['outputs']['candidate']);initial=load_candidate(proposal['outputs']['initial'])
    selected=provider.selection(ArrayValue(initial['contact_world_xyz_m']),types.SimpleNamespace(root=ArrayValue(initial['root_xyz_yaw'][:3])),yaw,state['selected'])
    require(consumption['selection']==selected and proposal['memory_records_consumed']==selected['records_selected'],
        'Consumed record is not the actual selected initial candidate')
    require(np.array_equal(arrays['contact_world_xyz_m'],initial['contact_world_xyz_m'])
        and np.array_equal(arrays['contact_world_xyz_m'],proposal['contact_world_xyz_m'])
        and np.array_equal(initial['body_pose_axis_angle'],provider.retained.sit_seed()),'Pose/contact initialization drift')
    require(consumption['m0_bypass'] is (not state['selected']) and consumption['real_positive_credit_granted']==0
        and consumption['real_records_selected']==(len(selected['records_selected']) if purpose=='production' else 0)
        and consumption['policy']==provider.POLICY and consumption['stage3_handoff_registered'] is False,
        'Consumption count/credit/bounds mismatch')
    require(sources()==closure and artifact(path)==binding,'Source changed during proposal validation')
    return dict(proposal=proposal,source_proposal=binding,lookup=lookup,prepared=prepared,store=store,current=current,arrays=arrays,
        numeric_pass=passed,sources=closure)


def numerical_recheck(proposal_path,output):
    context=validate_proposal(proposal_path)
    def schema_only(values,names,label):
        require(tuple(names)==numeric.NUMERIC_GATES,'Retained gate set changed');gates(values)
    function=types.FunctionType(numeric.numerical_recheck.__code__,{**vars(numeric),'_all_true':schema_only})
    proof,_,_=function(context['proposal'],context['arrays'],context['current'])
    require(proof['gates']==context['proposal']['gates'],'Recomputed current numerical gates differ')
    value=dict(schema=NUMERIC,source=artifact(__file__),source_retained_numeric=artifact(numeric.__file__),
        source_proposal=context['source_proposal'],source_candidate=context['proposal']['outputs']['candidate'],
        retained_result=proof,all_original_23_gates_passed=gates(proof['gates']),
        original_predicates_and_thresholds_unchanged=True,recomputed_actual_same_body_mesh=True,stage3_handoff_allowed=False,real_positive_credit=0)
    require(sources()==context['sources'],'Numerical source drift');write_once(output,value)
    return value


def validate_numeric(path,context):
    value=read_sealed(path);require(value['schema']==NUMERIC and value['source']==artifact(__file__)
        and value['source_retained_numeric']==artifact(numeric.__file__) and value['source_proposal']==context['source_proposal']
        and value['source_candidate']==context['proposal']['outputs']['candidate'],'Wrong numeric replay lineage')
    proof=value['retained_result'];require(proof['gates']==context['proposal']['gates'] and proof['thresholds']==numeric.THRESHOLDS
        and value['all_original_23_gates_passed'] is gates(proof['gates'])
        and proof['input_bindings']=={'candidate':context['proposal']['outputs']['candidate'],**context['proposal']['inputs']},'Numeric replay changed')
    for row in proof['numerical_sources']:verified(row)
    return value


def mask_evidence(row,vertices):
    masks=[]
    for name in ('amodal','scene_visible'):
        with Image.open(verified(row['masks'][name])) as image:
            require(image.mode=='L' and image.size==(640,640),'Mask dimensions/type changed');array=np.array(image)
        require(set(np.unique(array))<={0,255},'Mask must be binary');masks.append(array>0)
    amodal,visible=masks;require(amodal.any() and not np.any(visible&~amodal),'Invalid paired visible mask')
    require(type(row['unoccluded_body_mask_pixels']) is int and type(row['visible_body_mask_pixels']) is int
        and int(amodal.sum())==row['unoccluded_body_mask_pixels'] and int(visible.sum())==row['visible_body_mask_pixels']
        and row['actual_body_visibility_fraction']==float(visible.sum()/amodal.sum()),'Actual saved mask counts differ')
    require(row['scene_occluders_hidden'] is False and row['visibility_contract']=='all_original_scene_mesh_nodes_black_body_white_v2',
        'Occluder-skipping mask contract')
    require(row['camera_body_coverage']==retained_render.camera_body_coverage(vertices,row['world_to_camera'],row['K'],640,640),
        'Actual body/camera coverage differs')
    verified(row['image'])


def validate_render(path):
    binding=artifact(path);render=read_sealed(path)
    require(render['schema']==SCHEMA,'Wrong typed memory renderer schema')
    context=validate_proposal(verified(render['source_proposal']))
    require(render['source']['sha256']==artifact(HERE/'render_geometry_keypose_memory_v2.py')['sha256']
        and render['source_retained_renderer']==artifact(retained_render.__file__),'Unregistered typed renderer')
    verified(render['source']);validate_numeric(verified(render['source_numeric_recheck']),context)
    proposal=context['proposal'];view=read_sealed(verified(render['source_view_selection']));selection=view['selection']
    require(render['source_keypose']==proposal['outputs']['candidate'] and render['source_stage1']==proposal['inputs']['stage1']
        and view['source_stage1_execution']==proposal['inputs']['stage1'] and view['target']==context['current']['stage1']['target'],
        'Cross-current render/candidate/camera')
    require(render['original_camera']==selection['selected_camera'] and render['original_reference']==selection['selected_stage2_crop']
        and render['target_overlay']==selection['selected_stage2_crop_target_overlay'] and render['image_crop_xyxy']==selection['selected_crop_xyxy'],
        'Original camera/reference/crop changed')
    require(render['original_mesh']==context['current']['target']['artifacts']['original_scene_mesh'],'Original mesh drift')
    for name in ('original_scene_rendered','isolated_views_have_occluders_hidden','masks_count_all_scene_mesh_occluders','visible_and_amodal_masks_saved'):
        require(render[name] is True,'Missing scene/mask contract')
    for name in ('original_scene_vertices_textures_or_camera_changed','actual_body_vertices_changed','qwen_checked'):
        require(render[name] is False,'Altered scene/body/review claim')
    require(type(render['mask_contract_version']) is int and render['mask_contract_version']==2,'Wrong scene SEG version')
    extra=render['additional_original_scene_views'];require(len(extra)==2 and len(render['isolated_body_views'])==2,'Exact six-image view layout required')
    for row in extra:mask_evidence(row,context['arrays']['vertices_world_zup'])
    index=max(range(2),key=lambda i:(extra[i]['actual_body_visibility_fraction'],-i))
    require(type(render['selected_additional_view_index']) is int and render['selected_additional_view_index']==index,'Numeric camera selection drift')
    images=[render['original_reference'],render['target_overlay'],render['images']['original_scene_with_actual_body_crop'],
        *[r['image'] for r in render['isolated_body_views']],extra[index]['image']]
    for row in [*render['images'].values(),*images,render['original_camera'],render['source_scene_loader']]:verified(row)
    for row in images:
        with verified(row).open('rb') as stream:require(stream.read(8)==b'\x89PNG\r\n\x1a\n','Actual semantic image is not PNG')
    require(len(images)==6 and render['memory_lookup']==proposal['memory_consumption']['lookup']
        and render['canonical_episode_identity']==context['prepared']['canonical_episode_identity'],'Actual image/memory lineage drift')
    require(artifact(path)==binding and sources()==context['sources'],'Render source changed')
    return render,context,images
