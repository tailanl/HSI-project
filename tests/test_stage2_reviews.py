"""Fixed audit thresholds, visibility-only recovery and actual Qwen provenance."""
import copy
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from hsi.common.artifacts import write_once, read_sealed
from hsi.stage1.qwen import QwenTextClient
from hsi.stage2 import image_review, mesh_review, review_provenance, background_numeric, pipeline


def h3_judgement():
    return {**dict.fromkeys(image_review.CHECKS,True),'foot_contact_needs_refine':False,
            'needs_more_evidence':False,'confidence':.9,'reason':'fixture'}


@pytest.mark.parametrize('failed',['target_correct','action_complete','one_person','limbs_plausible','scene_preserved','needs_more_evidence'])
def test_nonvisibility_failure_cannot_enter_refine(failed):
    value=h3_judgement();value[failed]=failed=='needs_more_evidence'
    audit={'judgement':value,'decision':image_review.verdict(value)}
    assert not audit['decision']['semantic_prior_eligible_for_refine']
    assert not image_review.partial_prior_eligible(audit)


@pytest.mark.parametrize('confidence,allowed',[(.84,False),(.85,True),(.9,True)])
def test_only_visibility_deferred_with_higher_confidence(confidence,allowed):
    value=h3_judgement();value.update(body_complete=False,feet_visible=False,confidence=confidence)
    assert image_review.partial_prior_eligible({'judgement':value,'decision':image_review.verdict(value)}) is allowed


@pytest.mark.parametrize('check',mesh_review.CHECKS)
def test_every_final_visual_check_is_mandatory(check):
    value={**dict.fromkeys(mesh_review.CHECKS,True),'confidence':.9,'reason':'fixture'}
    value[check]=False
    assert mesh_review.decide(value)=={'post_refine_semantic_pass':False,'failed_checks':[check]}


def test_background_actual_png_math_and_scene_drift_rejection(tmp_path):
    rng=np.random.default_rng(12)
    rgb=rng.integers(0,255,(512,512,3),dtype=np.uint8)
    reference=tmp_path/'reference.png';same=tmp_path/'same.png';changed=tmp_path/'changed.png'
    Image.fromarray(rgb).save(reference);Image.fromarray(rgb).save(same);Image.fromarray(255-rgb).save(changed)
    actual,mask,difference=background_numeric.measure(reference,same,[180,100,330,400])
    assert actual['passed'] and not difference.any()
    other,_,_=background_numeric.measure(reference,changed,[180,100,330,400])
    assert not other['passed'] and not other['gates']['background_structure_preserved']


def test_actual_transport_receipt_review_roundtrip_and_image_order_rejected(tmp_path,monkeypatch):
    client=QwenTextClient('http://127.0.0.1:9000/v1','Qwen/cpu-fixture')
    judgement=h3_judgement()
    raw={'checks':[judgement[k] for k in image_review.TRANSPORT_CHECKS],'confidence':.9,'reason':'fixture'}
    def transport(self,suffix,payload=None):
        if suffix=='/models':return {'data':[{'id':self.model}]}
        return {'id':'synthetic-response','model':self.model,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(raw)}}]}
    monkeypatch.setattr(QwenTextClient,'_json',transport)
    images=[]
    for index in range(3):
        path=tmp_path/f'image{index}.png';Image.new('RGB',(512,512),(index,20,30)).save(path);images.append(path)
    output=tmp_path/'audit/receipt.json'
    value=image_review.audit(*images,'Sit on this chair','chair',output,'synthetic_cpu_fixture',qwen_client=client)
    assert review_provenance.validate(output,'h3')['actual_image_order_verified']
    altered=copy.deepcopy(value)
    altered['inputs']['reference'],altered['inputs']['marked_reference']=altered['inputs']['marked_reference'],altered['inputs']['reference']
    changed=tmp_path/'wrong.json';write_once(changed,altered)
    with pytest.raises(ValueError,match='another image'):review_provenance.validate(changed,'h3')


def test_caller_pass_flag_cannot_authorize_stage3(tmp_path):
    path=tmp_path/'forged.json'
    write_once(path,{'schema':pipeline.SCHEMA,'status':'complete_verified_stage2_keypose','verified_stage2_keypose':True,
        'stage3_handoff_allowed':True,'source_closure':{}})
    with pytest.raises(ValueError,match='source/schema'):pipeline.validate_success(path)
