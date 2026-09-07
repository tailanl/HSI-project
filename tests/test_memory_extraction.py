from pathlib import Path
import sys
import numpy as np
import pytest
from hsi.common.artifacts import MemoryContractError,digest,artifact,write_once
from hsi.memory.matching import FRAME_CONVENTION
from hsi.memory.schema import validate_record
from test_memory_store import record
from hsi.memory.extraction import derive_records,write_receipts,check_receipt


def data():
    r=record({'scene_fingerprint_sha256':'c'*64})
    context={'frame_convention':FRAME_CONVENTION,'stage':'stage2_contact','invariant':r['invariant'],
        'shape':r['shape'],'scene_fingerprint_sha256':'c'*64,
        'frame':{'local_to_world_rotation':np.eye(3).tolist(),'origin_world_zup_m':[10.,20.,.4]}}
    joints=np.zeros((6,22,3));joints[:,:,:]=[10.,20.,.7]
    joints[:,1,:]=[10.,20.1,.7];joints[:,2,:]=[10.,19.9,.7]
    tail={'actual_source_frame_indices':[2,3,4,5],'frames':[{'frame_index':i,
        'metrics':{'measured_gluteal_anchor_world_xyz_m':[10.,20.,.42]}} for i in (2,3,4,5)]}
    triangles=np.array([[[9.,19.,.4],[11.,19.,.4],[11.,21.,.4]],[[9.,19.,.4],[11.,21.,.4],[9.,21.,.4]]])
    return context,{'joints':joints},tail,triangles


def test_observed_local_payload_no_fullpose(tmp_path):
    args=data();records,proof=derive_records(*args)
    assert len(records)==2 and proof['observed_frame_index']==2
    for r in records:
        validate_record(r)
        assert np.allclose(r['payload']['contact_point_local_xyz_m'],[0,0,.02])
        assert np.allclose(r['payload']['root_offset_local_xyz_m'],[0,0,.28])
        assert np.allclose(r['payload']['facing_local_xy'],[1,0])
        assert r['payload']['contact_phase']==1.0
        assert not {'joints','pose','motion','world','route'}&set(r['payload'])
    source=tmp_path/'episode.json';geometry=tmp_path/'facts.json';write_once(source,{'schema':'isolated_fixture'});write_once(geometry,{'schema':'isolated_fixture'})
    bindings=write_receipts(tmp_path/'extracted',episode_id='d'*64,records=records,
        source_episode=artifact(source),source_geometry=artifact(geometry),extraction=proof)
    for b,r in zip(bindings,records):
        check_receipt(b,episode_id='d'*64,record=r,source_episode=artifact(source),source_geometry=artifact(geometry),extraction=proof)
    with pytest.raises(MemoryContractError):check_receipt(bindings[0],episode_id='e'*64,record=records[0],
        source_episode=artifact(source),source_geometry=artifact(geometry),extraction=proof)


@pytest.mark.parametrize('bad',['role','stage','frame','triangle','scale','scopes'])
def test_invalid_geometry_typed_extraction(bad):
    context,motion,tail,triangles=data();kwargs={}
    if bad=='role':context['invariant']['surface_role']='walkable_support'
    if bad=='stage':context['stage']='stage3_guidance'
    if bad=='frame':tail['actual_source_frame_indices']=[1,2,3,4]
    if bad=='triangle':triangles[:,:,2]+=.2
    if bad=='scale':context['frame']['local_to_world_rotation'][0][0]=2
    if bad=='scopes':kwargs['scopes']=('world_motion',)
    with pytest.raises(MemoryContractError):derive_records(context,motion,tail,triangles,**kwargs)
