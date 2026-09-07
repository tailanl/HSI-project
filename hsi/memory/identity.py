"""Complete episode identity; neither RNG seed nor output path grants credit."""
import math
from hsi.common.artifacts import require


def canonical_episode(*,scene_id,scene_fingerprint,fixed_geometry_sha256,instruction,
        start_world_xy_m,ordered_actions,body_model_sha256,betas):
    require(isinstance(scene_id,str) and scene_id and isinstance(instruction,str) and instruction.strip(),'Missing episode text')
    for value in (scene_fingerprint,fixed_geometry_sha256,body_model_sha256):
        require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'Invalid identity SHA')
    require(isinstance(start_world_xy_m,list) and len(start_world_xy_m)==2
        and all(type(x) in (int,float) and math.isfinite(x) for x in start_world_xy_m),'Invalid current start')
    require(isinstance(betas,list) and len(betas)==10 and all(type(x) in (int,float) and x==0 for x in betas),'Only exact neutral body')
    require(isinstance(ordered_actions,list) and ordered_actions,'Missing ordered actions')
    for action in ordered_actions:
        require(set(action)=={'action','target_ids','reference_ids'} and action['action'] in {'walk','sit'}
            and isinstance(action['target_ids'],list) and len(action['target_ids'])==1
            and isinstance(action['target_ids'][0],str) and action['target_ids'][0]
            and action['reference_ids']==[],'Only single-target walk/sit identity is registered')
    require([x['action'] for x in ordered_actions] in (['sit'],['walk','sit'])
        and len({x['target_ids'][0] for x in ordered_actions})==1,'Unregistered complete task')
    return dict(schema='p555.canonical_episode_identity.v2',scene_id=scene_id,
        scene_fingerprint=scene_fingerprint,fixed_geometry_sha256=fixed_geometry_sha256,
        instruction=instruction,start_world_xy_m=[float(x) for x in start_world_xy_m],
        ordered_actions=[{k:list(v) if isinstance(v,list) else v for k,v in a.items()} for a in ordered_actions],
        body_model_sha256=body_model_sha256,betas=[float(x) for x in betas])

