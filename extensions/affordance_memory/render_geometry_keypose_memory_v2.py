"""Typed Memory v2 wrapper; frozen actual-scene render2/SEG geometry unchanged."""
import argparse
import inspect
from pathlib import Path
import geometry_memory_contract_v2 as contract
import render_geometry_keypose_v2 as retained
from memory_common import artifact,verified,require

SOURCE=Path(__file__).resolve()


def compiled(context,numeric_binding):
    source=inspect.getsource(retained.run)
    replacements={'p555.geometry_guided_keypose_proposal.v1':contract.PROPOSAL,
        'p555.current_geometry_ik_candidate.v1':'p555.current_geometry_ik_candidate.v2',
        'p555.actual_geometric_provider_render.v1':contract.SCHEMA,
        'render_geometry_keypose_v2_source.py':'render_geometry_keypose_memory_v2_source.py',
        'verified(source_at_start)':'memory_revalidate()\n    verified(source_at_start)',
        'source_proposal=artifact(proposal_path),':'source_proposal=artifact(proposal_path), source_retained_renderer=artifact(retained_path),\n        source_numeric_recheck=numeric_binding, memory_lookup=memory_lookup, canonical_episode_identity=episode_identity,'}
    for old,new in replacements.items():
        require(source.count(old)==1,'Frozen renderer seam changed: '+old);source=source.replace(old,new)
    before=artifact(SOURCE)
    def revalidate():
        current=contract.validate_proposal(verified(context['source_proposal']))
        contract.validate_numeric(verified(numeric_binding),current)
        require(artifact(SOURCE)==before and current['sources']==context['sources'],'Typed renderer source changed')
    scope={**vars(retained),'SOURCE':SOURCE,'__file__':str(SOURCE),'memory_revalidate':revalidate,
        'retained_path':retained.__file__,'numeric_binding':numeric_binding,
        'memory_lookup':context['proposal']['memory_consumption']['lookup'],
        'episode_identity':context['prepared']['canonical_episode_identity']}
    exec(compile(source,str(SOURCE),'exec'),scope)
    result=scope['run'];result.__adapted_source__=source
    return result


def run(proposal_path,view_path,output,*,numeric_recheck,gpu=5):
    from run_geometry_stage2_memory_v2 import require_gl_hook
    require_gl_hook(gpu)
    context=contract.validate_proposal(proposal_path)
    binding=artifact(numeric_recheck);contract.validate_numeric(verified(binding),context)
    return compiled(context,binding)(proposal_path,view_path,output,gpu)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--view',type=Path,required=True);p.add_argument('--numeric-recheck',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=5)
    a=p.parse_args();run(a.proposal,a.view,a.output,numeric_recheck=a.numeric_recheck,gpu=a.gpu)
