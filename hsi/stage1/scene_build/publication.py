from ._common import *
from ..scene import verify_artifact_tree
from .checkpoints import phase,reuse
from .report import write_report
from .semantic_contract import SEATING
from .semantic_revision import review_decision
from . import geometry as geometry_builder, shapes as shape_builder, components as component_builder
from . import object_views,context_review,semantic_revision,direction,direction_revision,backrest

def review_context(scene_path, root, gpu, *, qwen_client):
    started = time.monotonic()
    scene = read_sealed(scene_path)
    snapshot_path = verified(scene['fixed_semantics'])
    snapshot = read_sealed(snapshot_path)
    final = root / 'receipt.json'
    if final.exists():
        result = read_sealed(final)
        if result['source_scene_receipt'] != artifact(scene_path):
            raise ValueError('Cannot resume a different source scene')
        verify_artifact_tree(result)
        return result
    root.mkdir(parents=True, exist_ok=True)
    launch = root / 'launch.json'
    sources = {'source_scene_receipt': artifact(scene_path), 'source_semantics': artifact(snapshot_path)}
    if launch.exists():
        if read_sealed(launch)['sources'] != sources:
            raise ValueError('Context-review source drift')
    else:
        write_once(launch, {'schema': 'p550.scene_only_context_publication_launch.v1', 'sources': sources, 'source': artifact(__file__), 'human_instruction_read': False}, seal=True)
    candidates = [r for r in snapshot['objects'] if r['decision']['query_usable'] and r['semantic']['is_sittable_object'] and (r['semantic']['object_class'] in SEATING) and (not r.get('represented_by_logical_object_id'))]
    (reviews, errors) = ({}, [])
    for row in candidates:
        key = row['instance_id']
        try:
            path = phase(root, 'context_' + key, lambda out: context_review.recover(snapshot_path, verified(scene['fixed_geometry']), key, out, gpu, qwen_client=qwen_client))
            review = read_sealed(path)
            if review['source_snapshot'] != artifact(snapshot_path) or review['source_object'] != row:
                raise ValueError('Original context review is not bound to this object snapshot')
            review_decision(review)
            reviews[key] = path
        except Exception as error:
            errors.append({'instance_id': key, 'error': f'{type(error).__name__}: {error}'})
    revised_path = root / 'fixed_semantics/receipt.json'
    if not revised_path.exists():
        revised = copy.deepcopy(snapshot)
        for row in revised['objects']:
            key = row['instance_id']
            if key in reviews:
                review = read_sealed(reviews[key])
                (semantic, decision) = review_decision(review)
                row['prior_semantic_evidence'] = {k: row[k] for k in ('semantic', 'decision', 'evidence')}
                row.update(semantic=semantic, decision=decision, evidence=artifact(reviews[key]), original_rgb_context_review_completed=True)
            elif key in {r['instance_id'] for r in candidates}:
                row['prior_semantic_evidence'] = {k: copy.deepcopy(row[k]) for k in ('semantic', 'decision', 'evidence')}
                row['decision'].update(query_usable=False, needs_visual_recovery=True)
                row['decision']['reasons'].append('original_context_review_incomplete')
                row['original_rgb_context_review_completed'] = False
            elif row.get('represented_by_logical_object_id'):
                row['decision']['query_usable'] = False
        revised.update(source_snapshot_before_context_revision=artifact(snapshot_path), status='scene_understanding_complete_with_unknowns' if snapshot['errors'] or errors or any((r['decision']['needs_visual_recovery'] for r in revised['objects'])) else 'scene_understanding_complete', context_review_errors=errors, context_review_publication_gate_applied=True, every_published_sitting_identity_has_original_context_review=True, source_code=artifact(__file__), geometry_snapshot_pending=True, task_instruction_read=False, future_motion_read=False, original_atoms_or_prior_evidence_deleted=False)
        write_once(revised_path, revised, seal=True)
    geometry = phase(root, 'context_geometry', lambda out: geometry_builder.run(revised_path, out))
    shapes = phase(root, 'generic_shapes', lambda out: shape_builder.run(geometry, out))
    report = write_report(root, revised_path, shapes)
    result_geometry = read_sealed(shapes)
    result = {'schema': 'p550.context_validated_fixed_scene_publication.v1', 'scene_id': scene['scene_id'], 'source_scene_receipt': artifact(scene_path), 'source_semantics': artifact(snapshot_path), 'fixed_semantics': artifact(revised_path), 'fixed_geometry': artifact(shapes), 'report': artifact(report), 'review_candidate_ids': [r['instance_id'] for r in candidates], 'context_reviews': {k: artifact(v) for (k, v) in reviews.items()}, 'context_review_errors': errors, 'sit_usable_ids': [r['instance_id'] for r in result_geometry['objects'] if r['query_usable_for_sit']], 'status': 'context_reviewed_with_unknowns', 'human_instruction_read': False, 'current_or_future_human_motion_read': False, 'stage2_or_stage3_output_read': False, 'affordance_memory_used': False, 'source_code': artifact(__file__), 'elapsed_seconds': time.monotonic() - started}
    write_once(final, result, seal=True)
    print({'scene': scene['scene_id'], 'context_reviewed': len(reviews), 'review_errors': len(errors), 'sit_usable': result['sit_usable_ids']}, flush=True)
    return result

def projection_views(review_path, depth=0):
    if depth > 4:
        raise ValueError('Unexpectedly deep scene evidence recovery chain')
    direct = review_path.parent / 'mesh_projection_evidence_extension.json'
    if direct.exists():
        ext = read_sealed(direct)
        if ext['actual_review'] != artifact(review_path):
            raise ValueError('Mesh context evidence extension mismatch')
        return verified(ext['source_object_views'])
    wrapper = review_path.parent / 'bounded_scene_recovery_extension.json'
    if wrapper.exists():
        ext = read_sealed(wrapper)
        if ext['published_review'] != artifact(review_path):
            raise ValueError('Bounded context evidence extension mismatch')
        return projection_views(verified(ext['actual_review']), depth + 1)
    return None

def review_geometry(scene_path, output, gpu, *, qwen_client):
    started = time.monotonic()
    scene = read_sealed(scene_path)
    if scene['schema'] != 'p550.context_validated_fixed_scene_publication.v1' or scene['human_instruction_read'] is not False:
        raise ValueError('Use a scene-only original-context reviewed inventory')
    output.mkdir(parents=True, exist_ok=True)
    launch_path = output / 'launch.json'
    if launch_path.exists():
        launch = read_sealed(launch_path)
        if launch['source_context_scene'] != artifact(scene_path) or launch['gpu'] != gpu:
            raise ValueError('Geometry-review scene source changed')
        verify_artifact_tree(launch)
    else:
        write_once(launch_path, {'schema': 'p550.scene_geometry_review_launch.v5', 'source_context_scene': artifact(scene_path), 'gpu': gpu, 'source': artifact(__file__), 'human_query_or_stage2_inputs': False, 'direction_review_policy': 'three_independent_single_camera_calls_all_must_agree'}, seal=True)
    final = output / 'receipt.json'
    if final.exists():
        result = read_sealed(final)
        verify_artifact_tree(result)
        return result
    snapshot_path = verified(scene['fixed_semantics'])
    scan = phase(output, 'component_scan', lambda out: component_builder.scan(snapshot_path, out / 'receipt.json'))
    scan_rows = read_sealed(scan)['objects']
    blocked_spatial_ids = {r['instance_id'] for r in scan_rows if r.get('automatic_geometry_publication') is False or r.get('ambiguous_disjoint_support_requires_review') is True}
    (errors, components) = ([], [])
    for row in scan_rows:
        if not row['candidate_proposed']:
            continue
        key = row['instance_id']
        try:
            proposal = phase(output, 'component_' + key, lambda out: component_builder.run(snapshot_path, key, out))
            snapshot_path = proposal
            proposal_evidence = read_sealed(proposal.parent / 'component_proposal.json')
            child = proposal_evidence['candidate_id']
            geo = phase(output, 'component_geometry_' + key, lambda out: geometry_builder.run(snapshot_path, out))
            views = phase(output, 'component_views_' + key, lambda out: object_views.run(geo, [child], out, gpu), gpu=gpu)
            review = phase(output, 'component_context_' + key, lambda out: context_review.review_projection(snapshot_path, views, child, out, qwen_client=qwen_client))
            snapshot_path = phase(output, 'component_semantics_' + key, lambda out: semantic_revision.run(snapshot_path, [review], out))
            components.append({'parent_id': key, 'candidate_id': child, 'proposal': artifact(proposal), 'views': artifact(views), 'review': artifact(review), 'decision': read_sealed(review)['review_decision']})
        except Exception as error:
            errors.append({'instance_id': key, 'phase': 'spatial_component_recovery', 'reason': f'{type(error).__name__}: {error}'})
            blocked_spatial_ids.add(key)
    geometry = phase(output, 'final_geometry', lambda out: geometry_builder.run(snapshot_path, out))
    generic = phase(output, 'generic_shapes', lambda out: shape_builder.run(geometry, out))
    geo_value = read_sealed(generic)
    semantic_rows = {r['instance_id']: r for r in read_sealed(snapshot_path)['objects']}
    directions = []
    for row in geo_value['objects']:
        if not row['query_usable_for_sit'] or row['instance_id'] in blocked_spatial_ids:
            continue
        key = row['instance_id']
        try:
            object_views_path = projection_views(verified(semantic_rows[key]['evidence']))

            def review_direction(out):
                return direction.run(generic, key, out, object_views_path=object_views_path, qwen_client=qwen_client)
            try:
                review = phase(output, 'direction_' + key, review_direction)
            except Exception as evidence_error:
                object_views_path = phase(output, 'direction_views_' + key, lambda out: object_views.run(generic, [key], out, gpu), gpu=gpu)
                review = phase(output, 'direction_recovery_' + key, review_direction)
            directions.append(review)
        except Exception as error:
            errors.append({'instance_id': key, 'phase': 'direction_review', 'reason': f'{type(error).__name__}: {error}'})
    if directions:
        directed = phase(output, 'direction_geometry', lambda out: direction_revision.run(generic, directions, out))
    else:
        directed = generic
    published_geometry = output / 'fixed_geometry/receipt.json'
    if not published_geometry.exists():
        value = copy.deepcopy(read_sealed(directed))
        completed_ids = {read_sealed(p)['instance_id'] for p in directions}
        for row in value['objects']:
            if row['instance_id'] in blocked_spatial_ids:
                row.update(query_usable_for_sit=False, spatial_identity_recovery_required=True, geometry_publication_reason='disjoint_support_identity_not_resolved')
            if row['query_usable_for_sit'] and row['instance_id'] not in completed_ids:
                row.update(query_usable_for_sit=False, direction_recovery_required=True, direction_publication_reason='actual_qwen_direction_review_not_completed')
        value.update(source_geometry_before_final_publication=artifact(directed), scene_geometry_review_errors=errors, numeric_geometry_computed_by_qwen=False, every_published_sitting_direction_has_actual_qwen_candidate_review=True, source_code=artifact(__file__), scene_only_geometry_publication_completed=True)
        write_once(published_geometry, value, seal=True)
    report = write_report(output, snapshot_path, published_geometry)
    result_geometry = read_sealed(published_geometry)
    result = {'schema': 'p550.spatial_and_direction_reviewed_fixed_scene.v1', 'scene_id': scene['scene_id'], 'source_context_scene': artifact(scene_path), 'source_scene_receipt': scene['source_scene_receipt'], 'fixed_semantics': artifact(snapshot_path), 'fixed_geometry': artifact(published_geometry), 'report': artifact(report), 'component_scan': artifact(scan), 'component_recoveries': components, 'direction_reviews': [artifact(p) for p in directions], 'per_object_errors': errors, 'sit_usable_ids': [r['instance_id'] for r in result_geometry['objects'] if r['query_usable_for_sit']], 'status': 'scene_understanding_processed_with_explicit_unknowns', 'human_instruction_read': False, 'stage2_or_stage3_output_read': False, 'affordance_memory_used': False, 'qwen_computed_coordinates': False, 'source': artifact(__file__), 'elapsed_seconds': time.monotonic() - started}
    write_once(final, result, seal=True)
    print({'scene': scene['scene_id'], 'usable': result['sit_usable_ids'], 'errors': len(errors)}, flush=True)
    return result

def review_envelope(source_path, output, gpu, *, qwen_client):
    started = time.monotonic()
    source = read_sealed(source_path)
    if source['schema'] != 'p550.spatial_and_direction_reviewed_fixed_scene.v1' or source['human_instruction_read'] is not False or source['stage2_or_stage3_output_read'] is not False:
        raise ValueError('Use the scene-only spatial/direction publication')
    output.mkdir(parents=True, exist_ok=True)
    launch = output / 'launch.json'
    if launch.exists():
        value = read_sealed(launch)
        if value['source_geometry_scene'] != artifact(source_path) or value['gpu'] != gpu:
            raise ValueError('Envelope follower source drift')
        verify_artifact_tree(value)
    else:
        write_once(launch, {'schema': 'p550.final_scene_envelope_launch.v1', 'source_geometry_scene': artifact(source_path), 'gpu': gpu, 'source': artifact(__file__), 'human_instruction_read': False}, seal=True)
    final = output / 'receipt.json'
    if final.exists():
        value = read_sealed(final)
        verify_artifact_tree(value)
        return value
    scan = phase(output, 'envelope_scan', lambda out: component_builder.scan(verified(source['fixed_semantics']), out / 'receipt.json'))
    proposals = [r['instance_id'] for r in read_sealed(scan)['objects'] if r['candidate_proposed']]
    context = verified(source['source_context_scene'])
    if proposals:
        published = phase(output, 'envelope_republication', lambda out: review_geometry(context, out, gpu, qwen_client=qwen_client), gpu=gpu)
        result = dict(read_sealed(published))
    else:
        published = source_path
        result = dict(source)
    result.update(schema='p550.final_fixed_scene_understanding.v1', source_geometry_scene=artifact(source_path), final_reviewed_scene_receipt=artifact(published), envelope_scan=artifact(scan), new_envelope_proposal_ids=proposals, envelope_republication_performed=bool(proposals), task_or_motion_used_to_change_scene=False, previous_good_scene_recomputed_without_new_geometry_evidence=False, final_publication_source=artifact(__file__), elapsed_seconds_final_review=time.monotonic() - started)
    write_once(final, result, seal=True)
    print({'scene': source['scene_id'], 'extra_proposals': proposals, 'usable': result['sit_usable_ids']}, flush=True)
    return result

def conflicts(geometry):
    result = []
    for row in geometry['objects']:
        front = row.get('front_evidence', {})
        candidate = row.get('direction_hypotheses', {}).get('candidates', [{}])[0]
        if row.get('query_usable_for_sit') and candidate.get('source_review_candidate_id') == 'B' and (front.get('high_support_vertex_count', 0) >= 100) and ((front.get('backrest_offset_m') or 0) >= 0.1) and (front.get('prior_geometric_hypothesis', {}).get('method') == 'opposite_high_target_support_backrest_centroid'):
            result.append(row['instance_id'])
    return result

def review_backrest(source_path, output, gpu, *, qwen_client):
    started = time.monotonic()
    scene = read_sealed(source_path)
    if scene['schema'] != 'p550.final_fixed_scene_understanding.v1' or scene['human_instruction_read'] is not False:
        raise ValueError('Use an instruction-independent final scene snapshot')
    verify_artifact_tree(scene)
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'receipt.json'
    if final.exists():
        value = read_sealed(final)
        if value['source_before_backrest_review'] != artifact(source_path):
            raise ValueError('Backrest publication source changed')
        verify_artifact_tree(value)
        return value
    geometry_path = verified(scene['fixed_geometry'])
    geometry = read_sealed(geometry_path)
    candidates = conflicts(geometry)
    reviews = []
    errors = []
    for key in candidates:
        try:
            review = phase(output, 'backrest_' + key, lambda out: backrest.run(geometry_path, key, out, qwen_client=qwen_client))
            reviews.append(review)
        except Exception as error:
            errors.append({'instance_id': key, 'phase': 'semantic_backrest', 'reason': f'{type(error).__name__}: {error}'})
    directed = geometry_path
    if reviews:
        directed = phase(output, 'backrest_geometry', lambda out: direction_revision.run(geometry_path, reviews, out))
    fixed = copy.deepcopy(read_sealed(directed))
    for row in fixed['objects']:
        if row['instance_id'] in {r['instance_id'] for r in errors}:
            row.update(query_usable_for_sit=False, direction_recovery_required=True, direction_publication_reason='opposed_backrest_geometry_and_arrow_not_resolved')
    fixed.update(source_geometry_before_semantic_backrest_review=artifact(geometry_path), semantic_backrest_conflict_ids=candidates, source_code=artifact(__file__))
    path = output / 'fixed_geometry/receipt.json'
    write_once(path, fixed, seal=True)
    result = dict(scene)
    result.update(source_before_backrest_review=artifact(source_path), fixed_geometry=artifact(path), semantic_backrest_reviews=[artifact(p) for p in reviews], semantic_backrest_conflict_ids=candidates, semantic_backrest_review_complete=True, sit_usable_ids=[r['instance_id'] for r in fixed['objects'] if r.get('query_usable_for_sit')], per_object_errors=[*scene['per_object_errors'], *errors], report=artifact(write_report(output, verified(scene['fixed_semantics']), path)), final_publication_source=artifact(__file__), elapsed_seconds_backrest_review=time.monotonic() - started, human_instruction_read=False, stage2_or_stage3_output_read=False)
    write_once(final, result, seal=True)
    print({'scene': scene['scene_id'], 'conflicts': candidates, 'usable': result['sit_usable_ids'], 'errors': errors}, flush=True)
    return result
