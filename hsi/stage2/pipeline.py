"""Current Stage2: fresh views -> H3 image -> HybrIK -> refine -> strict release.

Models and EGL own separate child processes. Qwen supplies semantics only.
Failed evidence is retained and no failure receipt authorizes Stage3.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time
from ._common import artifact, read, read_sealed, verified, write_once, require, source_closure, verify_artifact_tree

SCHEMA = 'hsi.stage2.verified_keypose.v1'


def _worker(output, operation, arguments):
    started = time.monotonic()
    values = {key: str(value) if isinstance(value, Path) else value for key, value in arguments.items()}
    path = output / 'workers' / (operation + '_arguments.json')
    write_once(path, {'operation': operation, 'arguments': values})
    log = output / 'workers' / (operation + '.log')
    command = [sys.executable, '-m', 'hsi.stage2.worker', '--operation', operation, '--arguments', str(path)]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMPY_MADVISE_HUGEPAGE='0')
    with log.open('xb') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    record = {'operation': operation, 'arguments': artifact(path), 'exit_code': completed.returncode,
              'log': artifact(log), 'elapsed_seconds': time.monotonic()-started}
    write_once(output / 'workers' / (operation + '_receipt.json'), record)
    require(completed.returncode == 0, operation + ' failed; see ' + str(log))
    return record


def _target_description(stage1_path, view_path, output, qwen_client):
    from .views import validate_receipt
    stage1 = read_sealed(stage1_path)
    selected = validate_receipt(view_path, stage1_path)['selection']
    properties = {'target_visual_identity': {'type': 'string', 'minLength': 12, 'maxLength': 280},
                  'needs_more_evidence': {'type': 'boolean'}}
    prompt = ('IMAGE_0 is the fixed scene crop. IMAGE_1 marks the exact target furniture in AMBER and its seat surface in GREEN. '
        'Describe this exact physical furniture in English so a generator can identify it in unmarked IMAGE_0. '
        'Use shape and nearby scene context, no annotation colors or coordinates. Do not infer a platform or extra seat not visible. '
        'Only describe the target, no person. If the identity is ambiguous, needs_more_evidence=true. Keep the description concise.')
    response = qwen_client.session(output / 'qwen_calls', max_tokens=170).call(
        [verified(selected['selected_stage2_crop']), verified(selected['selected_stage2_crop_target_overlay'])], prompt,
        schema={'type': 'object', 'additionalProperties': False, 'properties': properties, 'required': list(properties)},
        max_tokens=170, call_id='p550_view_bound_target_identity')
    semantic = json.loads(response['raw_completion'])
    path = output / 'receipt.json'
    write_once(path, {'schema': 'p550.view_bound_target_description.v1', 'target': stage1['target'],
        'view': artifact(view_path), 'semantic': semantic, 'generation': response})
    return path


def _evidence(value):
    from . import image_contract, image_job, recovery_contract, image_review, review_provenance, posture_quality
    from .direction import validate_direction
    from .views import validate_receipt
    from .physical_validation import validate as validate_physics
    import numpy as np
    require(value['schema'] == SCHEMA and value['source_closure'] == source_closure(), 'Stage2 source/schema drift')
    stage1_path = verified(value['source_stage1'])
    stage1 = read_sealed(stage1_path)
    target = read_sealed(verified(stage1['target']))
    view_path = verified(value['source_view'])
    view = validate_receipt(view_path, stage1_path)
    job = image_job.validate_job(verified(value['image_job']))
    require(job['stage1_execution'] == str(stage1_path), 'Image job belongs to another Stage1')
    h3 = image_contract.validate_receipt(verified(value['h3_receipt']), expected_case_id=job['case_id'], expected_image=verified(value['h3_image']))
    require(h3['inputs']['camera'] == view['selection']['selected_camera'], 'H3 camera changed')
    trace = read_sealed(verified(h3['sampler_trace']))
    from . import h3 as h3_runner
    require(trace['sampler_source'] == artifact(h3_runner.__file__)
            and trace['pipeline_class'] == 'hsi.stage2.h3.ComfyUINativeSingleImage'
            and trace['latent_shape'] == list(h3_runner.VISUAL_SHAPE)
            and trace['auxiliary_audio_latent_shape'] == list(h3_runner.AUDIO_SHAPE)
            and trace['decoded_tensor_shape'] == list(h3_runner.DECODED_SHAPE)
            and trace['steps'] == 8 and trace['sampler'] == 'er_sde' and trace['scheduler'] == 'sgm_uniform'
            and trace['modality_sigma_shifts'] == {'visual': 12., 'auxiliary_audio': 3.}
            and trace['adapter_strengths'] == {'distill_lora': .75, 'image_lora': .5},
            'H3 trace does not bind the actual current single-image runner and tensor policy')
    verify_artifact_tree(trace)
    recovery_path = verified(value['recovery_receipt'])
    recovery = read_sealed(recovery_path)
    recovery_contract.validate_h3_hybrikx_prior(verified(recovery['outputs']['articulation_prior']), recovery_path,
        scene_id=stage1['scene_id'], instruction=stage1['instruction'], target_instance_id=target['target_instance_id'],
        target_surface_id=target['selected_surface_id'])
    verify_artifact_tree(recovery)
    require(recovery['inputs']['h3_generation_receipt'] == value['h3_receipt']
            and recovery['inputs']['stage1_artifact'] == stage1['bundle'], 'Recovery lineage differs')
    require(validate_direction(stage1_path, verified(value['direction_validation']['source_direction_review']))
            == value['direction_validation'], 'Current direction proof differs')
    original_path = verified(value['h3_semantic_audit'])
    review_provenance.validate(original_path, 'h3')
    original = read_sealed(original_path)
    require(original['inputs']['generated_keypose'] == value['h3_image']
            and original['inputs']['reference'] == view['selection']['selected_stage2_crop']
            and original['inputs']['marked_reference'] == view['selection']['selected_stage2_crop_target_overlay'], 'H3 review evidence differs')
    partial = not original['decision']['semantic_prior_eligible_for_refine']
    require(not partial or image_review.partial_prior_eligible(original), 'Non-visibility semantic failure cannot be recovered')
    require(partial == value['explicit_visibility_only_recovery'], 'Visibility recovery declaration differs')
    background = read_sealed(verified(value['background_consistency']))
    from .background_numeric import measure, POLICY
    require(background['source_h3'] == value['h3_receipt'] and background['source_recovery'] == value['recovery_receipt']
            and background['source_view'] == value['source_view'] and background['policy'] == POLICY, 'Background source/policy differs')
    with np.load(verified(recovery['outputs']['raw_hybrikx_recovery']), allow_pickle=False) as raw:
        box = raw['selected_person_bbox_xyxy']
    actual, _, _ = measure(verified(view['selection']['selected_stage2_crop']), verified(value['h3_image']), box)
    require(all(background[key] == actual[key] for key in ('gates', 'metrics', 'passed')) and actual['passed'], 'Fresh background check failed')
    physical_path = verified(value['physical_refine'])
    physical = read_sealed(physical_path)
    require(physical['source_stage1'] == value['source_stage1'] and physical['source_recovery'] == value['recovery_receipt']
            and physical['outputs']['published_keypose'] == value['keypose'], 'Physical candidate lineage differs')
    validate_physics(physical_path)
    quality = read_sealed(verified(value['neutral_sitting_quality']))
    with np.load(verified(value['keypose']), allow_pickle=False) as data:
        recomputed = posture_quality.metrics(data['joints_world_zup'], float(data['terminal_facing_yaw_rad']))
    require(quality['bundle'] == value['physical_refine'] and quality['keypose'] == value['keypose']
            and quality['quality'] == recomputed and recomputed['all_neutral_sit_quality_gates_passed'] is True,
            'Additional four neutral-sit checks failed')
    post_path = verified(value['post_refine_qwen'])
    review_provenance.validate(post_path, 'refined')
    post = read_sealed(post_path)
    require(post['source_refine_bundle'] == value['physical_refine'] and post['source_stage2'] == value['candidate_execution']
            and post['source_rendered_mesh_manifest'] == value['final_mesh_visualization'], 'Final visual review lineage differs')
    visual = read_sealed(verified(value['final_mesh_visualization']))
    verify_artifact_tree(visual)
    require(visual['source'] == artifact(Path(__file__).with_name('render_evidence.py'))
            and visual['inputs']['p533_complete_bundle'] == value['physical_refine']
            and visual['inputs']['stage1_target'] == stage1['target']
            and visual['inputs']['stage1_bundle'] == stage1['bundle'], 'Final mesh render source changed')
    isolated = read_sealed(verified(visual['isolated_body_diagnostic']))
    verify_artifact_tree(isolated)
    expected_images = [view['selection']['selected_stage2_crop'], view['selection']['selected_stage2_crop_target_overlay'],
        value['h3_image'], visual['views'][0]['image'], *[row['image'] for row in isolated['views']]]
    require(post['actual_input_images'] == expected_images and len(expected_images) == 6,
            'Final Qwen images do not represent exact current body/camera evidence')
    require(visual['inputs']['p533_published_keypose'] == isolated['source_keypose'] == value['keypose']
            and isolated['actual_body_world_vertices_or_pose_modified'] is False
            and visual['actual_body_or_scene_world_geometry_modified'] is False,
            'Final mesh visualization changed actual body geometry')
    require(post['decision']['post_refine_semantic_pass'] is True, 'Final actual-Qwen nine-check review failed')
    require(value['all_original_physical_gate_count'] == 18 and value['additional_neutral_sit_gate_count'] == 4,
            'Publication gate set changed')


def validate_success(receipt_path):
    """Strict Stage3 entry: rerun source, actual-Qwen and full-mesh physical checks."""
    value = read_sealed(receipt_path)
    require(value.get('verified_stage2_keypose') is True and value.get('status') == 'complete_verified_stage2_keypose',
            'Stage2 has not published a verified keypose')
    _evidence(value)
    return value


def run(stage1_path, render_path, descriptions_path, output, *, qwen_client, comfy_root, models_root,
        recovery_runtime, segmentation, gpu=0, seed=55000, mode='full', facts_path=None, store_root=None):
    from . import image_job, image_contract, image_review, background, posture_review, mesh_review, projection
    from .direction import validate_direction
    started = time.monotonic()
    stage1_path, render_path, descriptions_path = (Path(p).resolve(strict=True) for p in (stage1_path, render_path, descriptions_path))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage1 = read_sealed(stage1_path)
    descriptions = read_sealed(descriptions_path)
    require(descriptions['source_stage1'] == artifact(stage1_path) and descriptions['all_keypoints_have_descriptions'] is True,
            'Current Stage1 keypoint descriptions required')
    geometry = read_sealed(verified(stage1['source_fixed_geometry']))
    target = read_sealed(verified(stage1['target']))
    row = next(item for item in geometry['objects'] if item['instance_id'] == target['target_instance_id'])
    direction = validate_direction(stage1_path, verified(row['direction_review']))
    runtime = {key: str(val) if isinstance(val, Path) else [str(p) for p in val] if key == 'extra_python_paths' else val
               for key, val in dict(recovery_runtime).items()}
    value = {'schema': SCHEMA, 'source_closure': source_closure(), 'source_stage1': artifact(stage1_path),
        'source_render': artifact(render_path), 'keypoint_descriptions': artifact(descriptions_path),
        'direction_validation': direction, 'verified_stage2_keypose': False, 'stage3_handoff_allowed': False,
        'actual_stage3_motion_generated': False, 'qwen_generated_coordinates': False, 'seed': seed, 'timings': {}}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            view_job = pool.submit(_worker, output, 'views', dict(stage1_path=stage1_path, render_path=render_path,
                output=output/'views', gpu=gpu, mode=mode, facts_path=facts_path, store_root=store_root))
            sdf_job = pool.submit(_worker, output, 'sdf', dict(stage1_path=stage1_path, output=output/'sdf'))
            value['timings']['views'] = view_job.result()['elapsed_seconds']
            value['timings']['sdf'] = sdf_job.result()['elapsed_seconds']
        view_path, sdf_path = output/'views/receipt.json', output/'sdf/receipt.json'
        value.update(source_view=artifact(view_path), source_sdf=artifact(sdf_path))
        description_path = _target_description(stage1_path, view_path, output/'target_description', qwen_client)
        job = image_job.run(stage1_path=stage1_path, view_path=view_path, description_path=description_path,
            output=output/'image_job', case_id=output.name, seed=seed)
        job_path = output/'image_job/single_image_job.json'
        value['image_job'] = artifact(job_path)
        value['timings']['h3'] = _worker(output, 'h3', dict(job_path=job_path, output=output/'h3',
            comfy_root=comfy_root, models_root=models_root, gpu=gpu))['elapsed_seconds']
        h3_path = output/'h3/receipt.json'
        h3 = image_contract.validate_receipt(h3_path)
        image = verified(h3['artifacts']['h3_image'])
        value.update(h3_receipt=artifact(h3_path), h3_image=artifact(image))
        config = {'case_id': job['case_id'], 'scene_id': stage1['scene_id'], 'instruction': stage1['instruction'],
            'action_family': 'sit', 'stage1_binding': {'target_instance_id': target['target_instance_id'],
                'target_surface_id': target['selected_surface_id']}}
        recovery_config = output/'recovery_job.json'
        write_once(recovery_config, config)
        selected = read_sealed(view_path)['selection']
        with ThreadPoolExecutor(max_workers=2) as pool:
            recovery_task = pool.submit(_worker, output, 'recovery', dict(image=image, h3_receipt=h3_path,
                camera=Path(job['camera']), stage1_bundle=verified(stage1['bundle']), config=recovery_config,
                output=output/'recovery', runtime=runtime, gpu=gpu, case_id=job['case_id']))
            semantic_task = pool.submit(image_review.audit, verified(selected['selected_stage2_crop']),
                verified(selected['selected_stage2_crop_target_overlay']), image, stage1['instruction'],
                target['target_instance_id'], output/'image_review/receipt.json', 'current_new_scene', qwen_client=qwen_client)
            value['timings']['recovery'] = recovery_task.result()['elapsed_seconds']
            audit = semantic_task.result()
        recovery_path = output/'recovery/receipt.json'
        value.update(recovery_receipt=artifact(recovery_path), h3_semantic_audit=artifact(output/'image_review/receipt.json'))
        result_background = background.run(stage1_path, view_path, h3_path, recovery_path, output/'background')
        value['background_consistency'] = artifact(output/'background/receipt.json')
        require(result_background['passed'] is True, 'H3 changed the fixed-camera background')
        partial = not audit['decision']['semantic_prior_eligible_for_refine']
        require(not partial or image_review.partial_prior_eligible(audit), 'H3 semantic failure cannot enter visibility-only recovery')
        value['explicit_visibility_only_recovery'] = partial
        render = read_sealed(render_path)
        depth = verified(render['views'][selected['selected_view_index']]['depth'])
        projection.run(Path(job['camera']), Path(job['crop_derivation']), depth, image, output/'projection')
        projection_path = output/'projection/projection_observation.json'
        value['timings']['refine'] = _worker(output, 'placement', dict(stage1_path=stage1_path, recovery_path=recovery_path,
            projection_path=projection_path, sdf_path=sdf_path, output=output/'physical', smplx_model=runtime['neutral_smplx'],
            segmentation=segmentation, steps=180))['elapsed_seconds']
        physical_path = output/'physical/receipt.json'
        physical = read_sealed(physical_path)
        value['physical_refine'] = value['refine_bundle'] = artifact(physical_path)
        require(physical['stage3_handoff_allowed'] is True, 'Refine failed one or more full-mesh physical gates')
        value['keypose'] = physical['outputs']['published_keypose']
        candidate_path = output/'candidate_execution.json'
        write_once(candidate_path, {'schema': 'hsi.stage2.unpublished_physical_candidate_execution.v1',
            'source_stage1': value['source_stage1'], 'source_view': value['source_view'], 'hybrikx_receipt': value['recovery_receipt'],
            'h3_semantic_audit': value['h3_semantic_audit'], 'h3_receipt': value['h3_receipt'],
            'physical_refine': value['physical_refine'], 'verified_stage2_keypose': False, 'stage3_handoff_allowed': False})
        value['candidate_execution'] = artifact(candidate_path)
        quality_path = output/'neutral_sitting_quality.json'
        posture_review.run(physical_path, quality_path)
        value['neutral_sitting_quality'] = artifact(quality_path)
        require(read_sealed(quality_path)['quality']['all_neutral_sit_quality_gates_passed'] is True, 'Neutral-sit quality failed')
        value['timings']['render'] = _worker(output, 'render', dict(source_path=candidate_path,
            bundle_path=physical_path, output=output/'mesh_evidence', gpu=gpu))['elapsed_seconds']
        manifest = output/'mesh_evidence/manifest.json'
        mesh_review.run(candidate_path, physical_path, manifest, output/'mesh_evidence/isolated_body/receipt.json',
            output/'post_refine_qwen', qwen_client=qwen_client)
        value.update(post_refine_qwen=artifact(output/'post_refine_qwen/receipt.json'), final_mesh_visualization=artifact(manifest),
            all_original_physical_gate_count=18, additional_neutral_sit_gate_count=4)
        _evidence(value)
        value.update(status='complete_verified_stage2_keypose', verified_stage2_keypose=True, stage3_handoff_allowed=True)
    except Exception as error:
        value.update(status='failed_not_publishable', reason=f'{type(error).__name__}: {error}',
            verified_stage2_keypose=False, stage3_handoff_allowed=False)
    value['elapsed_seconds'] = time.monotonic()-started
    return write_once(output/'receipt.json', value)
