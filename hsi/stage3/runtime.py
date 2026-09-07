"""One explicit current ReMoGen run; no server, queue, retry or motion editing."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import time
from hsi.common.artifacts import artifact, read_json, read_sealed, require, verified, write_once
from .model_runtime import MotionRuntime, configure, CHECKPOINT_HASHES

SCHEMA = 'hsi.stage3.execution.v1'


def load_inputs(adapter_path):
    from .compiler import load_adapter
    from .seed import load_seed
    path = Path(adapter_path).resolve(strict=True)
    adapter, packet, seed_receipt = load_adapter(path)
    seed = load_seed(verified(adapter['static_seed_receipt']))
    return {'adapter_path': path, 'adapter': adapter, 'packet': packet,
        'packet_path': verified(adapter['packet']), 'seed_path': verified(adapter['static_seed']),
        'seed_receipt': seed_receipt, 'seed_receipt_path': verified(adapter['static_seed_receipt']),
        'seed': seed, 'occupancy': verified(adapter['inputs']['raw_occupancy'])}


def generation_arguments(inputs, output, seed, runtime, *, dry_run=False):
    """Current numerical defaults, copied without tuning from the active policy."""
    require(type(seed) is int and 0 <= seed < 2**32, 'Invalid generation seed')
    values = {
        'terminal-rolewise-hold-frames': 6, 'terminal-rolewise-maximum-extra-primitives': 12,
        'terminal-rolewise-contact-max-m': .08, 'terminal-rolewise-support-chain-max-m': .20,
        'terminal-rolewise-axial-posture-max-m': .16, 'terminal-rolewise-upper-limb-max-m': .20,
        'terminal-rolewise-other-sparse-max-m': .20, 'terminal-rolewise-global-per-joint-max-m': .20,
        'terminal-e226-mode': 'off', 'bridge-config': Path(__file__).with_name('causal_bridge_config.json'),
        'query-static-seed': inputs['seed_path'], 'plan': inputs['packet_path'],
        'raw-occupancy': inputs['occupancy'], 'occupancy-layout': 'auto', 'sdf-downsample': 2,
        'ablation': 'M4_KEYPOSE_SDF_E226_ICGF', 'device': 'cuda', 'seed': seed,
        'guidance-param': 7., 'maximum-nodes': 8, 'minimum-primitives': 0, 'max-primitives': 40,
        'post-completion-primitives': 0, 'max-attempts-per-node': 0, 'root-xy-threshold': .10,
        'root-z-threshold': .05, 'yaw-threshold-deg': 10, 'joint-threshold': .20,
        'sdf-weight': 1., 'icgf-weight': .40, 'contact-frames': 4,
        'scene-guidance-scale': 1.5, 'scene-guidance-mode': 'xstart_posterior',
        'clean-latent-step-size': .1, 'max-clean-latent-update-norm': .15, 'denoise-start-fraction': .4,
        'icgf-source': 'sparse_joint_targets', 'icgf-active-node-policy': 'terminal', 'icgf-contact-joints': '0,1,2',
        'icgf-guidance-joints': ','.join(map(str, range(22))), 'sdf-exempt-joints': '0,1,2',
        'collision-joint-ids': ','.join(map(str, range(22))), 'output-dir': output,
        'adapter-checkpoint': runtime.adapter_checkpoint, 'base-checkpoint': runtime.base_checkpoint,
        'mvae-checkpoint': runtime.mvae_checkpoint, 'data-dir': runtime.statistics_directory,
        'scene-root': runtime.scene_directory, 'cfg-path': runtime.primitive_config, 'split': 'val',
    }
    argv = [item for k, v in values.items() for item in ('--'+k, str(v))]
    argv += ['--terminal-rolewise-feedback', '--normalize-scene-gradient',
        '--include-target-component-in-sdf', '--exempt-contact-sdf-tail']
    if dry_run:
        argv.append('--dry-run')
    return argv


@contextmanager
def _bound_clip(record):
    """Use the declared installed checkpoint; never let CLIP download a default."""
    import clip
    original = clip.load
    calls = []
    def load(name, *args, **kwargs):
        require(name == 'ViT-B/32', 'Unregistered text encoder requested')
        path = verified(record)
        result = original(str(path), *args, **kwargs)
        calls.append({'requested_model': name, 'actual_checkpoint': artifact(path)})
        return result
    clip.load = load
    try:
        yield calls
    finally:
        clip.load = original


def runtime_proof(output, *, dry_run):
    from .full22 import validate_runtime_output
    metadata = read_json(output / ('dry_run_receipt.json' if dry_run else 'generation_metadata.json'))
    rows = None if dry_run else [read_json_line(line) for line in (output/'energy_log.jsonl').read_text().splitlines() if line.strip()]
    proof = validate_runtime_output(metadata, dry_run=dry_run, energy_log_records=rows,
        guidance_joint_ids=tuple(range(22)), semantic_contact_joint_ids=(0,1,2),
        sdf_exempt_joint_ids=(0,1,2), contact_frames=4)
    return proof, metadata, rows


def read_json_line(line):
    import json
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, 'Duplicate energy-log key')
            value[key] = item
        return value
    return json.loads(line, object_pairs_hook=pairs,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite energy log: '+v)))


def _network_postflight(records, directory, *, clip_calls):
    for row in records.values():
        verified(row)
    manifest = read_json(directory/'run_manifest.json')
    actual = manifest['input_provenance']['artifacts']
    for role in ('adapter_checkpoint','base_checkpoint','mvae_checkpoint'):
        require(actual[role]['path'] == records[role]['path'] and actual[role]['sha256'] == records[role]['sha256']
            and actual[role]['exists'] is True, 'Actual runtime checkpoint differs from preflight')
    frozen = {row['path'] for row in records.values()}
    source_roots = {str(Path(row['path']).parent) for key,row in records.items() if key.startswith(('source:','clip_source:'))}
    loaded = []
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and str(path.parent) in source_roots:
                require(str(path) in frozen, 'Loaded numerical source not bound before generation')
                loaded.append(str(path))
    require(clip_calls and all(row['actual_checkpoint'] == records['clip_checkpoint'] for row in clip_calls),
        'Actual CLIP checkpoint use was not observed')
    return {'actual_checkpoint_paths_and_hashes_verified': True, 'actual_imported_sources': sorted(set(loaded)),
        'clip_calls': clip_calls, 'runtime_manifest': artifact(directory/'run_manifest.json')}


def run(adapter_path, output, *, runtime: MotionRuntime, seed=55000, affordance=True, dry_run=False):
    from . import compiler, executor, full22, physics_runtime
    from .physics import CurrentStepwisePhysicsConfig
    from .guidance import GuidanceBinding, J22ProxyMap
    from .runtime_hooks import admitted_provider, expose_affordance, validate_affordance_runtime
    require(type(affordance) is bool and type(dry_run) is bool, 'Explicit runtime modes required')
    inputs = load_inputs(adapter_path)
    configure(runtime)
    assets = runtime.bind()
    implementation = compiler.sources()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    directory = output/'runtime'
    argv = generation_arguments(inputs, directory, seed, runtime, dry_run=dry_run)
    config = CurrentStepwisePhysicsConfig(slip_weight=4., contact_band_weight=1.,
        floor_penetration_weight=8., lower_body_collision_weight=2., root_speed_release_mps=.85).validate()
    binding = GuidanceBinding(verified(inputs['adapter']['guidance_input']))
    proxy = J22ProxyMap(**{k: tuple(v) if isinstance(v,list) else v for k,v in inputs['adapter']['guidance_proxy_map'].items()})
    launch = {'schema': 'hsi.stage3.launch.v1', 'adapter': artifact(adapter_path),
        'network_assets': assets, 'checkpoint_hash_policy': CHECKPOINT_HASHES, 'implementation_sources': implementation,
        'actual_runner_argv': argv, 'runtime_paths': {k: str(v) for k,v in vars(runtime).items()},
        'affordance': affordance, 'dry_run': dry_run, 'seed': seed,
        'parent_physics': config.receipt(), 'motion_release_authorized': False, 'positive_credit': 0}
    write_once(output/'launch.json', launch)
    status, error, proof, stats, added_proof, postflight = None, None, None, None, None, None
    started, before = time.monotonic(), Path.cwd()
    try:
        with _bound_clip(assets['clip_checkpoint']) as clip_calls, admitted_provider(executor, inputs), \
            physics_runtime.expose_decoupled_joint_fields(executor), \
            physics_runtime.expose_adaptive_guidance_policy(executor), \
            physics_runtime.expose_p533_current_stepwise_physics_guidance(executor, config), \
            full22.expose_full22_sparse_icgf(executor, guidance_joint_ids=tuple(range(22)),
                semantic_contact_joint_ids=(0,1,2), sdf_exempt_joint_ids=(0,1,2), contact_frames=4), \
            expose_affordance(executor, binding=binding, proxy=proxy, physics_module=physics_runtime,
                enabled=affordance, proxy_receipt_path=verified(inputs['adapter']['guidance_proxy_receipt'])) as stats:
            status = int(executor.main(argv))
        require(status == 0, 'Motion runner failed')
        proof, metadata, rows = runtime_proof(directory, dry_run=dry_run)
        if not dry_run:
            require((directory/'generated_motion.npz').is_file(), 'No generated motion')
            added_proof = validate_affordance_runtime(metadata, rows, stats, inputs['packet'], enabled=affordance)
            postflight = _network_postflight(assets, directory, clip_calls=clip_calls)
        binding.verify_sources()
        for row in [artifact(adapter_path), *assets.values(), *implementation.values()]:
            verified(row)
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        raise
    finally:
        os.chdir(before)
        result = {'schema': SCHEMA, 'status': 'failed_not_publishable' if error else
            'dry_run_completed_not_motion' if dry_run else 'generated_pending_fullmesh_evaluation',
            'launch': artifact(output/'launch.json'), 'adapter': launch['adapter'], 'runner_returncode': status,
            'error': error, 'dry_run': dry_run, 'outputs': {name: artifact(directory/name) for name in
                ('generated_motion.npz','generation_metadata.json','run_manifest.json','energy_log.jsonl','dry_run_receipt.json')
                if (directory/name).is_file()}, 'full22_runtime_proof': proof, 'affordance_runtime': stats,
            'affordance_runtime_proof': added_proof, 'network_postflight': postflight,
            'elapsed_seconds': time.monotonic()-started, 'motion_publishable': False,
            'full_motion_evaluation_performed': False, 'positive_credit': 0}
        write_once(output/'receipt.json', result)
    return read_sealed(output/'receipt.json')


def validate_execution(path):
    from . import compiler
    from .runtime_hooks import validate_affordance_runtime
    value = read_sealed(path)
    require(value['schema'] == SCHEMA and value['status'] == 'generated_pending_fullmesh_evaluation'
        and value['runner_returncode'] == 0 and value['error'] is None and value['dry_run'] is False,
        'No completed current motion generation')
    launch = read_sealed(verified(value['launch']))
    require(launch['adapter'] == value['adapter'] and launch['implementation_sources'] == compiler.sources()
        and launch['dry_run'] is False and launch['checkpoint_hash_policy'] == CHECKPOINT_HASHES,
        'Generation source policy differs')
    inputs = load_inputs(verified(value['adapter']))
    runtime = MotionRuntime.from_mapping(launch['runtime_paths'])
    require(runtime.bind() == launch['network_assets'] and generation_arguments(inputs,
        verified(value['outputs']['generated_motion.npz']).parent, launch['seed'], runtime) == launch['actual_runner_argv'],
        'Generation model assets or numerical arguments differ')
    for row in value['outputs'].values():
        verified(row)
    directory = verified(value['outputs']['generation_metadata.json']).parent
    proof, metadata, rows = runtime_proof(directory, dry_run=False)
    require(proof == value['full22_runtime_proof'] and validate_affordance_runtime(metadata, rows,
        value['affordance_runtime'], inputs['packet'], enabled=launch['affordance']) == value['affordance_runtime_proof'],
        'Current actual denoising proof differs')
    post = value['network_postflight']
    require(post is not None and post['runtime_manifest'] == value['outputs']['run_manifest.json']
        and post['actual_checkpoint_paths_and_hashes_verified'] is True and post['clip_calls']
        and all(row['actual_checkpoint'] == launch['network_assets']['clip_checkpoint'] for row in post['clip_calls']),
        'Missing exact actual model postflight')
    manifest = read_json(verified(post['runtime_manifest']))
    for role in ('adapter_checkpoint','base_checkpoint','mvae_checkpoint'):
        actual = manifest['input_provenance']['artifacts'][role]
        expected = launch['network_assets'][role]
        require(actual['path'] == expected['path'] and actual['sha256'] == expected['sha256'] and actual['exists'] is True,
            'Runtime used another checkpoint')
    return {'receipt': value, 'packet_path': inputs['packet_path'], 'metadata_path': directory/'generation_metadata.json',
        'motion_path': verified(value['outputs']['generated_motion.npz']), 'stage2_path': verified(inputs['adapter']['inputs']['stage2']),
        'occupancy_path': inputs['occupancy'], 'body_model_record': inputs['adapter']['inputs']['body_model'],
        'runner_returncode': value['runner_returncode'], 'adapter': inputs['adapter'], 'packet': inputs['packet']}
