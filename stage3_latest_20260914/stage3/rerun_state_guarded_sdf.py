"""Three fresh state-guarded SDF rollouts; explicitly reuse prior OFF controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[1]
ENTRY = HERE/'unihsi_probe/run_stage12_transfer_v3.py'
ENTRY_SHA = '8b8740004a7e1fff67abb64bcaeb4e7b4c251af4ef48b6cf483ae9d2b151c94f'


def binding(path):
    return dict(path=str(path.resolve(strict=True)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Explicit --execute required; fixed isolated pilot output')
    if binding(ENTRY)['sha256'] != ENTRY_SHA:
        raise ValueError('Corrected source changed')
    output = AGENT/'runs/paper_structure_stage3_20260913/unihsi_stage12_state_guarded03'
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='5', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               LD_LIBRARY_PATH='/home/lzsh2025/miniconda3/lib')
    started = time.monotonic()
    records = []
    for scene in ('scene006', 'scene027', 'scene037'):
        prior = AGENT/'runs/paper_structure_stage3_20260913/unihsi_stage12_batch02'/scene/'actor/execution.json'
        off = json.loads(prior.read_text())
        assert off['variant'] == 'trained_actor' and off['seed'] == 390919 and off['sdf_guidance_added'] is False
        folder = output/scene/'actor_sdf'
        folder.parent.mkdir()
        command = [sys.executable, str(ENTRY), '--packet', str(AGENT/'runs/paper_structure_stage3_20260913/stage12_packets'/scene/'manifest.json'),
                   '--output', str(folder), '--frames', '900', '--seed', '390919', '--gpu', '0', '--sdf-filter', '--execute']
        print(json.dumps(dict(event='start', scene=scene, new_rollout='SDF_ON_only', reused_OFF=str(prior))), flush=True)
        begun = time.monotonic()
        with (folder.parent/'actor_sdf.process.log').open('x') as stream:
            proc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError('Fresh SDF rollout failed; preserve log, do not substitute old ON')
        on = folder/'execution.json'
        record = dict(scene=scene, off=binding(prior), on=binding(on), off_reused_not_new=True,
                      new_on=True, command=command, elapsed_seconds=time.monotonic()-begun,
                      paired_initial_state_verification_required=True)
        records.append(record)
        with (folder.parent/'pair_manifest.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(dict(event='finish', **record)), flush=True)
    with (output/'batch_execution.json').open('x') as stream:
        json.dump(dict(pairs=records, new_rollouts=3, reused_off_controls=3,
                       elapsed_seconds=time.monotonic()-started, physical_gpu=5,
                       runner=binding(ENTRY), full_stage3_contract_satisfied=False), stream, indent=2)


if __name__ == '__main__':
    main()
