"""Complete static fresh mask/depth selector; camera memory is proposal-only.

Ordinary package imports replace all historical AST/exec/module rewriting.
The full 48-camera inventory and the numerical crop, rendering and ranking
operators are retained. This module grants image-generation handoff only,
never a publishable human pose.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from hsi.common import artifacts
from hsi.common.artifacts import artifact, digest, read_json as read, read_sealed, verified, write_once, require
from hsi.stage1 import binding
from hsi.stage1.binding import bind, checked
from . import _view_config as view_config, view_geometry, view_numeric
from .view_numeric import POLICY, area_samples, body_probes, project_probe, mask_visibility, board

SCHEMA = "hsi.stage2.fresh_geometry_view_selection.v1"
SOURCE = Path(__file__).resolve()
seal_source = verified


def choose_audit_candidates(campaign, selected_ids):
    if selected_ids is None:
        return campaign.candidates
    require(bool(selected_ids), "Empty hint subset")
    chosen = tuple(c for c in campaign.candidates if c.candidate_id in selected_ids)
    require(len(chosen) == len(set(selected_ids)), "Hint absent from calibrated campaign")
    return chosen


def contact_geometry_audit(campaign, candidate, approach):
    selected = next(r for r in campaign.target["candidate_surfaces"] if r["candidate_id"] == campaign.target["selected_surface_id"])
    cosine = view_numeric.facing_cosine(selected["surface"]["centre_world_xyz_zup_m"], approach["contact_forward_yaw_rad"], candidate.camera.position_world_xy)
    return view_numeric.soft_contact_gate(view_numeric.replace_arrival_gate, view_geometry.geometry_audit(campaign, candidate), cosine, float(campaign.geometry["minimum_approach_side_cosine"]))

def _run_attempt(stage1_path, render_path, output, selected_ids=None):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    stage1 = read(stage1_path)
    approach = bind(stage1)
    target = checked(stage1["target"])
    write_once(output / "route_selected_approach.json", approach)
    builder = view_config
    config_path = output / "config.json"
    builder.build_config(seal_source(stage1["target"]), render_path, output / "route_selected_approach.json", config_path, "automatic-only")
    old = view_geometry
    campaign = old.load_campaign(config_path)
    render = read(render_path)
    loader = view_numeric
    mesh_path = seal_source(target["artifacts"]["original_scene_mesh"])
    mesh = loader.load_scene_mesh(mesh_path)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    archive_path = seal_source(target["artifacts"]["target_surface_faces"])
    with np.load(archive_path, allow_pickle=False) as archive:
        if len(archive["mesh_face_mask"]) != len(faces) or str(archive["target_instance_id"]) != target["target_instance_id"]:
            raise ValueError("Target face archive is not bound to the original mesh")
        surface_faces = faces[np.asarray(archive["mesh_face_mask"], bool)]
        owner_ids = np.asarray(archive["target_support_vertex_ids"], int)
    owned = np.zeros(len(vertices), bool)
    owned[owner_ids] = True
    owner_faces = faces[owned[faces].sum(1) >= 2]
    if not len(owner_faces): raise ValueError("No directly supported owner faces")
    points = area_samples(vertices, surface_faces, POLICY["surface_area_samples"], POLICY["sample_seed"])
    probes = body_probes(target, approach)
    import pyrender
    import trimesh
    material = pyrender.MetallicRoughnessMaterial(doubleSided=True)
    scene = pyrender.Scene()
    owner_node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, owner_faces, process=False), material=material, smooth=False))
    surface_node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices, surface_faces, process=False), material=material, smooth=False))
    renderer = pyrender.OffscreenRenderer(campaign.candidates[0].camera.width, campaign.candidates[0].camera.height)
    rows, panels, candidates = [], [], {item.candidate_id: item for item in campaign.candidates}
    try:
        for candidate in choose_audit_candidates(campaign, selected_ids):
            row = contact_geometry_audit(campaign, candidate, approach)
            row["mask_depth_checks_performed"] = False
            if row["all_geometry_gates_passed"]:
                camera = read(candidate.camera_path)
                view = render["views"][candidate.view_index]
                observed = np.load(seal_source(view["depth"]), allow_pickle=False)
                intrinsics = np.asarray(camera["K"])
                node = scene.add(pyrender.IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]),
                    pose=loader.opencv_to_pyrender_pose(np.asarray(camera["world_to_camera"])))
                owner_node.mesh.is_visible, surface_node.mesh.is_visible = True, False
                owner_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                owner_fraction, owner_mask, owner_visible = mask_visibility(owner_depth, observed, POLICY["depth_tolerance_m"])
                owner_node.mesh.is_visible, surface_node.mesh.is_visible = False, True
                surface_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                fraction, expected, visible = mask_visibility(surface_depth, observed, POLICY["depth_tolerance_m"])
                scene.remove_node(node)
                crop = row["derived_crop_xyxy"]
                _, in_crop, _ = project_probe(points, camera, observed, crop)
                probe_scores = {name: float(project_probe(value, camera, observed, crop)[2].mean()) for name, value in probes.items()}
                metrics = {"surface_area_in_crop_fraction": float(in_crop.mean()), "surface_mask_visible_fraction": fraction,
                    "surface_expected_mask_pixels": int(expected.sum()), "surface_visible_mask_pixels": int(visible.sum()),
                    "owner_mask_visible_fraction": owner_fraction, "probe_visibility": probe_scores}
                gates = {"surface_area_in_crop": metrics["surface_area_in_crop_fraction"] >= POLICY["surface_area_in_crop_min"],
                    "surface_mostly_unoccluded": fraction >= POLICY["surface_mask_visible_fraction_min"],
                    "surface_mask_large_enough": int(visible.sum()) >= POLICY["surface_mask_pixels_min"],
                    "owner_not_cropped": row["target_bbox_retained_fraction"] >= POLICY["owner_bbox_retained_fraction_min"],
                    "owner_mostly_unoccluded": owner_fraction >= POLICY["owner_mask_visible_fraction_min"],
                    "head_space_visible": probe_scores["head"] >= POLICY["head_probe_visible_fraction_min"],
                    "left_foot_space_visible": probe_scores["left_foot"] >= POLICY["each_foot_probe_visible_fraction_min"],
                    "right_foot_space_visible": probe_scores["right_foot"] >= POLICY["each_foot_probe_visible_fraction_min"]}
                row["hypothetical_body_probe_gates_ranking_only"] = {name: gates.pop(name) for name in ("head_space_visible", "left_foot_space_visible", "right_foot_space_visible")}
                row.update(mask_depth_checks_performed=True, mask_metrics=metrics, mask_gates=gates)
                row["gates"].update(gates)
                with Image.open(candidate.rgb_path) as source:
                    pixels = np.asarray(source.convert("RGB")).copy()
                overlay = pixels.copy()
                overlay[owner_visible] = (255, 180, 20)
                overlay[visible] = (0, 245, 180)
                overlay[expected & ~visible] = (245, 40, 60)
                marked = Image.fromarray((pixels*.35+overlay*.65).astype(np.uint8))
                path = output / (candidate.candidate_id + "_masks.png")
                marked.save(path)
                row["mask_visualization"] = artifact(path)
                panels.append((path, f"{candidate.candidate_id} | surface visible={fraction:.2f}; owner={owner_fraction:.2f}; "
                    f"head={probe_scores['head']:.2f}; feet={probe_scores['left_foot']:.2f}/{probe_scores['right_foot']:.2f}"))
            row["all_geometry_gates_passed"] = all(row["gates"].values())
            row["failed_geometry_gates"] = sorted(key for key, passed in row["gates"].items() if not passed)
            rows.append(row)
    finally:
        renderer.delete()
    eligible = [row for row in rows if row["all_geometry_gates_passed"]]
    eligible.sort(key=lambda row: (-row["mask_metrics"]["surface_mask_visible_fraction"],
        -min(row["mask_metrics"]["probe_visibility"].values()), -row["mask_metrics"]["owner_mask_visible_fraction"],
        (-float(row["camera_contact_facing_cosine"]) if row["camera_contact_facing_cosine"] is not None else 1.0), -row["mask_metrics"]["surface_visible_mask_pixels"], row["view_index"]))
    selection = None
    if eligible:
        winner = eligible[0]
        candidate = candidates[winner["candidate_id"]]
        crop = winner["derived_crop_xyxy"]
        for source, destination in [(candidate.rgb_path, output / "selected_crop.png"),
            (seal_source(winner["mask_visualization"]), output / "selected_crop_target_overlay.png")]:
            with Image.open(source) as raw:
                raw.crop(crop).resize((512, 512), Image.Resampling.LANCZOS).save(destination)
        selection = {"selected_candidate_id": candidate.candidate_id, "selected_view_index": candidate.view_index,
            "selected_source_rgb": artifact(candidate.rgb_path), "selected_camera": artifact(candidate.camera_path),
            "selected_crop_xyxy": crop, "selected_stage2_crop": artifact(output / "selected_crop.png"),
            "selected_stage2_crop_target_overlay": artifact(output / "selected_crop_target_overlay.png"),
            "selection_authority": "deterministic_depth_tested_surface_and_owner_masks_with_hypothetical_probe_ranking_only"}
    sheet = board(panels, output / "mask_candidates.png", target["scene_id"]+" / geometric masks: green=surface, red=occluded, amber=owner", width=480, height=360)
    result = {"schema": SCHEMA, "status": "geometry_view_ready" if selection else "rejected_no_complete_visible_view",
        "source_stage1_execution": artifact(stage1_path), "target": stage1["target"], "source_render": artifact(render_path),
        "surface_faces": artifact(archive_path), "policy": POLICY, "geometry_audit": rows, "selection": selection,
        "publication": {"stage2_generation_handoff_allowed": bool(selection), "keypose_publishable": False},
        "qwen_calls": 0, "numeric_geometry_computed_by_qwen": False,
        "owner_completeness_scope": "visibility of the current segmented instance; not proof of semantic segmentation completeness",
        "body_probes_not_a_visibility_guarantee_for_arbitrary_generated_pose": True,
        "visualization": artifact(sheet) if sheet else None, "elapsed_seconds": time.monotonic()-started, "source": artifact(__file__)}
    result["receipt_payload_sha256"] = digest(result)
    write_once(output / "receipt.json", result)
    return result


def source_closure():
    """The actual integrated source files, never the historical source pins."""
    return [artifact(path) for path in (SOURCE, view_numeric.__file__,
        view_geometry.__file__, view_config.__file__, binding.__file__, artifacts.__file__)]


def validate_current_inputs(stage1_path, render_path):
    stage1 = read_sealed(stage1_path)
    approach = bind(stage1)
    target = checked(stage1['target'])
    render = read_sealed(render_path)
    require(render.get('schema') == 'p508.lingo_fullscene_multiview_render.v1'
        and render.get('status') == 'fullscene_multiview_ready', 'Complete calibrated scene render required')
    require(render['scene_id'] == target['scene_id'] == stage1['scene_id'] == approach['scene_id'],
        'Cross-scene selector input')
    require(target['instruction'] == stage1['instruction'] == approach['instruction'],
        'Current instruction identity differs')
    require(len(render['views']) == render['view_count'] == 48
        and [row['view_index'] for row in render['views']] == list(range(48)),
        'Original complete 48-camera inventory required')
    manifest = read(verified(render['calibrated_camera_manifest']))
    view_config.validate_payload_hash(manifest, 'payload_sha256', 'Calibrated camera inventory')
    require(manifest.get('view_count') == 48 and len(manifest.get('views', [])) == 48,
        'Calibrated manifest must preserve all 48 views')
    for current, calibrated in zip(render['views'], sorted(manifest['views'], key=lambda row: row['view_index'])):
        require(all(current[key] == calibrated[key] for key in ('view_index', 'camera', 'rgb')),
            'Render and calibrated manifest camera/RGB inventories differ')
    for render_key, target_key in (('original_scene_mesh', 'original_scene_mesh'),
                                   ('query_occupancy', 'scene_occupancy')):
        require(render['inputs'][render_key] == target['artifacts'][target_key],
            'Render/target geometry identity differs: ' + target_key)
        verified(render['inputs'][render_key])
    for row in render['views']:
        for name in ('camera', 'rgb', 'depth'):
            verified(row[name])
    return stage1, approach, target, render


def candidate_ids(lookup, render, stage1_path, facts_path):
    """Calibration proposals only: historical crop/pass fields are not consumed."""
    require(lookup['grants_new_query_pass'] is False
        and lookup['role'] == 'pre_keypose_generation', 'Hint does not have proposal-only authority')
    require(lookup['current_stage1_execution'] == artifact(stage1_path)
        and lookup['source_facts'] == artifact(facts_path), 'Hint current inputs differ')
    ids = set()
    for hint in lookup['candidates']:
        require(hint.get('historical_only') is True, 'Hint history role missing')
        camera = read(verified(hint['camera']))
        require(camera == hint['camera_parameters'], 'Hint camera copy drift')
        index = camera['camera_id']
        require(type(index) is int and 0 <= index < 48, 'Hint original view index invalid')
        require(read(verified(render['views'][index]['camera'])) == camera,
            'Hint calibration differs from current render')
        expected = f'view_{index:02d}'
        require(hint['original_selected_candidate_id'] == expected, 'Hint candidate ID mismatch')
        ids.add(expected)
    return tuple(sorted(ids))


def _attempt(stage1_path, render_path, output, selected_ids):
    started = time.monotonic()
    receipt = _run_attempt(stage1_path, render_path, output, selected_ids)
    count = sum(row['mask_depth_checks_performed'] is True for row in receipt['geometry_audit'])
    return receipt, dict(receipt=artifact(output / 'receipt.json'),
        candidate_count=len(receipt['geometry_audit']), mask_depth_candidate_count=count,
        actual_depth_render_count=2 * count, seconds=time.monotonic() - started)


def select_views(stage1_path, render_path, output, *, gpu=0, mode='full', facts_path=None, store_root=None):
    """Run current geometry; memory mode internally queries a verified hint bank.

    The default implementation genuinely renders isolated owner and surface
    depths with pyrender/EGL. A cached camera never supplies masks or gate flags.
    Geometry memory misses/stale hints trigger a complete fresh 48-view attempt.
    This operation performs no H3 generation, Qwen body review or pose recovery.
    """
    started = time.monotonic()
    stage1_path, render_path, output = Path(stage1_path), Path(render_path), Path(output)
    require(mode in {'full', 'memory'}, 'Unknown view selection mode')
    require(type(gpu) is int and gpu >= 0, 'GPU must be a physical nonnegative integer')
    require(not output.exists(), 'Refuse existing selector output')
    if mode == 'memory':
        require(facts_path is not None and store_root is not None, 'Memory requires current facts and store')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM='egl', EGL_DEVICE_ID=str(gpu))
    closure = source_closure()
    initial = validate_current_inputs(stage1_path, render_path)
    context = None
    if facts_path is not None:
        from hsi.memory.views import _context
        context = _context(Path(facts_path), stage1_path)
    validation_seconds = time.monotonic() - started
    lookup, lookup_error, selected_ids = None, None, None
    lookup_start = time.monotonic()
    if mode == 'memory':
        from hsi.memory.views import query_views
        try:
            lookup = query_views(Path(facts_path), stage1_path, Path(store_root))
            require(lookup['geometry_identity'] == context['key'], 'Hint geometry identity differs')
            selected_ids = candidate_ids(lookup, initial[3], stage1_path, facts_path) or None
        except (ValueError, OSError, KeyError) as error:
            lookup_error = f'{type(error).__name__}: {error}'
            # Do not publish a rejected hint as a valid memory hit.
            lookup, selected_ids = None, None
    lookup_seconds = time.monotonic() - lookup_start if mode == 'memory' else 0.0
    output.mkdir(parents=True, exist_ok=False)
    if lookup is not None:
        write_once(output / 'lookup.json', lookup)
    chosen, measurement = _attempt(stage1_path, render_path, output / 'attempt_001', selected_ids)
    attempts = [measurement]
    fallback = mode == 'memory' and selected_ids is None
    fallback_reason = lookup_error or ('memory_miss' if fallback else None)
    if selected_ids is not None and chosen['selection'] is None:
        fallback, fallback_reason = True, 'fresh_hint_mask_or_geometry_checks_failed'
        chosen, measurement = _attempt(stage1_path, render_path, output / 'attempt_002_full_fallback', None)
        attempts.append(measurement)
    require(source_closure() == closure, 'Selector source changed during execution')
    require(validate_current_inputs(stage1_path, render_path) == initial, 'Current inputs changed during execution')
    if context is not None:
        from hsi.memory.views import _context
        require(_context(Path(facts_path), stage1_path) == context, 'Current geometry identity changed')
    result = {key: value for key, value in chosen.items() if key != 'receipt_payload_sha256'}
    result.update(source_closure=closure, mode=mode,
        source_facts=artifact(facts_path) if facts_path is not None else None,
        current_geometry_identity=context['key'] if context is not None else None,
        lookup=artifact(output / 'lookup.json') if lookup is not None else None,
        lookup_error=lookup_error, memory_hint_count=len(selected_ids or ()),
        fallback_performed=fallback, fallback_reason=fallback_reason, attempts=attempts,
        original_render_candidate_count=48,
        audited_candidate_count=sum(a['candidate_count'] for a in attempts),
        mask_depth_candidate_count=sum(a['mask_depth_candidate_count'] for a in attempts),
        actual_depth_render_count=sum(a['actual_depth_render_count'] for a in attempts),
        selection_scope='fresh_checked_memory_proposal_subset' if selected_ids and not fallback else 'fresh_full_48_camera_search',
        global_48_camera_optimum_recomputed=selected_ids is None or fallback,
        fresh_check_contract=dict(historical_gate_pass_reused=False, historical_crop_used=False,
            full_render_inventory_preserved=True, current_target_crop_owner_surface_depth_recomputed=True,
            body_probe_ranking_recomputed_with_current_contact_yaw=True,
            retained_numeric_thresholds_unchanged=True, no_navmesh_persisted=True,
            complete_stage2_execution_claimed=False, current_h3_body_and_qwen_checks_still_required=True),
        timing=dict(common_initial_validation_seconds=validation_seconds,
            memory_query_seconds=lookup_seconds, selector_attempts_seconds=sum(a['seconds'] for a in attempts)),
        elapsed_seconds=time.monotonic() - started)
    tick = time.monotonic()
    write_once(output / 'receipt.json', result)
    write_once(output / 'timing.json', dict(schema='hsi.stage2.fresh_view_timing.v1',
        receipt=artifact(output / 'receipt.json'), total_through_receipt_write_seconds=time.monotonic() - started,
        final_receipt_write_seconds=time.monotonic() - tick,
        excludes='Python startup/imports and writing this timing receipt', **result['timing']))
    return output / 'receipt.json'


def validate_receipt(path, stage1_path):
    """Validate truthful current selector authority, not a publishable keypose."""
    result = read_sealed(path)
    require(result['schema'] == SCHEMA and result['source'] == artifact(SOURCE), 'Not integrated current selector')
    require(result['source_stage1_execution'] == artifact(stage1_path), 'Stale Stage1 source')
    require(result['source_closure'] == source_closure() and result['policy'] == POLICY, 'Selector source or policy drift')
    require(result['qwen_calls'] == 0 and result['publication']['keypose_publishable'] is False,
        'View receipt grants pose authority')
    require(result['fresh_check_contract']['historical_gate_pass_reused'] is False
        and result['fresh_check_contract']['historical_crop_used'] is False
        and result['fresh_check_contract']['full_render_inventory_preserved'] is True,
        'Invalid fresh mask/depth contract')
    stage1, approach, target, render = validate_current_inputs(stage1_path, verified(result['source_render']))
    require(result['target'] == stage1['target'], 'Selected target binding drift')
    require(1 <= len(result['attempts']) <= 2, 'Unexpected fresh attempt count')
    attempts = [read_sealed(verified(row['receipt'])) for row in result['attempts']]
    for attempt in attempts:
        require(attempt['schema'] == SCHEMA and attempt['source'] == artifact(SOURCE)
            and attempt['source_stage1_execution'] == result['source_stage1_execution']
            and attempt['source_render'] == result['source_render'] and attempt['policy'] == POLICY,
            'Attempt source/policy mismatch')
    require(attempts[-1]['geometry_audit'] == result['geometry_audit']
        and attempts[-1]['selection'] == result['selection'], 'Final fresh audit differs from actual attempt')
    require(sum(len(a['geometry_audit']) for a in attempts) == result['audited_candidate_count'],
        'Fresh candidate count differs')
    mask_count = sum(row['mask_depth_checks_performed'] is True for a in attempts for row in a['geometry_audit'])
    require(mask_count == result['mask_depth_candidate_count'] and 2 * mask_count == result['actual_depth_render_count'],
        'Fresh mask/depth counts differ')
    require(result['publication']['stage2_generation_handoff_allowed'] == bool(result['selection'])
        and result['status'] == ('geometry_view_ready' if result['selection'] else 'rejected_no_complete_visible_view'),
        'Selection status/authority mismatch')
    if result['selection']:
        selected = result['selection']; index = selected['selected_view_index']
        require(type(index) is int and 0 <= index < 48, 'Selected original index invalid')
        require(render['views'][index]['camera'] == selected['selected_camera']
            and render['views'][index]['rgb'] == selected['selected_source_rgb'], 'Selected camera/RGB drift')
        for key in ('selected_camera', 'selected_source_rgb', 'selected_stage2_crop', 'selected_stage2_crop_target_overlay'):
            verified(selected[key])
        rows = [row for row in result['geometry_audit'] if row['view_index'] == index]
        require(len(rows) == 1 and rows[0]['all_geometry_gates_passed'] is True
            and rows[0]['mask_depth_checks_performed'] is True
            and bool(rows[0]['gates']) and all(rows[0]['gates'].values())
            and bool(rows[0]['mask_gates']) and all(rows[0]['mask_gates'].values())
            and rows[0]['derived_crop_xyxy'] == selected['selected_crop_xyxy'], 'No current passing mask/crop winner')
    for source in result['source_closure']:
        verified(source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage1', 'render', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--facts', type=Path)
    parser.add_argument('--store', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', choices=('full', 'memory'), default='full')
    args = parser.parse_args()
    print(select_views(args.stage1, args.render, args.output, gpu=args.gpu, mode=args.mode,
        facts_path=args.facts, store_root=args.store))


if __name__ == '__main__':
    main()
