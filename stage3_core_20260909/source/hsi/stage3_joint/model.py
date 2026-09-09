"""R4 joint-training architecture, independent of the source-bound R3 runtime.

Retains R3's fusion/C token topology, duration head, scene-first denoiser and
four narrow encoders. Adds encoded-pose semantic differences, separately
learned action/time part importance, and spatial neighborhood aggregation.
Construction never loads or freezes weights. These mechanisms are not claims
that a newly constructed model understands semantics or completes a task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import torch
from torch import Tensor, nn

from hsi.stage3_sequence.contracts import JOINTS, REGIONS, SequenceCondition
from hsi.stage3_sequence.constraints import rotation_geodesic
from hsi.stage3_sequence.model import (
    ContextEncoder, EncodedContext, ParallelFeatureEncoder, SequenceModelConfig,
    SequenceBlock, WholeSequenceDenoiser, positional_encoding,
)
from hsi.stage3_sequence.timing import allocate_durations

ARCHITECTURE = "hsi.stage3_joint.r4.v1"
PARTS = 1 + JOINTS + REGIONS + 1  # root, 22 joints, 8 relations, text


@dataclass(frozen=True)
class JointModelConfig(SequenceModelConfig):
    architecture: str = ARCHITECTURE
    semantic_transition: bool = True
    part_importance: bool = True
    scene_local_neighbors: int = 8
    importance_width: int = 32
    depth_expansion: bool = False
    expansion_source_layers: int = 0
    expansion_residual_init: float = .01

    def validate(self):
        super().validate()
        if self.architecture != ARCHITECTURE:
            raise ValueError("Explicit R4 architecture identity required")
        if not (self.parallel_encoders and self.compact_encoders and
                self.scene_first and self.dynamic_timing and self.encoder_branches == 4):
            raise ValueError("R4 retains four compact encoders, scene-first and timing feedback")
        if type(self.semantic_transition) is not bool or type(self.part_importance) is not bool:
            raise ValueError("R4 ablation switches must be boolean")
        if type(self.importance_width) is not int or not 4 <= self.importance_width <= 128:
            raise ValueError("importance_width must be an integer in [4,128]")
        if type(self.depth_expansion) is not bool or type(self.expansion_source_layers) is not int:
            raise ValueError("Depth expansion requires explicit boolean and integer flags")
        if (not math.isfinite(self.expansion_residual_init)
                or not 0 < self.expansion_residual_init <= .1):
            raise ValueError("Expansion residual initialization must be finite in (0,.1]")
        if self.depth_expansion:
            if not 1 <= self.expansion_source_layers < self.layers:
                raise ValueError("Expansion source depth must be positive and smaller than target depth")
        elif self.expansion_source_layers != 0:
            raise ValueError("Nonexpanded models must declare expansion_source_layers=0")
        return self

    @classmethod
    def large(cls, **kwargs):
        """Explicit 16-block D512 joint model; narrow encoders remain 4x128.

        Keyword overrides permit small CPU structural tests. Depth, gate and
        source-depth flags are serialized; calling this never loads weights.
        """
        options = dict(layers=16, depth_expansion=True, expansion_source_layers=8,
                       expansion_residual_init=.01)
        options.update(kwargs)
        return cls(**options).validate()

    @classmethod
    def from_checkpoint(cls, values):
        explicit = {"architecture", "semantic_transition", "part_importance", "importance_width",
                    "compact_encoders", "parallel_encoders", "encoder_branches", "scene_first",
                    "dynamic_timing", "scene_local_neighbors", "depth_expansion",
                    "expansion_source_layers", "expansion_residual_init"}
        if not isinstance(values, dict) or not explicit.issubset(values):
            raise ValueError("Checkpoint lacks explicit R4 architecture flags")
        return cls(**values).validate()


@dataclass(frozen=True)
class JointEncodedContext(EncodedContext):
    # Graph-connected diagnostics, scoped to this encode call; not model caches.
    diagnostics: dict[str, Tensor]


class SpatialNeighborhoodAggregation(nn.Module):
    """Narrow kNN messages with relative XYZ, learned edge weights and masks.

Self edges are excluded. Queries are chunked, never retaining an N*N matrix.
Discrete neighbor membership is not differentiable, but selected geometry and
features are. Exact distance ties can select different tied neighbors under
permutation; no claim of a mature rotation-equivariant point-cloud backbone.
"""
    def __init__(self, width: int, neighbors: int, chunk_size: int = 128):
        super().__init__()
        if width < 2 or not 1 <= neighbors <= 32 or chunk_size < 1:
            raise ValueError("Invalid spatial graph dimensions")
        self.width, self.neighbors, self.chunk_size = width, neighbors, chunk_size
        self.message = nn.Sequential(nn.Linear(width + 3, width), nn.SiLU(), nn.Linear(width, width))
        self.edge_weight = nn.Sequential(nn.Linear(4, min(width, 32)), nn.SiLU(),
                                         nn.Linear(min(width, 32), 1))
        self.norm = nn.LayerNorm(width)
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, features: Tensor, xyz: Tensor, mask: Tensor) -> Tensor:
        b, n, w = features.shape
        if w != self.width or xyz.shape != (b, n, 3) or mask.shape != (b, n) or mask.dtype != torch.bool:
            raise ValueError("Spatial graph expects features[B,N,W], xyz[B,N,3], mask[B,N]")
        if not torch.isfinite(features).all() or not torch.isfinite(xyz).all():
            raise ValueError("Spatial graph inputs must be finite, including padding")
        count = min(self.neighbors, max(n - 1, 1))
        rows = torch.arange(b, device=xyz.device)[:, None, None]
        outputs = []
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            with torch.no_grad():
                distance = torch.cdist(xyz[:, start:stop].float(), xyz.float())
                self_edge = torch.arange(start, stop, device=xyz.device)[:, None] == torch.arange(n, device=xyz.device)[None]
                distance = distance.masked_fill(~mask[:, None] | self_edge[None], torch.inf)
                nearest, indices = distance.topk(count, dim=-1, largest=False)
                valid = torch.isfinite(nearest) & mask[:, start:stop, None]
            relative = xyz[rows, indices] - xyz[:, start:stop, None]
            delta = features[rows, indices] - features[:, start:stop, None]
            relative = relative.masked_fill(~valid[..., None], 0)
            delta = delta.masked_fill(~valid[..., None], 0)
            message = self.message(torch.cat((relative, delta), -1))
            edge = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
            weight = self.edge_weight(edge).sigmoid().squeeze(-1) * valid.to(features.dtype)
            update = (message * weight[..., None]).sum(-2) / weight.sum(-1, keepdim=True).clamp_min(1e-6)
            changed = self.norm(features[:, start:stop] + self.gain.tanh() * update)
            # With no neighbors, do not manufacture a graph update/normalization.
            changed = torch.where(valid.any(-1, keepdim=True), changed, features[:, start:stop])
            outputs.append(changed.masked_fill(~mask[:, start:stop, None], 0))
        return torch.cat(outputs, 1)


class LearnedPartImportance(nn.Module):
    """Distinct contextual action/time weights over root, joints and relations.

There is no hardcoded root ranking. Part identity, encoded meaning and explicit
change evidence jointly determine normalized weights. Neither head can change
the immutable condition permissions or erase a required contact.
"""
    def __init__(self, dimension: int, width: int):
        super().__init__()
        self.features = nn.Linear(dimension, width)
        self.context = nn.Linear(dimension, width, bias=False)
        self.changes = nn.Linear(4, width, bias=False)
        self.part_id = nn.Embedding(PARTS, width)
        self.score = nn.Linear(width, 2)

    def forward(self, parts: Tensor, changes: Tensor) -> tuple[Tensor, Tensor]:
        if parts.shape[-2] != PARTS or changes.shape != (*parts.shape[:-1], 4):
            raise ValueError("Part importance requires 32 parts and four change channels")
        encoded = (self.features(parts) + self.context(parts.mean(-2))[..., None, :]
                   + self.changes(changes) + self.part_id.weight)
        weights = self.score(torch.tanh(encoded)).softmax(-2)
        return weights[..., 0], weights[..., 1]


class JointContextEncoder(ContextEncoder):
    def __init__(self, config: JointModelConfig):
        super().__init__(config)
        for branch in self.scene.branches:
            branch.local = (SpatialNeighborhoodAggregation(config.feature_width, config.scene_local_neighbors)
                            if config.scene_local_neighbors else None)
        self.semantic_delta_encoder = ParallelFeatureEncoder(config.hidden_dim, config)
        self.semantic_action_gain = nn.Parameter(torch.tensor(.1))
        self.semantic_time_gain = nn.Parameter(torch.tensor(.1))
        self.importance = LearnedPartImportance(config.hidden_dim, config.importance_width)
        # History has no text nor measured target-point normals in this schema.
        # Learned missingness is explicit; no future keypose/contact target is borrowed.
        self.history_contact_geometry_missing = nn.Parameter(torch.zeros(REGIONS, config.hidden_dim))
        self.history_text_missing = nn.Parameter(torch.zeros(config.hidden_dim))

    def _history_parts(self, c, motion, joints):
        b = c.batch_size
        root = self.root(torch.cat((motion[:, :3], torch.ones_like(motion[:, :3]),
                                   torch.zeros_like(motion[:, :3])), -1))
        rotations = motion[:, 3:].reshape(b, 1, JOINTS, 6)
        body = self.body(torch.cat((rotations, joints[:, None],
                                   torch.ones_like(joints[:, None, :, :1]),
                                   torch.zeros_like(joints[:, None, :, :1])), -1))[:, 0]
        contact_input = motion.new_zeros(b, 1, REGIONS, 7)
        contact_input[..., -1] = c.initial_contact_active[:, None].to(motion.dtype)
        unknown_target_body = torch.full((b, 1, REGIONS), -1, device=c.device, dtype=torch.long)
        contact = self.contact(contact_input, unknown_target_body)[:, 0] + self.history_contact_geometry_missing
        text = self.history_text_missing[None, None].expand(b, 1, -1)
        return torch.cat((root[:, None], body, contact, text), 1)

    @staticmethod
    def _part_changes(c, previous_motion, previous_joints, previous_contacts, previous_ids):
        """Distance[m], rotation[rad], relation-state change, evidence-available.

These separately visible channels permit learned relative importance without
declaring root universally dominant. First contact/text geometry differences
are unknown, not replaced with future targets or fabricated history labels.
"""
        b, k = c.keypose_mask.shape
        change = c.history.new_zeros(b, k, PARTS, 4)
        change[..., 0, 0] = (c.keyposes[..., :3] - previous_motion[..., :3]).norm(dim=-1)
        rotations = c.keyposes[..., 3:].reshape(b, k, JOINTS, 6)
        previous_rotations = previous_motion[..., 3:].reshape(b, k, JOINTS, 6)
        angles = rotation_geodesic(rotations, previous_rotations)
        change[..., 0, 1] = angles[..., 0]
        change[..., 1:1+JOINTS, 0] = (c.keypose_joints - previous_joints).norm(dim=-1)
        change[..., 1:1+JOINTS, 1] = angles
        change[..., :1+JOINTS, 3] = 1
        relation_changed = ((c.contact_active != previous_contacts) |
                            ((c.contact_target_ids != previous_ids) & (c.contact_active | previous_contacts)))
        change[..., 1+JOINTS:1+JOINTS+REGIONS, 2] = relation_changed.to(change.dtype)
        # The contact-state comparison is known even without historic target geometry.
        change[..., 1+JOINTS:1+JOINTS+REGIONS, 3] = 1
        if k > 1:
            change[:, 1:, -1, 2] = (c.keypose_text[:, 1:] - c.keypose_text[:, :-1]).norm(dim=-1)
            change[:, 1:, -1, 3] = 1
        return change

    def forward(self, c: SequenceCondition) -> JointEncodedContext:
        c.validate()
        b, k = c.keypose_mask.shape
        m, d, device = c.slot_mask.shape[1], self.config.hidden_dim, c.device
        if k > self.config.max_keyposes or c.max_frames > self.config.max_frames:
            raise ValueError("condition exceeds configured sequence/keypose capacity")
        rows = torch.arange(b, device=device)
        last = c.history_mask.sum(1)-1
        history_last = c.history[rows, last]
        history_prev = c.history[rows, (last-1).clamp_min(0)]
        velocity = (history_last-history_prev)*c.fps
        htokens = self.history(c.history)
        htokens = htokens + positional_encoding(torch.arange(c.history.shape[1], device=device), d).to(htokens.dtype)
        hsummary = (htokens*c.history_mask[..., None]).sum(1)/c.history_mask.sum(1)[:, None]
        hsummary = hsummary + self.history_velocity(velocity)
        scene = self.scene(c)
        root = self.root(torch.cat((c.keyposes[..., :3], c.locked_root.to(c.history.dtype), c.root_tolerance), -1))
        rotations = c.keyposes[..., 3:].reshape(b, k, JOINTS, 6)
        body = self.body(torch.cat((rotations, c.keypose_joints,
                                   c.locked_joints[..., None].to(c.history.dtype), c.joint_tolerance[..., None]), -1))
        contact_features = torch.cat((c.contact_targets-c.keyposes[..., :3, None].transpose(-1, -2),
                                     c.contact_normals, c.contact_active[..., None].to(c.history.dtype)), -1)
        contacts = self.contact(contact_features, c.contact_body_regions)
        text = self.text(c.keypose_text)+self.semantic(c.semantic_ids.clamp(0, 31))
        parts = torch.cat((root[:, :, None], body, contacts, text[:, :, None]), 2)
        semantic = parts.mean(2)
        previous_motion = torch.cat((history_last[:, None], c.keyposes[:, :-1]), 1)
        previous_joints = torch.cat((c.history_joints[rows, last][:, None], c.keypose_joints[:, :-1]), 1)
        previous_contacts = torch.cat((c.initial_contact_active[:, None], c.contact_active[:, :-1]), 1)
        previous_ids = torch.cat((c.initial_contact_target_ids[:, None], c.contact_target_ids[:, :-1]), 1)
        keep = c.contact_active & previous_contacts
        make = c.contact_active & ~previous_contacts
        release = previous_contacts & ~c.contact_active
        changed = (c.contact_target_ids != previous_ids) & (c.contact_active | previous_contacts)
        key_routes = c.route_features[rows[:, None], c.keypose_slots.clamp_min(0)]
        original_transition = self.transition(torch.cat((c.keyposes-previous_motion,
            (c.keypose_joints-previous_joints).flatten(-2), keep.to(c.history.dtype), make.to(c.history.dtype),
            release.to(c.history.dtype), changed.to(c.history.dtype), key_routes), -1))
        changes = self._part_changes(c, previous_motion, previous_joints, previous_contacts, previous_ids)
        if self.config.part_importance:
            action_weight, time_weight = self.importance(parts, changes)
            action_summary = (parts*action_weight[..., None]).sum(2)
            time_summary = (parts*time_weight[..., None]).sum(2)
        else:
            action_summary = time_summary = semantic
            action_weight = time_weight = parts.new_full((b, k, PARTS), 1/PARTS)
        if self.config.semantic_transition:
            history_semantic = self._history_parts(c, history_last, c.history_joints[rows, last]).mean(1)
            previous_semantic = torch.cat((history_semantic[:, None], semantic[:, :-1]), 1)
            semantic_delta = semantic-previous_semantic
            encoded_delta = self.semantic_delta_encoder(semantic_delta)
            transitions = original_transition+self.semantic_action_gain.tanh()*encoded_delta
            time_transitions = original_transition+self.semantic_time_gain.tanh()*encoded_delta
        else:
            history_semantic = parts.new_zeros(b, d)
            semantic_delta = parts.new_zeros(b, k, d)
            transitions = time_transitions = original_transition
        order = positional_encoding(torch.arange(k, device=device), d).to(semantic.dtype)
        keys = action_summary+order+self.kind.weight[0]
        action_transitions = transitions+order+self.kind.weight[1]
        time_transitions = time_transitions+order+self.kind.weight[1]
        interleaved = torch.stack((action_transitions, keys), 2).flatten(1, 2)
        owners = c.slot_keyposes.clamp_min(0)
        # Same M event tokens, same shared fusion and duration head. The event
        # summary has its own part weighting; this is not an independent C graph.
        slot_context = (time_summary[rows[:, None], owners]+time_transitions[rows[:, None], owners]
                        +self.route(c.route_features)+self.event_type(c.slot_types.clamp(0, 3)))
        slot_context = slot_context+positional_encoding(torch.arange(m, device=device), d).to(slot_context.dtype)
        length = self.length(torch.stack((c.total_frames.to(c.history.dtype).log1p(),
            torch.full((b,), math.log1p(c.fps), device=device, dtype=c.history.dtype)), -1))
        prefix = torch.stack((hsummary+self.kind.weight[2], scene.mean(1)+self.kind.weight[3], length+self.kind.weight[4]), 1)
        tokens = torch.cat((prefix, interleaved, slot_context), 1)
        pad = torch.cat((torch.zeros(b, 3, dtype=torch.bool, device=device),
                         ~c.keypose_mask.repeat_interleave(2, dim=1), ~c.slot_mask), 1)
        fused = self.fusion(tokens, src_key_padding_mask=pad)
        fused_slots = fused[:, 3+2*k:]
        schedule = allocate_durations(self.duration(fused_slots, c.total_frames), c.minimum_frames,
                                      c.slot_mask, c.total_frames, c.keypose_slots)
        key_context = fused[:, 4:3+2*k:2]
        detailed = (parts+key_context[:, :, None]).flatten(1, 2)
        all_tokens = torch.cat((fused, detailed), 1)
        all_pad = torch.cat((pad, ~c.keypose_mask.repeat_interleave(PARTS, dim=1)), 1)
        all_tokens = all_tokens.masked_fill(all_pad[..., None], 0)
        token_keys = torch.cat((torch.full((b, 3), -1, device=device, dtype=torch.long),
            torch.arange(k, device=device)[None].expand(b, -1).repeat_interleave(2, dim=1), owners,
            torch.arange(k, device=device)[None].expand(b, -1).repeat_interleave(PARTS, dim=1)), 1)
        diagnostics = dict(encoded_pose_semantics=semantic, history_semantics=history_semantic,
            semantic_deltas=semantic_delta, action_part_importance=action_weight,
            time_part_importance=time_weight, part_change_evidence=changes)
        return JointEncodedContext(all_tokens, all_pad, token_keys, scene, fused_slots, schedule, c, diagnostics)


class ExpandedSequenceBlock(SequenceBlock):
    """An appended full block with a small, nonzero trainable residual gain.

    Subclassing retains all original block tensor names. A nonzero gain gives
    copied attention/FF/scene/relation weights gradients immediately; no block
    is frozen. The gain is allowed to learn either sign after initialization.
    """
    def __init__(self, config: JointModelConfig):
        super().__init__(config)
        self.expansion_gain = nn.Parameter(torch.tensor(float(config.expansion_residual_init)))

    def forward(self, x, frame_mask, context, attention_bias, relations,
                phase=None, phase_encoder=None):
        full = super().forward(x, frame_mask, context, attention_bias, relations,
                               phase=phase, phase_encoder=phase_encoder)
        return (x+self.expansion_gain.tanh()*(full-x)).masked_fill(~frame_mask[..., None], 0)


class JointSequenceDenoiser(WholeSequenceDenoiser):
    """Same denoise/sampler interface as R3, with explicit R4 context semantics."""
    def __init__(self, config: JointModelConfig | None = None):
        config = (config or JointModelConfig()).validate()
        if not isinstance(config, JointModelConfig):
            raise TypeError("Use JointModelConfig, not an implicitly upgraded R3 config")
        super().__init__(config)
        self.context_encoder = JointContextEncoder(config)
        if config.depth_expansion:
            for index in range(config.expansion_source_layers, config.layers):
                block = ExpandedSequenceBlock(config)
                base = self.blocks[index % config.expansion_source_layers].state_dict()
                merged = dict(block.state_dict())
                merged.update(base)
                block.load_state_dict(merged, strict=True)
                self.blocks[index] = block
        for parameter in self.parameters():
            parameter.requires_grad_(True)


class MigrationError(ValueError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def migrate_r3_checkpoint(model: JointSequenceDenoiser, payload: dict) -> dict:
    """Explicit compatible-weight transfer, never a silent non-strict load.

R3 checkpoint provenance/source SHA verification is the caller's responsibility.
This function checks architecture, tensor completeness, shapes and finiteness;
then strict-loads a full merged R4 state and reports every newly initialized key.
Optimizer and normalizer states are not migrated here. New branches remain
random/trainable and must receive actual joint supervision before quality claims.
"""
    if not isinstance(model, JointSequenceDenoiser):
        raise TypeError("R4 target model required")
    accepted = {"hsi.stage3_sequence.observed_motion_only.v1", "hsi.stage3_sequence.jitter_lingo_warm_start.v1"}
    if not isinstance(payload, dict) or payload.get("schema") not in accepted:
        raise ValueError("Explicit original R3 motion-only checkpoint schema required")
    source_config = SequenceModelConfig.from_checkpoint(payload.get("model_config"))
    if not (source_config.parallel_encoders and source_config.encoder_branches == 4 and
            source_config.scene_first and source_config.dynamic_timing):
        raise ValueError("Only explicit four-branch scene-first R3 migration is supported")
    source = payload.get("model_state")
    if not isinstance(source, dict) or not source or not all(isinstance(x, Tensor) for x in source.values()):
        raise ValueError("Checkpoint needs a tensor model_state")
    target = model.state_dict()
    permitted = {"scene_local_neighbors", "max_frames", "max_keyposes"}
    if model.config.depth_expansion:
        permitted.add("layers")
    config_changes = {f.name: {"source": getattr(source_config, f.name), "target": getattr(model.config, f.name)}
                      for f in fields(SequenceModelConfig) if getattr(source_config, f.name) != getattr(model.config, f.name)}
    report = dict(schema="hsi.stage3_joint.r3_migration.v1", source_schema=payload["schema"],
        source_config=asdict(source_config), target_config=asdict(model.config), config_changes=config_changes,
        transferred=[], newly_initialized=[], depth_copied=[], source_unexpected=[], shape_mismatch=[],
        source_missing_shared=[], all_parameters_trainable=True, optimizer_migrated=False,
        normalizer_migrated=False, joint_training_completed=False,
        depth_expansion=dict(enabled=model.config.depth_expansion,
            source_layers=source_config.layers, target_layers=model.config.layers,
            mapping="appended block i copies source block i % source_layers",
            gain_parameter="expansion_gain", gain_init=model.config.expansion_residual_init,
            gain_effective_init=math.tanh(model.config.expansion_residual_init),
            gates_zero_initialized=False, added_blocks_frozen=False))
    if model.config.depth_expansion and source_config.layers != model.config.expansion_source_layers:
        raise MigrationError("Declared expansion source depth differs from source checkpoint", report)
    if set(config_changes)-permitted:
        raise MigrationError("Shared R3 architecture/config differs; refusing ambiguous weight transfer", report)
    new_prefixes = ("context_encoder.semantic_", "context_encoder.importance.",
                    "context_encoder.history_contact_geometry_missing", "context_encoder.history_text_missing")
    for key, value in source.items():
        if not bool(torch.isfinite(value).all()):
            raise MigrationError("Nonfinite source tensor: "+key, report)
        source_block = int(key.split('.')[1]) if key.startswith("blocks.") and key.split('.')[1].isdigit() else None
        if key not in target or (source_block is not None and source_block >= source_config.layers):
            report["source_unexpected"].append(key)
        elif value.shape != target[key].shape:
            report["shape_mismatch"].append(dict(name=key, source=list(value.shape), target=list(target[key].shape)))
        else:
            report["transferred"].append(key)
    copied = {}
    for key in target.keys()-source.keys():
        if model.config.depth_expansion and key.startswith("blocks."):
            _, index, suffix = key.split('.', 2)
            index = int(index)
            if index >= model.config.expansion_source_layers:
                if suffix == "expansion_gain":
                    report["newly_initialized"].append(key)
                    continue
                old_key = f"blocks.{index % model.config.expansion_source_layers}.{suffix}"
                if old_key in source and source[old_key].shape == target[key].shape:
                    copied[key] = source[old_key]
                    report["depth_copied"].append(dict(source=old_key, target=key))
                    continue
                report["source_missing_shared"].append(old_key)
                continue
        report["newly_initialized"].append(key)
        if not (key.startswith(new_prefixes) or (key.startswith("context_encoder.scene.branches.") and ".local." in key)):
            report["source_missing_shared"].append(key)
    for name in ("transferred", "newly_initialized", "source_unexpected", "source_missing_shared"):
        report[name].sort()
    report["depth_copied"].sort(key=lambda item: item["target"])
    if report["shape_mismatch"] or report["source_unexpected"] or report["source_missing_shared"]:
        raise MigrationError("Incomplete/incompatible source checkpoint; see explicit migration report", report)
    merged = {key: value.detach().to(device=target[key].device, dtype=target[key].dtype)
              if key in report["transferred"] else target[key] for key, value in source.items() if key in target}
    merged.update({key: target[key] for key in report["newly_initialized"]})
    merged.update({key: value.detach().to(device=target[key].device, dtype=target[key].dtype)
                   for key, value in copied.items()})
    if model.config.depth_expansion:
        for index in range(model.config.expansion_source_layers, model.config.layers):
            key = f"blocks.{index}.expansion_gain"
            merged[key] = target[key].new_tensor(model.config.expansion_residual_init)
    model.load_state_dict(merged, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    report.update(transferred_count=len(report["transferred"]), new_count=len(report["newly_initialized"]),
                  depth_copied_count=len(report["depth_copied"]),
                  status="explicit_compatible_transfer_not_joint_training")
    return report
