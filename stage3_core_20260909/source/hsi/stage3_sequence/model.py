"""Whole-sequence denoiser with shared bidirectional context and event timing."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .contracts import (
    SequenceCondition, MOTION_DIM, JOINTS, REGIONS, ROUTE_DIM,
    SCENE_FEATURE_DIM, TEXT_DIM, future_mask,
)
from .timing import DurationHead, EventSchedule, allocate_durations


def positional_encoding(positions: Tensor, width: int) -> Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device, dtype=torch.float32)
        / max(half-1,1))
    phase = positions.float()[...,None] * frequencies
    return torch.cat((phase.sin(),phase.cos()),dim=-1).to(
        dtype=positions.dtype if positions.is_floating_point() else torch.float32)


def mlp(input_dim, hidden_dim, output_dim=None):
    return nn.Sequential(nn.Linear(input_dim,hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim,output_dim or hidden_dim))


def progressive_mlp(input_dim: int, output_dim: int, *, first_width: int = 32):
    """Extract low-dimensional features before exposing a D-wide token.

    Only the final projection uses the common backbone width. Scene/body
    callers deliberately stop at a smaller width until after their pooling
    or joint-context operation; this is not a bottleneck followed by N*D.
    """
    widths = []
    width = min(first_width, output_dim)
    while width < output_dim:
        widths.append(width)
        width *= 2
    widths.append(output_dim)
    layers = []
    previous = input_dim
    for i, width in enumerate(widths):
        layers.append(nn.Linear(previous, width))
        if i + 1 < len(widths):
            layers.append(nn.SiLU())
        previous = width
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class SequenceModelConfig:
    hidden_dim: int = 512
    heads: int = 8
    layers: int = 8
    context_layers: int = 2
    body_layers: int = 1
    ff_multiplier: int = 4
    scene_tokens: int = 32
    relation_dim: int = 14
    dropout: float = 0.0
    max_frames: int = 240
    max_keyposes: int = 16
    time_bias_width: float = 12.0
    compact_encoders: bool = True
    scene_first: bool = True
    dynamic_timing: bool = True
    scene_local_neighbors: int = 0  # optional spatial ablation, not enabled by default
    parallel_encoders: bool = True
    encoder_branches: int = 4

    @classmethod
    def legacy(cls, **kwargs):
        """Explicit v1 architecture for controlled ablations/old state dicts.

        New checkpoint configs must serialize all flags; an old checkpoint
        must never silently acquire v2 semantics through missing defaults.
        """
        options = dict(compact_encoders=False, scene_first=False,
                       dynamic_timing=False, scene_local_neighbors=0,
                       parallel_encoders=False)
        options.update(kwargs)
        return cls(**options)

    @classmethod
    def revision_r2(cls, **kwargs):
        """Explicit saved R2 progressive-width architecture, not the new default."""
        options = dict(compact_encoders=True, scene_first=True,
                       dynamic_timing=True, scene_local_neighbors=0,
                       parallel_encoders=False)
        options.update(kwargs)
        return cls(**options)

    @classmethod
    def from_checkpoint(cls, values):
        """Reject ambiguous architecture semantics in a saved configuration.

        In particular scene_first has no state-dict shape signature. Missing
        flags cannot safely be inferred from a successful weight-shape check.
        Explicit legacy loading is available separately via legacy().
        """
        required = {'compact_encoders', 'scene_first', 'dynamic_timing', 'scene_local_neighbors',
                    'parallel_encoders', 'encoder_branches'}
        if not isinstance(values, dict) or not required.issubset(values):
            raise ValueError("Checkpoint lacks explicit architecture flags; use its original code "
                             "or an explicitly verified legacy/revision_r2 configuration, "
                             "not new defaults")
        return cls(**values).validate()

    @property
    def feature_width(self):
        if self.parallel_encoders:
            return self.hidden_dim // self.encoder_branches
        return min(self.hidden_dim, 128) if self.compact_encoders else self.hidden_dim

    def validate(self):
        if self.heads < 1 or self.hidden_dim < 8 or self.hidden_dim % 2 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be even and divisible by heads")
        if min(self.layers,self.context_layers,self.body_layers,self.scene_tokens,
               self.ff_multiplier,self.max_frames,self.max_keyposes,self.relation_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1 or not self.time_bias_width > 0:
            raise ValueError("invalid dropout/time attention width")
        if any(type(getattr(self, name)) is not bool for name in
               ('compact_encoders', 'scene_first', 'dynamic_timing', 'parallel_encoders')):
            raise ValueError("architecture switches must be boolean")
        if type(self.encoder_branches) is not int or self.encoder_branches < 2:
            raise ValueError("encoder_branches must be an integer >= 2")
        if self.parallel_encoders and (not self.compact_encoders or
                self.hidden_dim % self.encoder_branches or self.feature_width < 2):
            raise ValueError("parallel encoders require compact_encoders and equal branches "
                             "of at least two channels dividing hidden_dim")
        if type(self.scene_local_neighbors) is not int or not 0 <= self.scene_local_neighbors <= 32:
            raise ValueError("scene_local_neighbors must be an integer in [0,32]")
        return self


@dataclass(frozen=True)
class EncodedContext:
    tokens: Tensor
    padding_mask: Tensor
    token_keyposes: Tensor
    scene_tokens: Tensor
    slot_context: Tensor
    schedule: EventSchedule
    condition: SequenceCondition


@dataclass(frozen=True)
class DenoiserOutput:
    clean_motion: Tensor
    contact_logits: Tensor
    frame_mask: Tensor


class LocalPointAggregation(nn.Module):
    """Optional chunked kNN graph message passing, tested separately from DIM.

    Relative xyz and neighbor-minus-center features give an explicit local
    spatial operation. No point-order convolution is used. Distances are
    chunked so an N*N distance tensor is not kept. Exact kNN distance ties
    can choose different equivalent-distance neighbors after permutation.
    """
    def __init__(self, width, neighbors):
        super().__init__()
        self.neighbors = neighbors
        self.message = mlp(width + 3, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, features, xyz, mask):
        b, n, _ = features.shape
        count = min(self.neighbors, n)
        rows = torch.arange(b, device=xyz.device)[:, None, None]
        outputs = []
        for start in range(0, n, 128):
            stop = min(start + 128, n)
            with torch.no_grad():
                distances = torch.cdist(xyz[:, start:stop].float(), xyz.float())
                distances = distances.masked_fill(~mask[:, None], torch.inf)
                indices = distances.topk(count, dim=-1, largest=False).indices
            valid = mask[rows, indices]
            relative = xyz[rows, indices] - xyz[:, start:stop, None]
            delta = features[rows, indices] - features[:, start:stop, None]
            message = self.message(torch.cat((relative, delta), dim=-1))
            update = message.masked_fill(~valid[..., None], 0).sum(-2)
            update = update / valid.sum(-1, keepdim=True).clamp_min(1)
            outputs.append(self.norm(features[:, start:stop] + update))
        return torch.cat(outputs, dim=1).masked_fill(~mask[..., None], 0)


class ParallelFeatureEncoder(nn.Module):
    """Independent equal-width feature extractors, combined ONLY by concatenation.

    Every branch reads the same complete input. Independent weights permit,
    but do not guarantee, complementary learned features. No semantic meaning
    is assigned to a branch, no weights are shared, and no wide projection is
    hidden after concatenation. This replaces the R2 serial width ladder.
    """
    def __init__(self, input_dim, config):
        super().__init__()
        w = config.feature_width
        self.branches = nn.ModuleList([
            nn.Sequential(*mlp(input_dim, w), nn.LayerNorm(w))
            for _ in range(config.encoder_branches)])

    def forward(self, features):
        return torch.cat([branch(features) for branch in self.branches], dim=-1)


class NarrowBodyBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.feature_width
        self.stem = mlp(11, w)
        self.joint_id = nn.Embedding(JOINTS, w)
        layer = nn.TransformerEncoderLayer(
            w, math.gcd(config.heads, w), w*config.ff_multiplier, config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, config.body_layers, norm=nn.LayerNorm(w), enable_nested_tensor=False)

    def forward(self, features):
        b, k, j, _ = features.shape
        encoded = self.stem(features) + self.joint_id.weight
        return self.transformer(encoded.reshape(b*k, j, -1)).reshape(b, k, j, -1)


class ParallelBodyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branches = nn.ModuleList([
            NarrowBodyBranch(config) for _ in range(config.encoder_branches)])

    def forward(self, features):
        return torch.cat([branch(features) for branch in self.branches], dim=-1)


class NarrowContactBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.feature_width
        self.stem = mlp(7, w)
        self.region_id = nn.Embedding(REGIONS, w)
        self.target_body_id = nn.Embedding(REGIONS+1, w)
        self.norm = nn.LayerNorm(w)

    def forward(self, features, target_body_ids):
        return self.norm(self.stem(features) + self.region_id.weight
                         + self.target_body_id(target_body_ids+1))


class ParallelContactEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branches = nn.ModuleList([
            NarrowContactBranch(config) for _ in range(config.encoder_branches)])

    def forward(self, features, target_body_ids):
        return torch.cat([branch(features, target_body_ids)
                          for branch in self.branches], dim=-1)


class NarrowSceneBranch(nn.Module):
    """Pool N*w into S*w inside this branch; never concatenate N-wide features."""
    def __init__(self, config):
        super().__init__()
        w = config.feature_width
        self.point_encoder = mlp(3+SCENE_FEATURE_DIM, w)
        self.local = (LocalPointAggregation(w, config.scene_local_neighbors)
                      if config.scene_local_neighbors else None)
        self.queries = nn.Parameter(torch.randn(config.scene_tokens, w)/math.sqrt(w))
        self.pool = nn.MultiheadAttention(w, math.gcd(config.heads, w),
                                         dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(w)

    def forward(self, features, condition):
        points = self.point_encoder(features)
        if self.local is not None:
            points = self.local(points, condition.scene_points, condition.scene_mask)
        queries = self.queries[None].expand(condition.batch_size, -1, -1)
        tokens, _ = self.pool(queries, points, points,
                             key_padding_mask=~condition.scene_mask, need_weights=False)
        return self.norm(tokens+queries)


class SceneEncoder(nn.Module):
    """Permutation-invariant learned pooling of explicit scene geometry."""
    def __init__(self, config):
        super().__init__()
        if config.parallel_encoders:
            self.branches = nn.ModuleList([
                NarrowSceneBranch(config) for _ in range(config.encoder_branches)])
            return
        d, w = config.hidden_dim, config.feature_width
        self.point_encoder = (progressive_mlp(3+SCENE_FEATURE_DIM,w)
                              if config.compact_encoders else mlp(3+SCENE_FEATURE_DIM,d))
        self.local = (LocalPointAggregation(w, config.scene_local_neighbors)
                      if config.scene_local_neighbors else None)
        self.queries = nn.Parameter(torch.randn(config.scene_tokens,w)/math.sqrt(w))
        self.pool = nn.MultiheadAttention(w,math.gcd(config.heads,w),dropout=config.dropout,batch_first=True)
        self.norm = nn.LayerNorm(w)
        self.output_projection = nn.Linear(w,d) if w != d else nn.Identity()

    def forward(self, condition):
        if hasattr(self, 'branches'):
            features = torch.cat((condition.scene_points, condition.scene_features), dim=-1)
            return torch.cat([branch(features, condition) for branch in self.branches], dim=-1)
        points = self.point_encoder(torch.cat(
            (condition.scene_points,condition.scene_features),dim=-1))
        if self.local is not None:
            points = self.local(points, condition.scene_points, condition.scene_mask)
        queries = self.queries[None].expand(condition.batch_size,-1,-1)
        tokens,_ = self.pool(queries,points,points,
                           key_padding_mask=~condition.scene_mask,need_weights=False)
        return self.output_projection(self.norm(tokens+queries))


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, w = config.hidden_dim, config.feature_width
        self.scene = SceneEncoder(config)
        self.history = mlp(MOTION_DIM,d)
        self.history_velocity = mlp(MOTION_DIM,d)
        stem = ((lambda input_dim, _: ParallelFeatureEncoder(input_dim, config))
                if config.parallel_encoders else
                progressive_mlp if config.compact_encoders else mlp)
        self.root = stem(3+3+3,d)  # position, immutable flags, tolerances
        if config.parallel_encoders:
            self.body = ParallelBodyEncoder(config)
            self.contact = ParallelContactEncoder(config)
        else:
            self.body = stem(6+3+1+1,w)  # rotation, xyz, immutable, angular tolerance
            self.joint_id = nn.Embedding(JOINTS,w)
            self.region_id = nn.Embedding(REGIONS,w)
            self.target_body_id = nn.Embedding(REGIONS+1,w)
            self.contact = stem(3+3+1,w)
            self.body_projection = nn.Linear(w,d) if w != d else nn.Identity()
            self.contact_projection = nn.Linear(w,d) if w != d else nn.Identity()
        self.text = nn.Linear(TEXT_DIM,d)
        self.semantic = nn.Embedding(32,d)
        if not config.parallel_encoders:
            body_layer = nn.TransformerEncoderLayer(
                w,math.gcd(config.heads,w),w*config.ff_multiplier,config.dropout,
                activation="gelu",batch_first=True,norm_first=True)
            self.body_transformer = nn.TransformerEncoder(
                body_layer,config.body_layers,norm=nn.LayerNorm(w),enable_nested_tensor=False)
        self.transition = mlp(MOTION_DIM+JOINTS*3+REGIONS*4+ROUTE_DIM,d)
        self.route = stem(ROUTE_DIM,d)
        self.event_type = nn.Embedding(4,d)
        self.length = (ParallelFeatureEncoder(2, config) if config.parallel_encoders else
                       progressive_mlp(2,d,first_width=8) if config.compact_encoders
                       else mlp(2,d))  # log future frames, fps
        self.kind = nn.Embedding(5,d)
        context_layer = nn.TransformerEncoderLayer(
            d,config.heads,d*config.ff_multiplier,config.dropout,
            activation="gelu",batch_first=True,norm_first=True)
        self.fusion = nn.TransformerEncoder(
            context_layer,config.context_layers,norm=nn.LayerNorm(d),enable_nested_tensor=False)
        self.duration = DurationHead(d)

    def forward(self, c: SequenceCondition) -> EncodedContext:
        c.validate()
        b,k = c.keypose_mask.shape
        m = c.slot_mask.shape[1]
        if k > self.config.max_keyposes or c.max_frames > self.config.max_frames:
            raise ValueError("condition exceeds configured sequence/keypose capacity")
        d, device = self.config.hidden_dim, c.device
        rows = torch.arange(b,device=device)
        last = c.history_mask.sum(1)-1
        history_last = c.history[rows,last]
        history_prev = c.history[rows,(last-1).clamp_min(0)]
        velocity = (history_last-history_prev)*c.fps
        htokens = self.history(c.history)
        hpos = positional_encoding(
            torch.arange(c.history.shape[1],device=device),d).to(htokens.dtype)
        htokens = htokens+hpos
        hsummary = (htokens*c.history_mask[...,None]).sum(1)/c.history_mask.sum(1)[:,None]
        hsummary = hsummary+self.history_velocity(velocity)
        scene = self.scene(c)
        root = self.root(torch.cat((c.keyposes[...,:3],c.locked_root.to(c.history.dtype),
                                   c.root_tolerance),dim=-1))
        rotations = c.keyposes[...,3:].reshape(b,k,JOINTS,6)
        body = self.body(torch.cat((rotations,c.keypose_joints,
            c.locked_joints[...,None].to(c.history.dtype),c.joint_tolerance[...,None]),dim=-1))
        if not self.config.parallel_encoders:
            body = body+self.joint_id.weight[None,None]
            w = self.config.feature_width
            body = self.body_transformer(body.reshape(b*k,JOINTS,w)).reshape(b,k,JOINTS,w)
            body = self.body_projection(body)
        contact_features = torch.cat((
            c.contact_targets-c.keyposes[...,:3,None].transpose(-1,-2),
            c.contact_normals,c.contact_active[...,None].to(c.history.dtype)),dim=-1)
        if self.config.parallel_encoders:
            contacts = self.contact(contact_features, c.contact_body_regions)
        else:
            contacts = self.contact(contact_features)
            contacts = contacts+self.region_id.weight[None,None]+self.target_body_id(c.contact_body_regions+1)
            contacts = self.contact_projection(contacts)
        text = self.text(c.keypose_text)+self.semantic(c.semantic_ids.clamp(0,31))
        parts = torch.cat((root[:,:,None],body,contacts,text[:,:,None]),dim=2)
        key_summary = parts.mean(2)
        previous_motion = torch.cat((history_last[:,None],c.keyposes[:,:-1]),dim=1)
        previous_joints = torch.cat((c.history_joints[rows,last][:,None],
                                    c.keypose_joints[:,:-1]),dim=1)
        previous_contacts = torch.cat((c.initial_contact_active[:,None],
                                       c.contact_active[:,:-1]),dim=1)
        previous_ids = torch.cat((c.initial_contact_target_ids[:,None],
                                  c.contact_target_ids[:,:-1]),dim=1)
        keep = c.contact_active & previous_contacts
        make = c.contact_active & ~previous_contacts
        release = previous_contacts & ~c.contact_active
        changed = (c.contact_target_ids != previous_ids) & (c.contact_active | previous_contacts)
        key_routes = c.route_features[rows[:,None],c.keypose_slots.clamp_min(0)]
        transitions = self.transition(torch.cat((
            c.keyposes-previous_motion,
            (c.keypose_joints-previous_joints).flatten(-2),
            keep.to(c.history.dtype),make.to(c.history.dtype),
            release.to(c.history.dtype),changed.to(c.history.dtype),key_routes),dim=-1))
        order = positional_encoding(torch.arange(k,device=device),d).to(key_summary.dtype)
        keys = key_summary+order+self.kind.weight[0]
        transitions = transitions+order+self.kind.weight[1]
        interleaved = torch.stack((transitions,keys),dim=2).flatten(1,2)
        owners = c.slot_keyposes.clamp_min(0)
        slot_context = (key_summary[rows[:,None],owners]+transitions[rows[:,None],owners]
                        +self.route(c.route_features)+self.event_type(c.slot_types.clamp(0,3)))
        slot_context = slot_context+positional_encoding(torch.arange(m,device=device),d).to(slot_context.dtype)
        length = self.length(torch.stack(
            (c.total_frames.to(c.history.dtype).log1p(),
             torch.full((b,),math.log1p(c.fps),device=device,dtype=c.history.dtype)),dim=-1))
        prefix = torch.stack((hsummary+self.kind.weight[2],
                              scene.mean(1)+self.kind.weight[3],
                              length+self.kind.weight[4]),dim=1)
        tokens = torch.cat((prefix,interleaved,slot_context),dim=1)
        pad = torch.cat((torch.zeros(b,3,dtype=torch.bool,device=device),
                         ~c.keypose_mask.repeat_interleave(2,dim=1),~c.slot_mask),dim=1)
        fused = self.fusion(tokens,src_key_padding_mask=pad)
        fused_slots = fused[:,3+2*k:]
        logits = self.duration(fused_slots,c.total_frames)
        schedule = allocate_durations(logits,c.minimum_frames,c.slot_mask,
                                      c.total_frames,c.keypose_slots)
        # Preserve body/contact granularity after global bidirectional fusion.
        key_context = fused[:,4:3+2*k:2]
        detailed = (parts+key_context[:,:,None]).flatten(1,2)
        p = parts.shape[2]
        all_tokens = torch.cat((fused,detailed),dim=1)
        all_pad = torch.cat((pad,~c.keypose_mask.repeat_interleave(p,dim=1)),dim=1)
        all_tokens = all_tokens.masked_fill(all_pad[...,None],0)
        token_keys = torch.cat((
            torch.full((b,3),-1,device=device,dtype=torch.long),
            torch.arange(k,device=device)[None].expand(b,-1).repeat_interleave(2,dim=1),
            owners,
            torch.arange(k,device=device)[None].expand(b,-1).repeat_interleave(p,dim=1)),dim=1)
        return EncodedContext(all_tokens,all_pad,token_keys,scene,fused_slots,schedule,c)


class SequenceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.heads = config.heads
        self.scene_first = config.scene_first
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(5)])
        self.temporal = nn.MultiheadAttention(d,config.heads,dropout=config.dropout,batch_first=True)
        self.context = nn.MultiheadAttention(d,config.heads,dropout=config.dropout,batch_first=True)
        self.scene = nn.MultiheadAttention(d,config.heads,dropout=config.dropout,batch_first=True)
        self.relation_scale_bias = nn.Linear(d,2*d)
        self.relation_gate = nn.Parameter(torch.tensor(0.1))
        self.ff = nn.Sequential(nn.Linear(d,d*config.ff_multiplier),nn.GELU(),
                                nn.Dropout(config.dropout),nn.Linear(d*config.ff_multiplier,d))

    def forward(self, x, frame_mask, context, attention_bias, relations,
                phase=None, phase_encoder=None):
        q = self.norms[0](x)
        update,_ = self.temporal(q,q,q,key_padding_mask=~frame_mask,need_weights=False)
        x = x+update
        if self.scene_first:
            q = self.norms[2](x)
            update,_ = self.scene(q,context.scene_tokens,context.scene_tokens,need_weights=False)
            x = x+update
            scale,offset = self.relation_scale_bias(relations).chunk(2,dim=-1)
            x = x+torch.tanh(self.relation_gate)*(self.norms[3](x)*torch.tanh(scale)+offset)
            # Event phase is deliberately computed/injected only AFTER the
            # first scene/geometry read, not silently added at B's input.
            if phase is not None:
                x = x+phase_encoder(phase)
        q = self.norms[1](x)
        # Merge padding into additive attention mask: no mixed bool/float mask,
        # and global/history tokens remain visible at every valid motion frame.
        bias = attention_bias.masked_fill(context.padding_mask[:,None],-torch.inf)
        mask = bias[:,None].expand(-1,self.heads,-1,-1).reshape(
            x.shape[0]*self.heads,x.shape[1],context.tokens.shape[1])
        update,_ = self.context(q,context.tokens,context.tokens,attn_mask=mask,need_weights=False)
        x = x+update
        if not self.scene_first:
            q = self.norms[2](x)
            update,_ = self.scene(q,context.scene_tokens,context.scene_tokens,need_weights=False)
            x = x+update
            scale,bias = self.relation_scale_bias(relations).chunk(2,dim=-1)
            x = x+torch.tanh(self.relation_gate)*(self.norms[3](x)*torch.tanh(scale)+bias)
        x = x+self.ff(self.norms[4](x))
        return x.masked_fill(~frame_mask[...,None],0)


class WholeSequenceDenoiser(nn.Module):
    """Trainable research model. Construction never loads external checkpoints."""
    def __init__(self, config: SequenceModelConfig | None = None):
        super().__init__()
        self.config = (config or SequenceModelConfig()).validate()
        d = self.config.hidden_dim
        self.context_encoder = ContextEncoder(self.config)
        self.motion_encoder = nn.Linear(MOTION_DIM,d)
        self.diffusion_step = mlp(d,d)
        self.phase_encoder = (ParallelFeatureEncoder(4+1, self.config)
                              if self.config.parallel_encoders else
                              progressive_mlp(4+1,d,first_width=16)
                              if self.config.compact_encoders else mlp(4+1,d))
        self.relation_encoder = (ParallelFeatureEncoder(REGIONS*self.config.relation_dim, self.config)
                                 if self.config.parallel_encoders else
                                 progressive_mlp(REGIONS*self.config.relation_dim,d,first_width=128)
                                 if self.config.compact_encoders else mlp(REGIONS*self.config.relation_dim,d))
        self.relation_missing = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([SequenceBlock(self.config) for _ in range(self.config.layers)])
        self.output_norm = nn.LayerNorm(d)
        self.motion_head = nn.Linear(d,MOTION_DIM)
        self.contact_head = nn.Linear(d,REGIONS)
        if self.config.dynamic_timing:
            from .timing_feedback import TimingFeedbackHead
            self.timing_feedback = TimingFeedbackHead(d,min(d,64))
        else:
            self.timing_feedback = None

    def encode(self, condition: SequenceCondition) -> EncodedContext:
        return self.context_encoder(condition)

    def denoise(self, noisy_motion: Tensor, diffusion_steps: Tensor, context: EncodedContext,
                *, schedule: EventSchedule | None = None, relations: Tensor | None = None) -> DenoiserOutput:
        c = context.condition
        b,t,dim = noisy_motion.shape
        if b != c.batch_size or dim != MOTION_DIM or t < c.max_frames or t > self.config.max_frames:
            raise ValueError("noisy_motion shape/capacity differs from condition")
        if diffusion_steps.shape != (b,) or not bool(torch.isfinite(noisy_motion).all()):
            raise ValueError("invalid diffusion steps or nonfinite motion")
        schedule = context.schedule if schedule is None else schedule
        frame_mask = future_mask(c,t)
        x = self.motion_encoder(noisy_motion)
        x = x+positional_encoding(torch.arange(t,device=x.device),self.config.hidden_dim).to(x.dtype)
        step = positional_encoding(diffusion_steps,self.config.hidden_dim).to(x.dtype)
        x = x+self.diffusion_step(step)[:,None]
        frames = torch.arange(1,t+1,device=x.device)[None,:,None]
        boundaries = schedule.boundaries.detach()
        selected = ((frames <= boundaries[:,None]) & c.slot_mask[:,None]).long().argmax(-1)
        rows = torch.arange(b,device=x.device)[:,None]
        starts = torch.cat((torch.zeros_like(boundaries[:,:1]),boundaries[:,:-1]),dim=1)
        progress = ((frames.squeeze(-1)-starts[rows,selected])
                    /schedule.durations.detach()[rows,selected].clamp_min(1)).clamp(0,1)
        event_types = c.slot_types[rows,selected].clamp(0,3)
        phase = torch.cat((torch.nn.functional.one_hot(event_types,4).to(x.dtype),
                           progress.to(x.dtype)[...,None]),dim=-1)
        if not self.config.scene_first:
            x = x+self.phase_encoder(phase)
        key_times = schedule.keypose_times.detach().to(x.dtype)
        token_times = key_times.gather(1,context.token_keyposes.clamp_min(0))
        # Clipped finite bias preserves visibility of distant future goals.
        distance = frames.squeeze(-1).to(x.dtype)[:,:,None]-token_times[:,None]
        bias = -(distance/self.config.time_bias_width).square()/2
        bias = bias.clamp_min(-4.)
        bias = bias.masked_fill((context.token_keyposes < 0)[:,None],0)
        if relations is None:
            rel = self.relation_missing[None,None].expand(b,t,-1)
        else:
            if relations.shape != (b,t,REGIONS,self.config.relation_dim):
                raise ValueError("relations must be B,T,R,relation_dim")
            if not bool(torch.isfinite(relations).all()):
                raise ValueError("nonfinite dynamic relations")
            rel = self.relation_encoder(relations.flatten(-2).to(x.dtype))
        for index,block in enumerate(self.blocks):
            x = block(x,frame_mask,context,bias,rel,
                      phase=phase if self.config.scene_first and index == 0 else None,
                      phase_encoder=self.phase_encoder)
        x = self.output_norm(x)
        clean = self.motion_head(x).masked_fill(~frame_mask[...,None],0)
        contact = self.contact_head(x).masked_fill(~frame_mask[...,None],0)
        return DenoiserOutput(clean,contact,frame_mask)

    def forward(self, noisy_motion, diffusion_steps, condition, *, relations=None):
        context = self.encode(condition)
        return self.denoise(noisy_motion,diffusion_steps,context,relations=relations),context
