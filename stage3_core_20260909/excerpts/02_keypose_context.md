# Keypose、语义变化、部位权重与双向上下文

实际 forward 是 JointContextEncoder.forward。这里也摘出父类 ContextEncoder.__init__，说明四路编码器、fusion、route与duration在哪里创建；父类旧 forward 没有摘作当前逻辑。四路分支读取同一输入，但参数独立，不预设各支语义。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## ParallelFeatureEncoder

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:195)，原文件第 195–211 行。

```python
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
```

## NarrowBodyBranch

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:214)，原文件第 214–229 行。

```python
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
```

## ParallelBodyEncoder

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:232)，原文件第 232–239 行。

```python
class ParallelBodyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branches = nn.ModuleList([
            NarrowBodyBranch(config) for _ in range(config.encoder_branches)])

    def forward(self, features):
        return torch.cat([branch(features) for branch in self.branches], dim=-1)
```

## NarrowContactBranch

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:242)，原文件第 242–253 行。

```python
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
```

## ParallelContactEncoder

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:256)，原文件第 256–264 行。

```python
class ParallelContactEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branches = nn.ModuleList([
            NarrowContactBranch(config) for _ in range(config.encoder_branches)])

    def forward(self, features, target_body_ids):
        return torch.cat([branch(features, target_body_ids)
                          for branch in self.branches], dim=-1)
```

## ContextEncoder.__init__

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:323)，原文件第 323–365 行。

```python
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
```

## LearnedPartImportance

来源：[hsi/stage3_joint/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/model.py:144)，原文件第 144–165 行。

```python
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
```

## JointContextEncoder

来源：[hsi/stage3_joint/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/model.py:168)，原文件第 168–315 行。

```python
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
```
