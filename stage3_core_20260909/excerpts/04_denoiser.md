# 16层整序列扩散去噪主干

实际使用基础 WholeSequenceDenoiser.denoise；JointSequenceDenoiser 替换上下文编码器并扩展后8层。事件时间在场景/动态关系之后引入。每轮输出整个[B,T,135]干净动作和8部位接触logits，不是逐帧或两姿态插值。构造函数不自动加载权重，实际large配置另见model_config.json。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## SequenceBlock

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:462)，原文件第 462–507 行。

```python
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
```

## WholeSequenceDenoiser

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:510)，原文件第 510–593 行。

```python
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
```

## ExpandedSequenceBlock

来源：[hsi/stage3_joint/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/model.py:318)，原文件第 318–333 行。

```python
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
```

## JointSequenceDenoiser

来源：[hsi/stage3_joint/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/model.py:336)，原文件第 336–353 行。

```python
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
```
