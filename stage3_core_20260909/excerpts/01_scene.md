# 场景：四路窄维编码与局部几何聚合

R4 每支先在128维计算局部几何、池化为32×128，然后四支拼接成32×512。JointContextEncoder 初始化时把每支 local 替换成 SpatialNeighborhoodAggregation；不要将旧 LocalPointAggregation 当成当前实现。SceneEncoder 中未启用的 legacy 分支按原文保留。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## NarrowSceneBranch

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:267)，原文件第 267–287 行。

```python
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
```

## SceneEncoder

来源：[hsi/stage3_sequence/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_sequence/model.py:290)，原文件第 290–319 行。

```python
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
```

## SpatialNeighborhoodAggregation

来源：[hsi/stage3_joint/model.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/model.py:93)，原文件第 93–141 行。

```python
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
```
