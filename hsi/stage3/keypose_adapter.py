"""Ordered sparse-keypose conditioning for a ReMoGen-style denoiser.

This module deliberately depends only on PyTorch.  It mirrors the tensor layout
used by ReMoGen's control blocks (batch-first hidden states at the point of
injection), while keeping the keypose representation independent of the full
ReMoGen environment.

Tensor contract
---------------
``KeyposePacket`` describes an *ordered*, frame-free sequence of partial poses:

* ``root_xyz_yaw``: ``[B, K, 4]`` ``(x, y, z, yaw)`` in the caller-declared
  metric frame.  The ReMoGen integration requires its active canonical-local
  frame; a GenZI world-space result must be transformed before encoding;
* ``joint_xyz``: ``[B, K, J, 3]`` pelvis-relative joint positions;
* ``joint_mask``: ``[B, K, J]``; ``True`` means that joint is constrained;
* ``semantic_id``: ``[B, K]`` integer semantic phase/event identifiers;
* ``ordinal``: ``[B, K]`` integer logical ranks, not absolute frame indices;
* ``confidence``: ``[B, K]`` node confidence in ``[0, 1]``;
* ``node_mask``: ``[B, K]``; ``True`` means that keypose node exists;
* ``joint_confidence``: optional ``[B, K, J]`` confidence in ``[0, 1]``.

The encoder emits node-major tokens.  Every node owns one root token followed
by ``J`` possible joint tokens.  Invalid tokens are zeroed and marked by a
PyTorch-style key padding mask, where ``True`` means "ignore".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


def _ensure_shape(name: str, value: Tensor, expected: Tuple[int, ...]) -> None:
    if tuple(value.shape) != tuple(expected):
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(value.shape)}"
        )


def _check_finite_on_mask(name: str, value: Tensor, mask: Tensor) -> None:
    expanded_mask = mask
    while expanded_mask.ndim < value.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(value)
    selected = value.masked_select(expanded_mask)
    if selected.numel() and not torch.isfinite(selected).all().item():
        raise ValueError(f"{name} contains a non-finite value at a valid entry")


def _check_unit_interval_on_mask(name: str, value: Tensor, mask: Tensor) -> None:
    selected = value.masked_select(mask)
    if selected.numel() and ((selected < 0).any() or (selected > 1).any()):
        raise ValueError(f"{name} must be in [0, 1] at valid entries")


@dataclass(frozen=True)
class KeyposePacket:
    """Batched ordered partial-keypose packet.

    ``ordinal`` supplies order only.  It must be non-negative for valid nodes,
    but it need not encode a frame number or be contiguous.  Storage order may
    differ from logical order; the ordinal embedding carries the latter.
    """

    root_xyz_yaw: Tensor
    joint_xyz: Tensor
    joint_mask: Tensor
    semantic_id: Tensor
    ordinal: Tensor
    confidence: Tensor
    node_mask: Tensor
    joint_confidence: Optional[Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.root_xyz_yaw.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.root_xyz_yaw.shape[1])

    @property
    def num_joints(self) -> int:
        return int(self.joint_xyz.shape[2])

    @property
    def device(self) -> torch.device:
        return self.root_xyz_yaw.device

    def validate(self, require_sorted_ordinals: bool = False) -> "KeyposePacket":
        if self.root_xyz_yaw.ndim != 3 or self.root_xyz_yaw.shape[-1] != 4:
            raise ValueError(
                "root_xyz_yaw must have shape [B, K, 4], got "
                f"{tuple(self.root_xyz_yaw.shape)}"
            )
        if self.joint_xyz.ndim != 4 or self.joint_xyz.shape[-1] != 3:
            raise ValueError(
                "joint_xyz must have shape [B, K, J, 3], got "
                f"{tuple(self.joint_xyz.shape)}"
            )
        batch, nodes, _ = self.root_xyz_yaw.shape
        if batch < 1 or nodes < 1 or self.joint_xyz.shape[2] < 1:
            raise ValueError("B, K, and J must all be positive")
        joints = self.joint_xyz.shape[2]
        _ensure_shape("joint_xyz", self.joint_xyz, (batch, nodes, joints, 3))
        _ensure_shape("joint_mask", self.joint_mask, (batch, nodes, joints))
        _ensure_shape("semantic_id", self.semantic_id, (batch, nodes))
        _ensure_shape("ordinal", self.ordinal, (batch, nodes))
        _ensure_shape("confidence", self.confidence, (batch, nodes))
        _ensure_shape("node_mask", self.node_mask, (batch, nodes))
        if self.joint_confidence is not None:
            _ensure_shape(
                "joint_confidence", self.joint_confidence, (batch, nodes, joints)
            )

        tensors = (
            self.joint_xyz,
            self.joint_mask,
            self.semantic_id,
            self.ordinal,
            self.confidence,
            self.node_mask,
        )
        if self.joint_confidence is not None:
            tensors = tensors + (self.joint_confidence,)
        if any(t.device != self.device for t in tensors):
            raise ValueError("all KeyposePacket tensors must be on the same device")
        if self.joint_mask.dtype != torch.bool or self.node_mask.dtype != torch.bool:
            raise TypeError("joint_mask and node_mask must use torch.bool")
        if self.semantic_id.dtype != torch.long or self.ordinal.dtype != torch.long:
            raise TypeError("semantic_id and ordinal must use torch.long")
        if not self.root_xyz_yaw.is_floating_point() or not self.joint_xyz.is_floating_point():
            raise TypeError("root_xyz_yaw and joint_xyz must be floating-point tensors")
        if not self.confidence.is_floating_point():
            raise TypeError("confidence must be a floating-point tensor")
        if self.joint_confidence is not None and not self.joint_confidence.is_floating_point():
            raise TypeError("joint_confidence must be a floating-point tensor")

        effective_joint_mask = self.joint_mask & self.node_mask.unsqueeze(-1)
        _check_finite_on_mask("root_xyz_yaw", self.root_xyz_yaw, self.node_mask)
        _check_finite_on_mask("joint_xyz", self.joint_xyz, effective_joint_mask)
        _check_finite_on_mask("confidence", self.confidence, self.node_mask)
        _check_unit_interval_on_mask("confidence", self.confidence, self.node_mask)
        if self.joint_confidence is not None:
            _check_finite_on_mask(
                "joint_confidence", self.joint_confidence, effective_joint_mask
            )
            _check_unit_interval_on_mask(
                "joint_confidence", self.joint_confidence, effective_joint_mask
            )
        if (self.semantic_id.masked_select(self.node_mask) < 0).any():
            raise ValueError("semantic_id must be non-negative at valid nodes")
        if (self.ordinal.masked_select(self.node_mask) < 0).any():
            raise ValueError("ordinal must be non-negative at valid nodes")

        if require_sorted_ordinals:
            for batch_idx in range(batch):
                order = self.ordinal[batch_idx].masked_select(self.node_mask[batch_idx])
                if order.numel() > 1 and not torch.all(order[1:] > order[:-1]).item():
                    raise ValueError(
                        "valid ordinals must be strictly increasing in storage order"
                    )
        return self

    def to(self, *args, **kwargs) -> "KeyposePacket":
        """Return a packet moved/cast like ``Tensor.to`` while preserving ID dtypes."""

        floating_names = ("root_xyz_yaw", "joint_xyz", "confidence")
        updates = {name: getattr(self, name).to(*args, **kwargs) for name in floating_names}
        # Masks and IDs should move device, but must never inherit a requested float dtype.
        target_device = updates["root_xyz_yaw"].device
        updates.update(
            joint_mask=self.joint_mask.to(device=target_device),
            semantic_id=self.semantic_id.to(device=target_device),
            ordinal=self.ordinal.to(device=target_device),
            node_mask=self.node_mask.to(device=target_device),
            joint_confidence=(
                None
                if self.joint_confidence is None
                else self.joint_confidence.to(*args, **kwargs)
            ),
        )
        return replace(self, **updates)

    @classmethod
    def from_unbatched(
        cls,
        root_xyz_yaw: Tensor,
        joint_xyz: Tensor,
        joint_mask: Tensor,
        semantic_id: Tensor,
        ordinal: Tensor,
        confidence: Tensor,
        node_mask: Tensor,
        joint_confidence: Optional[Tensor] = None,
    ) -> "KeyposePacket":
        """Add the batch axis to tensors shaped ``[K, ...]``."""

        return cls(
            root_xyz_yaw=root_xyz_yaw.unsqueeze(0),
            joint_xyz=joint_xyz.unsqueeze(0),
            joint_mask=joint_mask.unsqueeze(0),
            semantic_id=semantic_id.unsqueeze(0),
            ordinal=ordinal.unsqueeze(0),
            confidence=confidence.unsqueeze(0),
            node_mask=node_mask.unsqueeze(0),
            joint_confidence=(
                None if joint_confidence is None else joint_confidence.unsqueeze(0)
            ),
        )


@dataclass(frozen=True)
class OrderedKeyposeEncoding:
    """Encoded control sequence and metadata.

    ``key_padding_mask`` follows PyTorch/ReMoGen convention: ``True`` entries
    are ignored by attention.
    """

    tokens: Tensor
    key_padding_mask: Tensor
    has_condition: Tensor
    token_node_index: Tensor
    token_joint_index: Tensor
    token_ordinal: Tensor

    @property
    def valid_token_count(self) -> Tensor:
        return (~self.key_padding_mask).sum(dim=-1)


class _ScalarEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value.unsqueeze(-1))


class OrderedPartialKeyposeEncoder(nn.Module):
    """Encode variable sparse keypose nodes into a padded control sequence."""

    def __init__(
        self,
        d_model: int,
        num_semantics: int,
        max_ordinals: int,
        max_joints: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(d_model, num_semantics, max_ordinals, max_joints) <= 0:
            raise ValueError("all encoder dimensions must be positive")
        self.d_model = int(d_model)
        self.num_semantics = int(num_semantics)
        self.max_ordinals = int(max_ordinals)
        self.max_joints = int(max_joints)

        # Yaw is periodic, so the four-value packet becomes xyz + sin/cos(yaw).
        # Unlike the original HSI GoalEncoder, no spatial coordinate is dropped.
        self.root_encoder = nn.Sequential(
            nn.Linear(5, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.semantic_embedding = nn.Embedding(num_semantics, d_model)
        self.ordinal_embedding = nn.Embedding(max_ordinals, d_model)
        self.joint_id_embedding = nn.Embedding(max_joints, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)  # root, joint
        self.node_confidence_embedding = _ScalarEmbedding(d_model)
        self.joint_confidence_embedding = _ScalarEmbedding(d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, packet: KeyposePacket) -> OrderedKeyposeEncoding:
        packet.validate()
        batch, nodes, _ = packet.root_xyz_yaw.shape
        joints = packet.num_joints
        if joints > self.max_joints:
            raise ValueError(
                f"packet has {joints} joints, encoder supports {self.max_joints}"
            )

        valid_nodes = packet.node_mask
        valid_joints = packet.joint_mask & valid_nodes.unsqueeze(-1)
        valid_semantics = packet.semantic_id.masked_select(valid_nodes)
        valid_ordinals = packet.ordinal.masked_select(valid_nodes)
        if valid_semantics.numel() and valid_semantics.max().item() >= self.num_semantics:
            raise ValueError(
                f"semantic_id exceeds encoder vocabulary ({self.num_semantics})"
            )
        if valid_ordinals.numel() and valid_ordinals.max().item() >= self.max_ordinals:
            raise ValueError(
                f"ordinal exceeds encoder range ({self.max_ordinals})"
            )

        zero_root = torch.zeros_like(packet.root_xyz_yaw)
        safe_root = torch.where(valid_nodes.unsqueeze(-1), packet.root_xyz_yaw, zero_root)
        safe_xyz = safe_root[..., :3]
        safe_yaw = safe_root[..., 3:4]
        periodic_root = torch.cat(
            (safe_xyz, torch.sin(safe_yaw), torch.cos(safe_yaw)), dim=-1
        )
        safe_joint = torch.where(
            valid_joints.unsqueeze(-1), packet.joint_xyz, torch.zeros_like(packet.joint_xyz)
        )
        safe_semantic = torch.where(
            valid_nodes, packet.semantic_id, torch.zeros_like(packet.semantic_id)
        )
        safe_ordinal = torch.where(
            valid_nodes, packet.ordinal, torch.zeros_like(packet.ordinal)
        )
        safe_node_confidence = torch.where(
            valid_nodes, packet.confidence, torch.zeros_like(packet.confidence)
        ).clamp(0.0, 1.0)
        if packet.joint_confidence is None:
            safe_joint_confidence = safe_node_confidence.unsqueeze(-1).expand(
                batch, nodes, joints
            )
        else:
            safe_joint_confidence = torch.where(
                valid_joints,
                packet.joint_confidence,
                torch.zeros_like(packet.joint_confidence),
            ).clamp(0.0, 1.0)

        semantic = self.semantic_embedding(safe_semantic)
        ordinal = self.ordinal_embedding(safe_ordinal)
        node_confidence = self.node_confidence_embedding(safe_node_confidence)
        node_context = semantic + ordinal + node_confidence

        root_type = self.token_type_embedding.weight[0].view(1, 1, -1)
        root_tokens = self.root_encoder(periodic_root) + node_context + root_type

        joint_ids = torch.arange(joints, device=packet.device, dtype=torch.long)
        joint_identity = self.joint_id_embedding(joint_ids).view(1, 1, joints, -1)
        joint_type = self.token_type_embedding.weight[1].view(1, 1, 1, -1)
        joint_tokens = (
            self.joint_encoder(safe_joint)
            + node_context.unsqueeze(2)
            + joint_identity
            + self.joint_confidence_embedding(safe_joint_confidence)
            + joint_type
        )

        # Node-major layout: root_0, joint_0_0...joint_0_J, root_1, ...
        tokens = torch.cat((root_tokens.unsqueeze(2), joint_tokens), dim=2)
        tokens = self.output(tokens).reshape(batch, nodes * (joints + 1), self.d_model)
        valid_tokens = torch.cat((valid_nodes.unsqueeze(-1), valid_joints), dim=-1)
        valid_tokens = valid_tokens.reshape(batch, nodes * (joints + 1))
        tokens = tokens.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        key_padding_mask = ~valid_tokens

        node_index = (
            torch.arange(nodes, device=packet.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(nodes, joints + 1)
            .reshape(-1)
        )
        joint_index_per_node = torch.cat(
            (
                torch.full((1,), -1, device=packet.device, dtype=torch.long),
                joint_ids,
            )
        )
        token_joint_index = joint_index_per_node.repeat(nodes)
        token_ordinal = (
            safe_ordinal.unsqueeze(-1)
            .expand(batch, nodes, joints + 1)
            .reshape(batch, -1)
        )
        token_ordinal = token_ordinal.masked_fill(~valid_tokens, -1)
        return OrderedKeyposeEncoding(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            has_condition=valid_nodes.any(dim=-1),
            token_node_index=node_index,
            token_joint_index=token_joint_index,
            token_ordinal=token_ordinal,
        )


class _NativeCrossAttentionAdapter(nn.Module):
    """Pure-PyTorch residual cross-attention fallback."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        initial_residual_scale: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        # Small non-zero initialization preserves gradients into the keypose
        # encoder during the first optimization step.
        self.cross_scale = nn.Parameter(torch.tensor(float(initial_residual_scale)))
        self.ff_scale = nn.Parameter(torch.tensor(float(initial_residual_scale)))

    def forward(
        self,
        hidden_states: Tensor,
        encoding: OrderedKeyposeEncoding,
        return_attention: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        safe_mask = encoding.key_padding_mask.clone()
        safe_context = encoding.tokens
        empty = ~encoding.has_condition
        if empty.any():
            # MultiheadAttention produces NaNs for an all-masked row.  Unmask a
            # zero token for computation, then restore the input exactly below.
            safe_mask[empty, 0] = False
            safe_context = safe_context.clone()
            safe_context[empty, 0] = 0.0

        delta, attention = self.cross_attention(
            query=self.hidden_norm(hidden_states),
            key=self.context_norm(safe_context),
            value=self.context_norm(safe_context),
            key_padding_mask=safe_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        output = hidden_states + self.cross_scale * self.cross_dropout(delta)
        output = output + self.ff_scale * self.ff(self.ff_norm(output))
        output = torch.where(
            encoding.has_condition[:, None, None], output, hidden_states
        )
        if attention is not None and empty.any():
            attention = attention.masked_fill(empty[:, None, None, None], 0.0)
        return output, attention


class _OfficialReMoGenAdapter(nn.Module):
    """Lazy wrapper around ReMoGen's InterAdapterBasicTransformerBlock."""

    def __init__(
        self,
        block_class,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        dropout: float,
        initial_residual_scale: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.block = block_class(
            dim=hidden_dim,
            num_attention_heads=num_heads,
            attention_head_dim=hidden_dim // num_heads,
            cross_attention_dim=context_dim,
            dropout=dropout,
            rel_bias_dim=48,
            film=True,
        )
        # The released ReMoGen block is itself a full residual transformer
        # whose three internal residual weights start at 1.  Applying it
        # directly would therefore replace a sizeable part of the frozen
        # motion prior with a randomly initialized branch on the first step.
        # Gate the *whole* block delta, matching the small-residual contract of
        # the native backend.
        self.residual_scale = nn.Parameter(
            torch.tensor(float(initial_residual_scale))
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoding: OrderedKeyposeEncoding,
        return_attention: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        del return_attention  # The official block does not expose attention weights.
        safe_mask = encoding.key_padding_mask.clone()
        safe_context = encoding.tokens
        empty = ~encoding.has_condition
        if empty.any():
            safe_mask[empty, 0] = False
            safe_context = safe_context.clone()
            safe_context[empty, 0] = 0.0
        _block_output, delta = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=safe_context,
            attention_mask=None,
            encoder_attention_mask=safe_mask,
            update_ema=self.training,
            return_delta=True,
        )
        output = hidden_states + self.residual_scale * delta
        output = torch.where(
            encoding.has_condition[:, None, None], output, hidden_states
        )
        return output, None


def _find_official_remogen_block():
    """Load ReMoGen's block only when its source is already on ``PYTHONPATH``."""

    module = importlib.import_module("model.fusion.blocks")
    return getattr(module, "InterAdapterBasicTransformerBlock")


@dataclass(frozen=True)
class KeyposeAdapterOutput:
    hidden_states: Tensor
    encoding: OrderedKeyposeEncoding
    attention_weights: Optional[Tensor]


class ReMoGenOrderedKeyposeAdapter(nn.Module):
    """Encode a packet and inject it into a ReMoGen hidden sequence.

    Args:
        backend: ``"native"`` uses only PyTorch; ``"remogen"`` requires
            ReMoGen's repository on ``PYTHONPATH``; ``"auto"`` tries the
            official block and otherwise falls back to the native block.

    ``forward`` consumes ``[B, S, D]`` by default.  Set ``batch_first=False``
    or call ``forward_remogen`` for ReMoGen's external ``[S, B, D]`` layout.
    """

    def __init__(
        self,
        hidden_dim: int,
        keypose_dim: int,
        num_semantics: int,
        max_ordinals: int,
        max_joints: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        backend: str = "auto",
        initial_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.packet_encoder = OrderedPartialKeyposeEncoder(
            d_model=keypose_dim,
            num_semantics=num_semantics,
            max_ordinals=max_ordinals,
            max_joints=max_joints,
            dropout=dropout,
        )
        backend = backend.lower()
        if backend not in {"auto", "native", "remogen"}:
            raise ValueError("backend must be one of: auto, native, remogen")

        selected_backend = backend
        block_class = None
        if backend in {"auto", "remogen"}:
            try:
                block_class = _find_official_remogen_block()
                selected_backend = "remogen"
            except (ImportError, AttributeError, ModuleNotFoundError) as error:
                if backend == "remogen":
                    raise RuntimeError(
                        "ReMoGen InterAdapterBasicTransformerBlock is unavailable; "
                        "put the ReMoGen repository on PYTHONPATH or use backend='native'"
                    ) from error
                selected_backend = "native"

        if selected_backend == "remogen":
            self.cross_attention_adapter = _OfficialReMoGenAdapter(
                block_class=block_class,
                hidden_dim=hidden_dim,
                context_dim=keypose_dim,
                num_heads=num_heads,
                dropout=dropout,
                initial_residual_scale=initial_residual_scale,
            )
        else:
            self.cross_attention_adapter = _NativeCrossAttentionAdapter(
                hidden_dim=hidden_dim,
                context_dim=keypose_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
                initial_residual_scale=initial_residual_scale,
            )
        self.backend = selected_backend
        self.hidden_dim = int(hidden_dim)

    def encode(self, packet: KeyposePacket) -> OrderedKeyposeEncoding:
        return self.packet_encoder(packet)

    def inject(
        self,
        hidden_states: Tensor,
        encoding: OrderedKeyposeEncoding,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_states must have shape [B, S, {self.hidden_dim}], got "
                f"{tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] != encoding.tokens.shape[0]:
            raise ValueError("hidden_states and encoding batch sizes differ")
        return self.cross_attention_adapter(
            hidden_states, encoding, return_attention=return_attention
        )

    def forward(
        self,
        hidden_states: Tensor,
        packet: KeyposePacket,
        batch_first: bool = True,
        return_details: bool = False,
        return_attention: bool = False,
    ) -> Union[Tensor, KeyposeAdapterOutput]:
        hidden_bsd = hidden_states if batch_first else hidden_states.permute(1, 0, 2)
        encoding = self.encode(packet)
        output_bsd, attention = self.inject(
            hidden_bsd, encoding, return_attention=return_attention
        )
        output = output_bsd if batch_first else output_bsd.permute(1, 0, 2)
        if return_details:
            return KeyposeAdapterOutput(
                hidden_states=output,
                encoding=encoding,
                attention_weights=attention,
            )
        return output

    def forward_remogen(
        self,
        hidden_states_sbd: Tensor,
        packet: KeyposePacket,
        return_details: bool = False,
        return_attention: bool = False,
    ) -> Union[Tensor, KeyposeAdapterOutput]:
        """Convenience wrapper for ReMoGen's ``[S, B, D]`` hidden layout."""

        return self.forward(
            hidden_states_sbd,
            packet,
            batch_first=False,
            return_details=return_details,
            return_attention=return_attention,
        )


__all__ = [
    "KeyposePacket",
    "OrderedKeyposeEncoding",
    "OrderedPartialKeyposeEncoder",
    "KeyposeAdapterOutput",
    "ReMoGenOrderedKeyposeAdapter",
]

