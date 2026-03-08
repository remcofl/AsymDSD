from abc import ABC
from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components import FactoryConfig
from .activation import ActivationLayer, is_gated_activation
from .drop_path import DropPath, drop_path_efficient
from .layer_scale import LayerScale
from .multilayer_perceptron import MLP
from .normalization import NormalizationLayer


@dataclass
class TransformerBaseConfig(FactoryConfig, ABC):
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 12
    hidden_ratio: float = 4.0
    norm_layer: NormalizationLayer = nn.LayerNorm
    act_layer: ActivationLayer = nn.GELU
    dropout_p: float = 0.0
    drop_path_p: float = 0.0
    uniform_drop_path: bool = False
    efficient_drop_path: bool = True
    add_pos_enc_every_layer: bool = False
    layer_scale_init: float | None = None
    bias: bool = True
    allow_grad_ckpt: bool = False


@dataclass
class TransformerModConfig(TransformerBaseConfig):
    self_attention: bool = True
    cross_attention: bool = True
    concat_tgt_memory: bool = False

    def __post_init__(self):
        if not self.self_attention and not self.cross_attention:
            raise ValueError(
                "At least one of self_attention or cross_attention must be True"
            )

    @property
    def CLS(self):
        return TransformerModule


@dataclass
class TransformerEncoderConfig(TransformerBaseConfig):
    @property
    def CLS(self):
        return TransformerEncoder

    def instantiate(self) -> "TransformerEncoder":
        return self.CLS(self)


@dataclass
class TransformerDecoderConfig(TransformerBaseConfig):
    self_attention: bool = True
    concat_tgt_memory: bool = False

    @property
    def CLS(self):
        return TransformerDecoder

    def instantiate(self) -> "TransformerDecoder":
        return self.CLS(self)


class TransformerOutput(NamedTuple):
    x: torch.Tensor
    attn_weights: list[torch.Tensor] | None = None
    hidden_states: list[torch.Tensor] | None = None


class FFN(nn.Module):
    def __init__(self, config: TransformerModConfig) -> None:
        super().__init__()
        self.norm = config.norm_layer(config.embed_dim)

        hidden_dim = int(config.hidden_ratio * config.embed_dim)
        if is_gated_activation(config.act_layer):
            # To keep the same number of parameters
            hidden_dim = int(hidden_dim * 2 / 3)
            # Round up to nearest factor of 8 to be Tensor Core-friendly
            hidden_dim += (-hidden_dim) % 8

        self.mlp = MLP(
            in_dim=config.embed_dim,
            hidden_dim=hidden_dim,
            norm_layer=None,
            act_layer=config.act_layer,
            dropout_p=config.dropout_p,
            bias=config.bias,
        )

        init_val = config.layer_scale_init
        self.layer_scale = (
            LayerScale(config.embed_dim, init_val) if init_val else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        x = self.norm(x)
        x = self.mlp(x)
        x = self.layer_scale(x)
        x = self.dropout(x)
        # Note: Does not apply residual connection
        return x


class Attention(nn.Module):
    def __init__(self, config: TransformerModConfig) -> None:
        super().__init__()
        self.norm = config.norm_layer(config.embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout_p,  # Could use separate attn_dropout_p
            bias=config.bias,
            batch_first=True,
        )
        self.concat_tgt_memory = config.concat_tgt_memory

        init_val = config.layer_scale_init
        self.layer_scale = (
            LayerScale(config.embed_dim, init_val) if init_val else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.norm(x)

        if memory is None:
            q = k = v = x
        else:
            q = x
            if self.concat_tgt_memory:
                k = v = torch.concat((memory, x), dim=1)
            else:
                k = v = memory

        x, attn_weights = self.attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.layer_scale(x)
        x = self.dropout(x)
        return (x, attn_weights) if return_attention else (x, None)


class Block(nn.Module):
    def __init__(self, config: TransformerModConfig, drop_path_p: float = 0.0) -> None:
        super().__init__()
        self.drop_path_p = drop_path_p
        self.efficient_drop_path = config.efficient_drop_path

        self.self_attn = Attention(config) if config.self_attention else None
        self.cross_attn = Attention(config) if config.cross_attention else None
        self.ffn = FFN(config)

        self.drop_path = DropPath(drop_path_p)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if not self.training or self.drop_path_p == 0.0:
            attn_weights = []

            if self.self_attn:
                self_attn_out = self.self_attn(
                    x,
                    attn_mask=self_mask,
                    key_padding_mask=self_key_padding_mask,
                    return_attention=return_attention,
                )
                x = x + self_attn_out[0]
                attn_weights.append(self_attn_out[1])

            if self.cross_attn:
                cross_attn_out = self.cross_attn(
                    x,
                    memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                    return_attention=return_attention,
                )
                x = x + cross_attn_out[0]
                attn_weights.append(cross_attn_out[1])

            x = x + self.ffn(x)
            return (x, attn_weights)
        elif self.drop_path_p < 0.1 or not self.efficient_drop_path:
            attn_weights = []

            if self.self_attn:
                self_attn_out = self.self_attn(
                    x,
                    attn_mask=self_mask,
                    key_padding_mask=self_key_padding_mask,
                    return_attention=return_attention,
                )
                x = x + self.drop_path(self_attn_out[0])
                attn_weights.append(self_attn_out[1])

            if self.cross_attn:
                cross_attn_out = self.cross_attn(
                    x,
                    memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                    return_attention=return_attention,
                )
                x = x + self.drop_path(cross_attn_out[0])
                attn_weights.append(cross_attn_out[1])

            x = x + self.drop_path(self.ffn(x))
            return (x, attn_weights)
        else:
            if self.self_attn:
                x = drop_path_efficient(
                    x,
                    path_fn=lambda *args, **kwargs: self.self_attn(*args, **kwargs)[0],  # type: ignore
                    drop_p=self.drop_path_p,
                    training=self.training,
                    residual_add=True,
                    attn_mask=self_mask,
                    key_padding_mask=self_key_padding_mask,
                )
            if self.cross_attn:
                x = drop_path_efficient(
                    x,
                    memory,
                    path_fn=lambda *args, **kwargs: self.cross_attn(*args, **kwargs)[0],  # type: ignore
                    drop_p=self.drop_path_p,
                    training=self.training,
                    residual_add=True,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                )
            x = drop_path_efficient(
                x,
                path_fn=lambda *args, **kwargs: self.ffn(*args, **kwargs),
                drop_p=self.drop_path_p,
                training=self.training,
                residual_add=True,
            )
            return x, None


class TransformerModule(nn.Module):
    def __init__(self, config: TransformerModConfig) -> None:
        super().__init__()
        self.config = config
        self.add_pos_enc = config.add_pos_enc_every_layer
        self.dropout = nn.Dropout(config.dropout_p)

        if config.uniform_drop_path:
            drop_path_layer = [config.drop_path_p] * config.num_layers
        else:
            drop_path_layer = [
                p for p in np.linspace(0.0, config.drop_path_p, config.num_layers)
            ]

        block_list = [
            Block(config, drop_path_layer[layer_i])
            for layer_i in range(config.num_layers)
        ]

        self.stack = nn.ModuleList(block_list)

        self.norm = config.norm_layer(config.embed_dim)

        self._gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        if not self.add_pos_enc:
            x = x + pos_enc

        attn_weights = [] if return_attention else None
        hidden_states = [] if return_hidden_states else None

        for block in self.stack:
            if self.add_pos_enc:
                x = x + pos_enc

            if self._gradient_checkpointing:
                # Do not use lambda on variables for ckpt func
                block_out: tuple[torch.Tensor] = checkpoint(  # type: ignore
                    block,
                    x,
                    memory,
                    self_mask=self_mask,
                    memory_mask=memory_mask,
                    self_key_padding_mask=self_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=return_attention,
                    use_reentrant=False,
                )
            else:
                block_out = block(
                    x,
                    memory,
                    self_mask=self_mask,
                    memory_mask=memory_mask,
                    self_key_padding_mask=self_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=return_attention,
                )

            x = block_out[0]

            if return_attention:
                attn_weights.extend(block_out[1])  # type: ignore
            if return_hidden_states:
                hidden_states.append(x)  # type: ignore

        x = self.norm(x)

        return TransformerOutput(
            x=x, attn_weights=attn_weights, hidden_states=hidden_states
        )

    def enable_gradient_checkpointing(self) -> None:
        if self.config.allow_grad_ckpt:
            self._gradient_checkpointing = True


class TransformerEncoder(TransformerModule):
    def __init__(self, config: TransformerEncoderConfig) -> None:
        self.original_config = config
        cfg = TransformerModConfig(
            **(asdict(config)), self_attention=True, cross_attention=False
        )
        super().__init__(cfg)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        *,
        self_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        return super().forward(
            x,
            pos_enc,
            self_mask=self_mask,
            self_key_padding_mask=self_key_padding_mask,
            return_attention=return_attention,
            return_hidden_states=return_hidden_states,
        )


class TransformerDecoder(TransformerModule):
    def __init__(self, config: TransformerDecoderConfig) -> None:
        self.original_config = config
        cfg = TransformerModConfig(**asdict(config), cross_attention=True)
        super().__init__(cfg)
