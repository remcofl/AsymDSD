from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto

import torch
from jsonargparse import lazy_instance
from torch import nn

from ..components import FactoryConfig
from ..components.common_types import OptionalTensor
from .multilayer_perceptron import MLPConfig, MLPVarLen
from .tokenization import TrainableToken


class ClassificationHeadType(StrEnum):
    LINEAR = auto()
    MLP = auto()


DEFAULT_CLS_HEAD_CONFIG = lazy_instance(
    MLPConfig,
    dims=[256, 256],
    dropout_p=0.5,
    norm_layer=nn.BatchNorm1d,
    bias=False,
)


@dataclass
class ClassificationHeadConfig(FactoryConfig):
    num_classes: int
    embed_dim: int = 384
    map_avg_pooling: bool = True
    map_max_pooling: bool = False
    map_cls_token: bool = False
    map_attn_pooling: bool | int = False
    classification_head_type: ClassificationHeadType = ClassificationHeadType.LINEAR
    mlp_head_config: MLPConfig | None = field(
        default_factory=lambda: DEFAULT_CLS_HEAD_CONFIG
    )

    @property
    def CLS(self):
        return ClassificationHead

    def instantiate(self) -> "ClassificationHead":
        return self.CLS(**asdict(self))


class ClassificationHead(nn.Module):
    """
    Classification head for the model.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 384,
        map_avg_pooling: bool = True,
        map_max_pooling: bool = False,
        map_cls_token: bool = False,
        map_attn_pooling: bool | int = False,
        classification_head_type: ClassificationHeadType = ClassificationHeadType.LINEAR,
        mlp_head_config: MLPConfig | None = DEFAULT_CLS_HEAD_CONFIG,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.map_avg_pooling = map_avg_pooling
        self.map_max_pooling = map_max_pooling
        self.map_cls_token = map_cls_token
        self.map_attn_pooling = map_attn_pooling

        self.classification_head_type = classification_head_type
        self.mlp_head_config = mlp_head_config

        if not (
            map_avg_pooling
            or map_max_pooling
            or map_cls_token
            or map_attn_pooling is not False
        ):
            raise ValueError(
                "At least one of map_avg_pooling, map_max_pooling, or map_cls_token must be True"
            )

        if (
            classification_head_type == ClassificationHeadType.MLP
            and mlp_head_config is None
        ):
            raise ValueError(
                "mlp_head_config must be specified when classification_mode is MLP."
            )

        self._init_classification_head(self.num_classes)

    def _init_classification_head(self, num_classes: int):
        input_dim = (
            int(self.map_cls_token)
            + int(self.map_avg_pooling)
            + int(self.map_max_pooling)
            + int(self.map_attn_pooling is not False)
        ) * self.embed_dim

        if self.classification_head_type == ClassificationHeadType.LINEAR:
            self.head = MLPVarLen(input_dim, num_classes, bias=True)
        elif self.classification_head_type == ClassificationHeadType.MLP:
            cfg: MLPConfig = self.mlp_head_config  # type: ignore
            self.head = MLPVarLen(
                *([input_dim] + cfg.dims + [num_classes]),
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
                dropout_p=cfg.dropout_p,
                bias=cfg.bias,
            )

        if self.map_attn_pooling is not False:
            num_heads = (
                self.map_attn_pooling if isinstance(self.map_attn_pooling, int) else 1
            )
            self.cls_token = TrainableToken(embed_dim=self.embed_dim)

            self.attention = nn.MultiheadAttention(
                self.embed_dim,
                num_heads=num_heads,
                # dropout=cfg.dropout_p,
                bias=cfg.bias,
                batch_first=True,
            )

    def forward(
        self,
        cls_features: OptionalTensor = None,
        patch_features: OptionalTensor = None,
    ) -> torch.Tensor:
        if not self.map_cls_token and cls_features is not None:
            all_embeddings = torch.cat(
                (cls_features.unsqueeze(1), patch_features),  # type: ignore
                dim=1,
            )
        else:
            all_embeddings = patch_features

        features = []
        if self.map_cls_token:
            features.append(cls_features)
        if self.map_avg_pooling:
            features.append(all_embeddings.mean(dim=1))  # type: ignore
        if self.map_max_pooling:
            features.append(all_embeddings.amax(dim=1))  # type: ignore
        if self.map_attn_pooling:
            patch_features = patch_features
            cls_token = self.cls_token.expand(patch_features.shape[0], 1, -1)  # type: ignore
            x, _ = self.attention(cls_token, patch_features, patch_features)
            features.append(x.squeeze(1))  # type: ignore

        x = torch.cat(features, dim=-1)
        x = self.head(x)

        return x  # type: ignore
