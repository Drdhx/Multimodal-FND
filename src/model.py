"""
Multimodal Fake News Detection — Model Architecture
====================================================
DistilBERT (Text) + EfficientNetV2S (Image) + Custom MLP (Metadata)
Fused via Gated Cross-Modal Attention

Author  : Dhanush D
Email   : dhanushd1812@gmail.com
GitHub  : github.com/Drdhx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
import timm
from typing import List


class TextEncoder(nn.Module):
    """
    DistilBERT text encoder.
    Input  : tokenized headline (input_ids, attention_mask)
    Output : projected [CLS] embedding, shape [B, out_dim]
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [B, 768]
        return self.proj(cls)                   # [B, out_dim]


class ImageEncoder(nn.Module):
    """
    EfficientNetV2S image encoder.
    Input  : RGB image tensor, shape [B, 3, 224, 224]
    Output : projected global-pool embedding, shape [B, out_dim]
    """

    def __init__(self, model_name: str = "tf_efficientnetv2_s",
                 out_dim: int = 256, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 1280
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)   # [B, 1280]
        return self.proj(feats)    # [B, out_dim]


class MetadataEncoder(nn.Module):
    """
    Custom MLP for tabular metadata.
    Architecture: Input -> [Linear -> BatchNorm -> GELU -> Dropout] x N -> out_dim

    Author  : Dhanush D
    """

    def __init__(self, in_dim: int, hidden_dims: List[int],
                 out_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, out_dim), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, out_dim]


class GatedFusion(nn.Module):
    """
    Gated Cross-Modal Attention Fusion.

    Fuses text (t), image (v), metadata (m) embeddings:
        gate  = Softmax(Linear(concat[t,v,m]))   -> [B, 3]
        fused = g_t * Pt + g_v * Pv + g_m * Pm
        out   = LayerNorm(fused)

    Author  : Dhanush D
    """

    def __init__(self, text_dim: int, img_dim: int,
                 meta_dim: int, fusion_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(text_dim,  fusion_dim)
        self.img_proj  = nn.Linear(img_dim,   fusion_dim)
        self.meta_proj = nn.Linear(meta_dim,  fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(text_dim + img_dim + meta_dim, 3),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, t: torch.Tensor, v: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
        gates = self.gate(torch.cat([t, v, m], dim=-1))   # [B, 3]
        g_t, g_v, g_m = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        fused = (g_t * self.text_proj(t) +
                 g_v * self.img_proj(v)  +
                 g_m * self.meta_proj(m))
        return self.norm(fused)   # [B, fusion_dim]


class MultimodalFakeNewsDetector(nn.Module):
    """
    Full multimodal fake news detection model.

    Architecture:
        Text    -> DistilBERT        -> projection (dim 256)  |
        Image   -> EfficientNetV2S   -> projection (dim 256)  | -> GatedFusion -> Classifier
        Metadata-> Custom MLP        -> dim 64                |

    Author  : Dhanush D
    Email   : dhanushd1812@gmail.com
    GitHub  : github.com/Drdhx
    """

    def __init__(self, meta_input_dim: int, num_classes: int = 2,
                 fusion_dim: int = 256,
                 text_model: str = "distilbert-base-uncased",
                 img_model:  str = "tf_efficientnetv2_s",
                 meta_hidden: List[int] = None,
                 meta_dim: int = 64,
                 text_dropout: float = 0.3,
                 img_dropout:  float = 0.3,
                 meta_dropout: float = 0.3):
        super().__init__()
        if meta_hidden is None:
            meta_hidden = [128, 64]

        self.text_encoder = TextEncoder(text_model, fusion_dim, text_dropout)
        self.img_encoder  = ImageEncoder(img_model, fusion_dim, img_dropout)
        self.meta_encoder = MetadataEncoder(
            meta_input_dim, meta_hidden, meta_dim, meta_dropout)
        self.fusion = GatedFusion(fusion_dim, fusion_dim, meta_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                image: torch.Tensor,
                metadata: torch.Tensor) -> torch.Tensor:
        t = self.text_encoder(input_ids, attention_mask)
        v = self.img_encoder(image)
        m = self.meta_encoder(metadata)
        fused = self.fusion(t, v, m)
        return self.classifier(fused)   # [B, num_classes]
