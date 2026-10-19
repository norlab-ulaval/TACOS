"""Frozen DINOv3-B encoder, DPT decoder with optional gated cross-attention (GCA) context, Huber loss."""
import math
from functools import cache, partial

import timm
import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tacos.dpt_head import DPTHead

CONTEXT_VARIANTS = {"none": [], "attitude": ["roll", "pitch"], "history": ["history_front", "history_hind"],
                    "all": ["roll", "pitch", "history_front", "history_hind"]}
ENCODER_NAME = "vit_base_patch16_dinov3.lvd1689m"
ENCODER_LAYERS = [1, 4, 7, 10]                             # 0-based blocks = HF hidden_states[2, 5, 8, 11]
EMBED_DIM = 768
IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # the encoder's pretrained_cfg (ImageNet)
CONTEXT_Z_CLAMP = 5.0


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 ViT-B/16. Returns the patch tokens of 4 intermediate blocks, (B, 4, 768, H/16, W/16).

    Inputs are ImageNet-normalized with `IMAGE_MEAN` / `IMAGE_STD` by the dataset."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(ENCODER_NAME, pretrained=True, dynamic_img_size=True).requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)                         # frozen: always in eval mode

    def forward(self, x):
        return torch.stack(self.backbone.forward_intermediates(x, indices=ENCODER_LAYERS, norm=False, output_fmt="NCHW",
                                                               intermediates_only=True), dim=1)


class GatedCrossAttention(nn.Module):
    """Let every image location attend to the context tokens through a tanh-gated residual.

    The gate starts at zero, so a fresh module leaves the image-only decoder unchanged."""

    def __init__(self, channels, attention_dim=128, num_heads=4):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError(f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads}).")
        self.channels, self.attention_dim, self.num_heads = channels, attention_dim, num_heads
        self.head_dim = attention_dim // num_heads
        self.image_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(attention_dim)
        self.query_projection = nn.Linear(channels, attention_dim)
        self.key_value_projection = nn.Linear(attention_dim, 2 * attention_dim)
        self.output_projection = nn.Linear(attention_dim, channels)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, features, tokens):
        batch, _, height, width = features.shape
        queries = self.query_projection(self.image_norm(features.flatten(2).transpose(1, 2)))
        keys, values = self.key_value_projection(self.context_norm(tokens)).chunk(2, dim=-1)

        def split_heads(t):
            return t.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(split_heads(queries), split_heads(keys), split_heads(values))
        attended = attended.transpose(1, 2).reshape(batch, height * width, self.attention_dim)
        residual = self.output_projection(attended).transpose(1, 2).reshape(batch, self.channels, height, width)
        return features + self.gate.tanh().to(features.dtype) * residual.to(features.dtype)


class ContextAttention(nn.Module):
    """One `GatedCrossAttention` per DPT fusion stage, plus the context tokenizer.

    Each field f becomes the token `z_f * value_embeddings[f] + field_embeddings[f]`, so equal values of different
    fields stay distinguishable. The module lives outside the DPT head, which keeps the image-only decoder keys."""

    def __init__(self, channels, num_stages, fields, attention_dim=128, num_heads=4):
        super().__init__()
        self.fields = list(fields)
        self.blocks = nn.ModuleList(GatedCrossAttention(channels, attention_dim, num_heads) for _ in range(num_stages))
        self.value_embeddings = nn.Parameter(torch.empty(len(self.fields), attention_dim))
        self.field_embeddings = nn.Parameter(torch.empty(len(self.fields), attention_dim))
        bound = 1.0 / math.sqrt(attention_dim)
        nn.init.uniform_(self.value_embeddings, -bound, bound)
        nn.init.uniform_(self.field_embeddings, -bound, bound)

    def tokenize(self, z):
        return z.to(self.value_embeddings.dtype).unsqueeze(-1) * self.value_embeddings + self.field_embeddings


class TacosNet(LightningModule):
    """Predicts the per-pixel log instability (B, 1, H, W) of an ImageNet-normalized image (B, 3, H, W).

    With `context != "none"`, the context vector (B, C), ordered as `CONTEXT_VARIANTS[context]`, is z-scored with
    `context_mean` / `context_std` (train-frame statistics; a NaN value becomes the mean), clamped to ±5 and injected
    into every DPT fusion stage through gated cross-attention. Checkpoints hold only the decoder (+ GCA); the frozen
    encoder is rebuilt."""

    def __init__(self, context: str = "none", lr=3e-5, weight_decay=0.01, warmup_steps=100, min_lr=1e-6,
                 gate_lr_scale=10.0, huber_delta=0.5, context_mean=None, context_std=None, attention_dim=128,
                 num_heads=4):
        super().__init__()
        self.save_hyperparameters()
        self.fields = CONTEXT_VARIANTS[context]
        self.encoder = DINOv3Encoder()
        self.decoder = DPTHead(in_channels=[EMBED_DIM] * 4, channels=256, n_output_channels=1)
        nn.init.normal_(self.decoder.conv_depth.head[-1].weight, mean=0.0, std=1e-3)   # near-zero initial output
        nn.init.zeros_(self.decoder.conv_depth.head[-1].bias)
        self.context_attention = None
        if self.fields:
            n = len(self.fields)
            self.context_attention = ContextAttention(self.decoder.channels, self.decoder.num_fusion_blocks, self.fields,
                                                      attention_dim, num_heads)
            self.register_buffer("context_mean", torch.zeros(n) if context_mean is None else torch.as_tensor(
                context_mean, dtype=torch.float32))
            self.register_buffer("context_std", torch.ones(n) if context_std is None else torch.as_tensor(
                context_std, dtype=torch.float32))

    def forward(self, x, context=None):
        features = self.encoder(x)
        if self.context_attention is None:
            return self.decoder(features)
        if context is None:
            raise ValueError(f"This model is conditioned on {self.fields}: pass context.")
        z = (context.to(self.context_mean) - self.context_mean) / self.context_std
        z = z.nan_to_num(0.0).clamp(-CONTEXT_Z_CLAMP, CONTEXT_Z_CLAMP)          # a NaN value is read as the train mean
        tokens = self.context_attention.tokenize(z)
        return self.decoder(features, [partial(block, tokens=tokens) for block in self.context_attention.blocks])

    def loss(self, outputs, targets):
        """Huber loss over the labelled (finite) target pixels; 0 when a batch has none."""
        if outputs.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets, size=outputs.shape[2:], mode="nearest")
        mask = torch.isfinite(targets)
        if not mask.any():
            return outputs.new_zeros((), requires_grad=outputs.requires_grad)
        with torch.autocast(device_type=outputs.device.type, enabled=False):
            return F.huber_loss(outputs[mask].float(), targets[mask].float(), delta=self.hparams.huber_delta)

    def training_step(self, batch, batch_idx):
        image, context, target = batch[:3]
        loss = self.loss(self(image, context), target.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, context, target = batch[:3]
        loss = self.loss(self(image, context), target.float())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(image))
        return loss

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in [k for k in state if k.startswith("encoder.")]:
            del state[key]
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        encoder_state = {f"encoder.{k}": v for k, v in self.encoder.state_dict().items()}
        return super().load_state_dict({**encoder_state, **state_dict}, strict=strict, assign=assign)

    def configure_optimizers(self):
        """AdamW with a linear warm-up into a per-step cosine decay down to `min_lr`. The GCA learns at lr × 10."""
        hp = self.hparams
        params = [p for p in self.parameters() if p.requires_grad]
        if self.context_attention is not None and hp.gate_lr_scale != 1.0:
            gca = {id(p) for p in self.context_attention.parameters()}
            params = [{"params": [p for p in params if id(p) not in gca], "lr": hp.lr},
                      {"params": [p for p in params if id(p) in gca], "lr": hp.lr * hp.gate_lr_scale}]
        optimizer = AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
        min_scale = hp.min_lr / hp.lr

        @cache
        def total_steps():                                  # needs the trainer, so only queried after warm-up
            return self.trainer.estimated_stepping_batches

        def lr_lambda(step):
            if step < hp.warmup_steps:
                return step / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, total_steps() - hp.warmup_steps)
            return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
