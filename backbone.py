"""Frozen CLIP ViT-B/16 and the trainable modality-continuous parameter adapter."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class GroupedLinear(nn.Module):
    """Generate large factors with a block-diagonal output projection."""

    def __init__(self, in_features, out_features, groups=16, zero_output=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, in_features // groups, out_features // groups))
        self.bias = nn.Parameter(torch.zeros(groups, out_features // groups))
        if zero_output:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=0.02)

    def forward(self, context):
        grouped = context.reshape(len(context), len(self.weight), -1)
        return (torch.einsum("bgi,gio->bgo", grouped, self.weight) + self.bias).flatten(1)


class DynamicLowRank(nn.Module):
    """Build a separate low-rank update for each slice and projection."""

    def __init__(self, width, out_features, rank, projections=1):
        super().__init__()
        self.rank = rank
        self.projections = projections
        self.out_features = out_features
        hidden = width // 4
        self.context = nn.Sequential(nn.Linear(width, hidden), nn.LeakyReLU(0.1))
        self.factor_a = GroupedLinear(hidden, projections * rank * width)
        # BA starts at zero and B receives a gradient from the first batch.
        self.factor_b = GroupedLinear(hidden, projections * rank * out_features, zero_output=True)

    def forward(self, tokens, context):
        batch, _, width = tokens.shape
        conditioned = self.context(context)
        # Store A as [rank, input] and B transposed as [rank, output] for B(Ax).
        factor_a = self.factor_a(conditioned).reshape(batch, self.projections, self.rank, width)
        factor_b = self.factor_b(conditioned).reshape(
            batch, self.projections, self.rank, self.out_features
        )
        low_rank = torch.einsum("bli,bpri->blpr", tokens, factor_a)
        return torch.einsum("blpr,bpro->blpo", low_rank, factor_b).flatten(2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads, rank):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads).requires_grad_(False)
        self.ln_1 = nn.LayerNorm(width).requires_grad_(False)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, 4 * width)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(4 * width, width)),
        ])).requires_grad_(False)
        self.ln_2 = nn.LayerNorm(width).requires_grad_(False)
        self.qkv_adapter = DynamicLowRank(width, width, rank, projections=3)
        # The first FFN projection expands from width to 4 * width.
        self.ffn_adapter = DynamicLowRank(width, 4 * width, rank)

    def forward(self, x, context):
        batch, length, width = x.shape
        heads = self.attn.num_heads
        tokens = self.ln_1(x)
        qkv = F.linear(tokens, self.attn.in_proj_weight, self.attn.in_proj_bias)
        qkv = qkv + self.qkv_adapter(tokens, context)
        q, k, v = qkv.reshape(batch, length, 3, heads, width // heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.attn.out_proj(attended)
        tokens = self.ln_2(x)
        hidden = self.mlp.c_fc(tokens) + self.ffn_adapter(tokens, context)
        return x + self.mlp.c_proj(self.mlp.gelu(hidden))


class Transformer(nn.Module):
    def __init__(self, width, heads, layers, rank):
        super().__init__()
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads, rank) for _ in range(layers)
        )

    def forward(self, x, context, checkpoint_blocks):
        for block in self.resblocks:
            if checkpoint_blocks:
                x = checkpoint(block, x, context, use_reentrant=False)
            else:
                x = block(x, context)
        return x


class VisionTransformer(nn.Module):
    width = 768
    image_size = 224
    patch_size = 16

    def __init__(self, rank, code_number, checkpoint_blocks=False):
        super().__init__()
        width = self.width
        self.checkpoint_blocks = checkpoint_blocks
        self.conv1 = nn.Conv2d(3, width, self.patch_size, self.patch_size, bias=False).requires_grad_(False)
        self.class_embedding = nn.Parameter(torch.empty(width), requires_grad=False)
        patch_count = (self.image_size // self.patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.empty(patch_count + 1, width), requires_grad=False)
        self.ln_pre = nn.LayerNorm(width).requires_grad_(False)
        self.transformer = Transformer(width, 12, 12, rank)
        self.ln_post = nn.LayerNorm(width).requires_grad_(False)

        self.router_mlp = nn.Sequential(
            nn.Linear(width, width), nn.LeakyReLU(0.1), nn.Linear(width, code_number)
        )
        self.codebook = nn.Parameter(torch.empty(code_number, width))
        nn.init.normal_(self.codebook, std=width ** -0.5)

    def forward(self, images):
        """Encode [batch, slices, 3, 224, 224] into patient features and queries."""
        batch, slices = images.shape[:2]
        rgb = images.flatten(0, 1)
        patches = self.conv1(rgb.mul(2).sub(1))
        pooled = patches.mean(dim=(-2, -1))
        context = self.router_mlp(pooled).softmax(dim=-1) @ self.codebook

        tokens = patches.flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(tokens.dtype).expand(len(tokens), 1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.positional_embedding.to(tokens.dtype)
        tokens = self.transformer(self.ln_pre(tokens), context, self.training and self.checkpoint_blocks)
        features = self.ln_post(tokens[:, 1:].mean(dim=1)).reshape(batch, slices, self.width).mean(dim=1)
        return features, context.reshape(batch, slices, self.width).mean(dim=1)


def load_clip_backbone(weights, rank, code_number, checkpoint_blocks=False):
    if not weights.is_file():
        raise FileNotFoundError(f"Shared CLIP ViT-B/16 checkpoint not found: {weights}")
    model = VisionTransformer(rank, code_number, checkpoint_blocks)
    pretrained = torch.jit.load(str(weights), map_location="cpu").visual.state_dict()
    expected = {name for name, param in model.named_parameters() if not param.requires_grad}
    missing = expected - pretrained.keys()
    if missing:
        raise RuntimeError(f"Incomplete CLIP ViT-B/16 checkpoint: {sorted(missing)}")
    model.load_state_dict({name: pretrained[name] for name in expected}, strict=False)
    return model
