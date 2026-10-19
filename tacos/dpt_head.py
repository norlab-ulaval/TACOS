"""DPT decoder (Ranftl, Bochkovskiy & Koltun, "Vision Transformers for Dense Prediction", ICCV 2021), written for TACOS:
the 4 encoder token maps are reassembled into a 4-level pyramid (x4, x2, x1, x0.5 the token grid), brought to a
common width, fused coarse-to-fine by RefineNet-style blocks built from pre-activation residual conv units, and
decoded by a small upsampling head. Each fusion stage takes an optional `condition` callable (the GCA context)."""
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    """A Conv2d (He-normal init, fan_out, zero bias) with an optional ReLU on its input and/or output."""

    def __init__(self, cin, cout, k, stride=1, bias=True, relu_in=False, relu_out=False):
        super().__init__()
        self.relu_in, self.relu_out = relu_in, relu_out
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(F.relu(x) if self.relu_in else x)
        return F.relu(x) if self.relu_out else x


class Reassemble(nn.Module):
    """Per level: 1x1 projection to the level width, then resampling (x4, x2, x1, x0.5) to build the pyramid."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projects = nn.ModuleList(Conv(cin, cout, 1) for cin, cout in zip(in_channels, out_channels))
        c0, c1, _, c3 = out_channels
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(c0, c0, 4, stride=4),
                                            nn.ConvTranspose2d(c1, c1, 2, stride=2), nn.Identity(),
                                            nn.Conv2d(c3, c3, 3, stride=2, padding=1)])

    def forward(self, maps):
        return [resize(project(m)) for m, project, resize in zip(maps, self.projects, self.resize_layers)]


class ResidualUnit(nn.Module):
    """x + conv(relu(conv(relu(x)))) with two bias-free 3x3 convs."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv(ch, ch, 3, bias=False, relu_in=True)
        self.conv2 = Conv(ch, ch, 3, bias=False, relu_in=True)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class FusionBlock(nn.Module):
    """RefineNet-style fusion: add the residual-refined skip feature (if any), optionally condition, refine again,
    upsample x2 (bilinear, corners aligned) and mix channels with a 1x1 conv."""

    def __init__(self, ch, has_skip=True):
        super().__init__()
        self.project = Conv(ch, ch, 1)
        skip_unit = ResidualUnit(ch)   # always built, so the parameter RNG stream does not depend on `has_skip`
        self.res_conv_unit1 = skip_unit if has_skip else None
        self.res_conv_unit2 = ResidualUnit(ch)

    def forward(self, x, skip=None, condition=None):
        if skip is not None:
            if skip.shape != x.shape:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.res_conv_unit1(skip)
        if condition is not None:
            x = condition(x)
        x = F.interpolate(self.res_conv_unit2(x), scale_factor=2, mode="bilinear", align_corners=True)
        return self.project(x)


class OutputHead(nn.Module):
    """3x3 conv (halve width), x2 bilinear upsampling, 3x3 conv + ReLU, 1x1 conv to the output channels."""

    def __init__(self, ch, n_out, n_hidden=32):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(ch, ch // 2, 3, padding=1),
                                  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                  nn.Conv2d(ch // 2, n_hidden, 3, padding=1), nn.ReLU(), nn.Conv2d(n_hidden, n_out, 1))

    def forward(self, x):
        return self.head(x)


class DPTHead(nn.Module):
    """Maps (B, 4, C, h, w) ViT token maps (shallow to deep) to a (B, n_output_channels, 16h, 16w) dense output.

    `stage_conditioning`, if given, holds one callable per fusion stage (deepest first), applied to that stage's fused
    features before its second residual unit."""

    def __init__(self, in_channels=(1024,) * 4, channels=256, post_process_channels=(128, 256, 512, 1024),
                 n_output_channels=256, n_hidden_channels=32):
        super().__init__()
        self.channels = channels
        self.reassemble_blocks = Reassemble(in_channels, post_process_channels)
        self.convs = nn.ModuleList(Conv(c, channels, 3, bias=False) for c in post_process_channels)
        self.fusion_blocks = nn.ModuleList(FusionBlock(channels, has_skip=i > 0) for i in range(len(self.convs)))
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.project = Conv(channels, channels, 3, relu_out=True)
        self.conv_depth = OutputHead(channels, n_output_channels, n_hidden_channels)

    def forward(self, maps, stage_conditioning=None):
        conditions = stage_conditioning or [None] * self.num_fusion_blocks
        pyramid = [conv(level) for conv, level in zip(self.convs, self.reassemble_blocks(maps.unbind(1)))]
        x = None
        for block, skip, condition in zip(self.fusion_blocks, reversed(pyramid), conditions):
            x = block(skip, condition=condition) if x is None else block(x, skip, condition=condition)
        return self.conv_depth(self.project(x))
