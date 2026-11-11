import torch
import torch.nn as nn
import torchvision.models as models

class AttentionGate(nn.Module):
    """
    Additive Attention Gate for skip fusion.
    Gates low-level features (x) using high-level decoder features (g).
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions if needed
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionDecoderBlock(nn.Module):
    """Decoder block incorporating the Attention Gate on the skip connection."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention = AttentionGate(
            F_g=in_channels, 
            F_l=skip_channels, 
            F_int=max(skip_channels // 2, 16)
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x_up = self.upsample(x)
        if skip is not None:
            # Apply attention gate to skip features using upsampled gating signal
            gated_skip = self.attention(g=x_up, x=skip)
            x_up = torch.cat([x_up, gated_skip], dim=1)
        return self.conv(x_up)


class AttentionUNet(nn.Module):
    """
    Controlled Variant: ResNet-18 Backbone with Attention Skip Fusion.
    Architecture, backbone, resolution, and heads are identical to BaselineUNet;
    only the skip fusion mechanism is modified.
    """
    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Identical Encoder stages
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modified Decoder stages (Attention Skip Fusion)
        self.dec4 = AttentionDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = AttentionDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = AttentionDecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)
        self.dec1 = AttentionDecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)

        # Identical Final head
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = self.initial(x)
        x1 = self.layer1(self.maxpool(x_init))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        bottleneck = self.layer4(x3)

        # Decoder with Attention Gated Skips
        d4 = self.dec4(bottleneck, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x_init)

        out = self.final_upsample(d1)
        return self.head(out)
