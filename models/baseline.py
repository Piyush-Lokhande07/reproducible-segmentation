import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    """Standard double convolution block used in decoder stages."""
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


class DecoderBlock(nn.Module):
    """Decoder stage with standard bilinear upsampling and skip concatenation."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """
    Modest Encoder-Decoder baseline:
    - Backbone: ResNet-18 (ImageNet pretrained)
    - Decoder: Bilinear upsampling + Standard Skip Concatenation
    - Output: 1x1 Conv projecting to num_classes (21)
    """
    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Encoder stages
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64 ch, 1/2 res
        self.maxpool = resnet.maxpool                                       # 64 ch, 1/4 res
        self.layer1 = resnet.layer1                                         # 64 ch, 1/4 res
        self.layer2 = resnet.layer2                                         # 128 ch, 1/8 res
        self.layer3 = resnet.layer3                                         # 256 ch, 1/16 res
        self.layer4 = resnet.layer4                                         # 512 ch, 1/32 res

        # Decoder stages (Standard concatenation)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256) # 1/16 res
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128) # 1/8 res
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)  # 1/4 res
        self.dec1 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)  # 1/2 res
        
        # Final head
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) # 1/1 res
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_init = self.initial(x)         # [B, 64, H/2, W/2]
        x1 = self.layer1(self.maxpool(x_init)) # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)             # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)             # [B, 256, H/16, W/16]
        bottleneck = self.layer4(x3)     # [B, 512, H/32, W/32]

        # Decoder with Standard Skip Connections
        d4 = self.dec4(bottleneck, x3)   # [B, 256, H/16, W/16]
        d3 = self.dec3(d4, x2)           # [B, 128, H/8, W/8]
        d2 = self.dec2(d3, x1)           # [B, 64, H/4, W/4]
        d1 = self.dec1(d2, x_init)       # [B, 32, H/2, W/2]

        out = self.final_upsample(d1)    # [B, 32, H, W]
        return self.head(out)            # [B, num_classes, H, W]
