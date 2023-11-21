import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.models import resnet50


class resnet_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights='ResNet50_Weights.DEFAULT').to(dtype=torch.float64)
        del self.backbone.fc

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x


class medium_cnn(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)).to(dtype=torch.float64)
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)).to(dtype=torch.float64)
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 2048, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)).to(dtype=torch.float64)

    def forward(self, xb):
        x = self.stage1(xb)
        x = self.stage2(x)
        x = self.stage3(x)

        return x


class brutefusion_backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x
