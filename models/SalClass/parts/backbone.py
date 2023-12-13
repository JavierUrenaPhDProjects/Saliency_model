import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.models import resnet50
import torch.nn.functional as F


class resnet_backbone(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.backbone = resnet50(weights='ResNet50_Weights.DEFAULT').to(dtype=dtype)
        del self.backbone.fc
        for param in self.backbone.parameters():
            param.requires_grad = False

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
        self.stage1 = nn.Sequential(nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 2048, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, xb):
        x = self.stage1(xb)
        x = self.stage2(x)
        x = self.stage3(x)

        return x


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, n_channels=3, output_dim=2048, depth='medium'):
        super(ConvolutionalNeuralNetwork, self).__init__()

        if depth == 'shallow':
            out_size = 128
            self.layers = nn.Sequential(
                nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1)
            )
        elif depth == 'medium':
            out_size = 512
            self.layers = nn.Sequential(
                nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1)
            )
        elif depth == 'deep':
            out_size = 512
            self.layers = nn.Sequential(
                nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512)
            )

        self.adaptativepool = nn.AdaptiveAvgPool2d((12, 12))
        self.last_layer = nn.Conv2d(out_size, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.adaptativepool(x)
        x = self.last_layer(x)
        return x


def shallow_CNN(n_channels, output_size=2048):
    model = ConvolutionalNeuralNetwork(n_channels, output_size, depth='shallow')
    return model


def medium_CNN(n_channels, output_size=2048):
    model = ConvolutionalNeuralNetwork(n_channels, output_size, depth='medium')
    return model


def deep_CNN(n_channels, output_size=2048):
    model = ConvolutionalNeuralNetwork(n_channels, output_size, depth='deep')
    return model


class brutefusion_backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x
