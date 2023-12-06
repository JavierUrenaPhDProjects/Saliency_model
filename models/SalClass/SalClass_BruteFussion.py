import torch
from torch import nn
from models.SalClass.parts.backbone import brutefusion_backbone
from models.SalClass.parts.vision_transformer import ViT

from toolbox.utils import test_inference


class SalClass(nn.Module):
    def __init__(self, img_size, patch_size=32, channels=4, num_classes=257, dim=768, depth=12, heads=12, mlp_dim=2048,
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.backbone = brutefusion_backbone()
        self.vit = ViT(img_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                       channels=channels, dropout=dropout, emb_dropout=emb_dropout)
        self.final = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x = self.backbone(x1, x2)
        x = self.vit(x)
        # x = self.final(x)
        # x = x.argmax(dim=1).unsqueeze(dim=1)
        return x


def SalClass_brute(args):
    model = SalClass(args['img_size'])
    return model


if __name__ == '__main__':
    dtype = torch.float32
    x1 = torch.randn(1, 3, 384, 384).to(dtype)
    x2 = torch.randn(1, 1, 384, 384).to(dtype)
    model = SalClass(img_size=384)
    model.to(dtype)

    y = model(x1, x2)
    print(f' Output shape {y.shape}')

    test = True
    if test:
        test_inference(model, img_size=384, dtype=dtype)
