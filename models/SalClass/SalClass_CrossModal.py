import torch
from torch import nn
from models.SalClass.parts.crossmod_vit import CrossModal_ViT


class SalClass(nn.Module):
    def __init__(self, img_size, patch_size=32, channels=3, num_classes=257, dim=768, depth=12, heads=12, mlp_dim=2048,
                 dropout=0.1, emb_dropout=0.1, cm_mode='mode_1'):
        super().__init__()
        self.vit = CrossModal_ViT(img_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                                  channels=channels, dropout=dropout, emb_dropout=emb_dropout, cm_mode=cm_mode)
        self.final = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x = self.vit(x1, x2)
        x = self.final(x)

        return x


def SalClass_crossmod_mode1(args):
    model = SalClass(img_size=args['img_size'], dropout=args['dropout'], cm_mode='mode_1')
    return model


def SalClass_crossmod_mode2(args):
    model = SalClass(img_size=args['img_size'], dropout=args['dropout'], cm_mode='mode_2')
    return model


if __name__ == '__main__':
    args = {'img_size': 384, 'dropout': 0.001}
    model = SalClass_crossmod_mode2(args)

    img = torch.randn(1, 3, 384, 384)
    sal = torch.randn(1, 1, 384, 384)

    y = model(img, sal)
    print(y.shape)
