import torch
from torch import nn
from models.SalClass.parts.backbone import resnet_backbone, medium_cnn
from models.SalClass.parts.vision_transformer import ViT
from models.SalClass.parts.backbone import brutefusion_backbone

from toolbox.utils import test_inference


class SalClass(nn.Module):
    # This model will receive an embedding instead of an image. This means that the shape of the tensor
    # is going to be something like [16, 2048, 12, 12] where 2048 is the embedding dimensionality.
    # If we consider that as an image of 12x12 and 2048 channels we can configure the ViT that way
    def __init__(self, channels=2048, num_classes=257, dim=768, depth=12, heads=12,
                 mlp_dim=2048, dropout=0.1, emb_dropout=0.1, backbone='resnet', dtype=torch.float64,
                 relationship='sum'):
        super().__init__()
        self.backbone_type = backbone
        self.relationship = relationship
        if backbone == 'resnet':
            self.emb_shape = 12
            self.patch_size = 1
            self.backbone = resnet_backbone(dtype)
            self.backbone.eval()
            self.common_bb = True
        elif backbone == 'cnn':
            self.emb_shape = 12  # n "pixels" the embedding has after the cnn
            self.patch_size = 1  # emb_shape / patch_size = n_embeddings the ViT will have (48/4 = 12)
            self.backbone1 = medium_cnn(n_channels=3)
            self.backbone2 = medium_cnn(n_channels=1)
            self.common_bb = False
        self.vit = ViT(self.emb_shape, self.patch_size, num_classes, dim, depth, heads, mlp_dim,
                       channels=channels, dropout=dropout, emb_dropout=emb_dropout)

        self.final = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        """
        :param x1: this should be the image
        :param x2: this should be the saliency
        """

        if self.common_bb:
            if x2.shape[1] != 3:
                x2 = torch.cat((x2, x2, x2), dim=1)

            x1 = self.backbone(x1)
            x2 = self.backbone(x2)
        else:
            x1 = self.backbone1(x1)
            x2 = self.backbone2(x2)

        x = x1+ x2
        x = self.vit(x)
        # x = self.final(x)
        return x


def SalClass_embedd_resnet(args):
    model = SalClass(dropout=args['dropout'], backbone='resnet', dtype=args['dtype'])
    return model


def SalClass_embedd_cnn(args):
    model = SalClass(dropout=args['dropout'], backbone='cnn')
    return model


# def SalClass_embedd_brutefusion(args):


if __name__ == '__main__':
    dtype = torch.float32
    x1 = torch.randn(1, 3, 384, 384).to(dtype)
    x2 = torch.randn(1, 1, 384, 384).to(dtype)
    model = SalClass(dropout=0.001, backbone='resnet', dtype=dtype, relationship='concat')
    model.to(dtype)

    y = model(x1, x2)
    print(f' Output shape {y.shape}')

    test = True
    if test:
        test_inference(model, img_size=384, dtype=dtype)
