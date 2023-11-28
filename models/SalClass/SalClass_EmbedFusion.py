import torch
from torch import nn
from models.SalClass.parts.backbone import resnet_backbone
from models.SalClass.parts.vision_transformer import ViT


class SalClass(nn.Module):

    # This model will receive an embedding instead of an image. This means that the shape of the tensor
    # is going to be something like [16, 2048, 12, 12] where 2048 is the embedding dimensionality.
    # If we consider that as an image of 12x12 and 2048 channels we can configure the ViT that way
    def __init__(self, emb_shape=12, patch_size=1, channels=2048, num_classes=257, dim=768, depth=12, heads=12,
                 mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.backbone = resnet_backbone()
        self.backbone.eval()
        self.vit = ViT(emb_shape, patch_size, num_classes, dim, depth, heads, mlp_dim,
                       channels=channels, dropout=dropout, emb_dropout=emb_dropout)
        self.vit.double()
        self.final = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        """
        :param x1: this should be the image
        :param x2: this should be the saliency
        """
        # try:
        #     assert x2.shape[1] == 3
        # except:
        #     x2 = torch.cat((x2, x2, x2), dim=1)

        if x2.shape[1] != 3:
            x2 = torch.cat((x2, x2, x2), dim=1)

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        x = x1 + x2

        x = self.vit(x)
        # x = self.final(x)
        return x


def SalClass_embedd(args):
    model = SalClass(dropout=args['dropout'])
    return model
