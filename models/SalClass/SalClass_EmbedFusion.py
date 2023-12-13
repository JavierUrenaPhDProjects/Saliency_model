import torch
from torch import nn
from models.SalClass.parts.vision_transformer import ViT
from models.SalClass.parts.backbone import *
from models.SalClass.parts.modulations import *
from toolbox.utils import test_inference


class SalClass(nn.Module):
    """
    This model will play in the feature space of the images and saliencies. This means that the shape of the tensor
    is going to be something like [16, 2048, 12, 12] where 2048 is the embedding dimensionality.
    If we consider that as an image of 12x12 and 2048 channels we can configure the ViT that way.
    """

    def __init__(self, channels=2048, num_classes=257, dim=768, depth=12, heads=12,
                 mlp_dim=2048, dropout=0.1, emb_dropout=0.1, backbone='resnet', modulation='AdditiveModulation',
                 dtype=torch.float64):
        super().__init__()

        self.emb_shape = 12  # n "pixels" the embedding has after the cnn
        self.patch_size = 1  # emb_shape / patch_size = n_embeddings the ViT will have (48/4 = 12)

        self.backbone_type = backbone
        self.modulation = modulation
        if backbone == 'resnet':
            # The feature extractors are resnet50. This means there is a common backbone with no learnable params.
            self.backbone = resnet_backbone(dtype)
            self.backbone.eval()
            self.common_bb = True
        else:
            # The feature extractors are CNNs (for now). This means
            self.backbone1 = eval(f"{backbone}(3)")
            self.backbone2 = eval(f"{backbone}(1)")
            self.common_bb = False

        self.modulation_layer = eval(f"{modulation}" + "()")

        self.vit = ViT(self.emb_shape, self.patch_size, num_classes, dim, depth, heads, mlp_dim,
                       channels=channels, dropout=dropout, emb_dropout=emb_dropout)

        # Softmax is not necessary in prior because the Loss function already applies softmax. But is still put
        # just in case
        # self.final = nn.Softmax(dim=1)

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

        x = self.modulation_layer(x1, x2)
        x = self.vit(x)

        return x


# _______________RESNET50-BASED MODELS_______________

def SalClass_embedd_resnet(args):
    """
    The pre-trained Resnet50 is the backbone of this model. Both image and saliency go through the same architecture
    In order for the Resnet50 to accept the saliency, this is concatenated with itself until it has 3 channels.
    :param args:
    :return:
    """
    model = SalClass(dropout=args['dropout'], backbone='resnet', dtype=args['dtype'])
    return model


# _______________CNN-BASED MODELS_______________

# 1) Different depths | No special modulation ("AdditiveModulation" applied)

def SalClass_embedd_shallow_CNN(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN')
    return model


def SalClass_embedd_medium_CNN(args):
    model = SalClass(dropout=args['dropout'], backbone='medium_CNN')
    return model


def SalClass_embedd_deep_CNN(args):
    model = SalClass(dropout=args['dropout'], backbone='deep_CNN')
    return model


# 2) Shallow depth | Different kinds of modulations

def SalClass_embedd_cnn_AdditiveModulation(args):
    # This architecture is the same as "SalClass_embedd_shallow_CNN"
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='AdditiveModulation')
    return model


def SalClass_embedd_cnn_MultiplicativeModulation(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='MultiplicativeModulation')
    return model


def SalClass_embedd_cnn_OriginalModulation(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='OriginalModulation')
    return model


def SalClass_embedd_cnn_ConcatConvModulation(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='ConcatConvModulation')
    return model


def SalClass_embedd_cnn_SaliencyGuidedModulation(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='SaliencyGuidedModulation')
    return model


def SalClass_embedd_cnn_FeaturePyramidsModulation(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='FeaturePyramidsModulation')
    return model


def SalClass_embedd_cnn_SqueezeExcitationModulation(args):
    model = SalClass(dropout=args['dropout'], backbone='shallow_CNN', modulation='SqueezeExcitationModulation')
    return model


if __name__ == '__main__':
    dtype = torch.float32
    x1 = torch.randn(1, 3, 384, 384).to(dtype)
    x2 = torch.randn(1, 1, 384, 384).to(dtype)
    model = SalClass(dropout=0.001, backbone='medium_CNN', dtype=dtype)
    model.to(dtype)

    y = model(x1, x2)
    print(f' Output shape {y.shape}')

    test = False
    if test:
        test_inference(model, img_size=384, dtype=dtype)
