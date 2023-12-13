import torch
from torch import nn


class ModulationLayer(nn.Module):
    def __init__(self):
        super(ModulationLayer, self).__init__()


class AdditiveModulation(ModulationLayer):
    """This method simply adds the feature maps from the RGB image and the salience map together.
    This is the most straightforward way to combine features, as it does not require any additional parameters and
    preserves the original dimensionality of the feature maps"""

    def forward(self, x1, x2):
        # Simply adds the two feature maps together
        return x1 + x2


class MultiplicativeModulation(ModulationLayer):
    """
    This method performs an element-wise multiplication (Hadamard product) of the two feature maps.
    It assumes that the salience map can provide a weighting mechanism, emphasizing or de-emphasizing features in
    the RGB map
    """

    def forward(self, x1, x2):
        # Element-wise multiplication of the two feature maps
        return x1 * x2


class OriginalModulation(ModulationLayer):
    """
    Hybrid approach that combines multiplicative and additive interactions between the two sets of features.
    Allows the network to learn which features from the RGB image are most important (amplified by the salience map)
    without losing the original feature information.
    Basically multiplies and then add a skip connection to maintain og features.
    """

    def forward(self, x1, x2):
        x = x1 * x2 + x1

        return x


class ConcatConvModulation(ModulationLayer):
    """
    This approach concatenates the feature maps along the channel dimension and then applies a convolutional layer
    to merge them into a single feature map with the same channel size as the original feature maps
    """

    def __init__(self, channel_size=2048):
        super(ConcatConvModulation, self).__init__()
        # A convolution layer that halves the channel size after concatenation
        self.conv = nn.Conv2d(in_channels=channel_size * 2, out_channels=channel_size, kernel_size=1)

    def forward(self, x1, x2):
        # Concatenate along the channel dimension
        concatenated_features = torch.cat((x1, x2), dim=1)
        # Apply convolution to combine features
        return self.conv(concatenated_features)


class SaliencyGuidedModulation(ModulationLayer):
    """
    Here, a convolutional layer transforms the salience map into a set of weights (using a sigmoid function to keep
    the weights between 0 and 1), which are then applied to the RGB features. This method allows the salience map
    to modulate the influence of each feature from the RGB map selectively
    """

    def __init__(self, channel_size=2048):
        super(SaliencyGuidedModulation, self).__init__()
        # A convolution layer to transform the salience map into weights
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Get the weights from the salience map
        gate = self.gate_conv(x2)
        # Apply the salience weights to the RGB features
        return x1 * gate


class FeaturePyramidsModulation(ModulationLayer):
    """
     Involves upsampling the salience features to match the spatial dimensions of the RGB features, concatenating
     both feature maps along the channel dimension, and then using a 1x1 convolution to combine them into a single
     feature map with the original channel size
    """

    def __init__(self, embedd_dim=(2048, 12, 12)):
        channel_size, height, width = embedd_dim
        super(FeaturePyramidsModulation, self).__init__()
        # Use a 1x1 convolution to adjust the channels after concatenation
        self.conv_adjust = nn.Conv2d(in_channels=channel_size * 2, out_channels=channel_size, kernel_size=1)
        # Upsample to the size of the RGB feature maps
        self.upsample = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        # Upsample salience features to match the size of RGB features
        upsampled_salience = self.upsample(x2)
        # Concatenate along the channel dimension
        concatenated_features = torch.cat((x1, upsampled_salience), dim=1)
        # Adjust the channels back to the original size
        combined_features = self.conv_adjust(concatenated_features)
        return combined_features


class SqueezeExcitationModulation(nn.Module):
    """
    Combines the features from the RGB image and the salience map, applies global average pooling to produce a
    channel descriptor, and then uses two fully connected layers with a reduction ratio (default of 16) to learn a
    set of channel-wise weights. The sigmoid function ensures these weights are between 0 and 1. The final output is
    the RGB feature map re-weighted by these learned channel-wise weights
    """

    def __init__(self, channel_size=2048, reduction_ratio=16):
        super(SqueezeExcitationModulation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel_size, channel_size // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_size // reduction_ratio, channel_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Squeeze operation
        y = self.squeeze(x1 + x2)
        y = torch.flatten(y, 1)
        # Excitation operation
        y = self.excitation(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        # Scale operation
        return x1 * y.expand_as(x1)


if __name__ == '__main__':
    batch_size = 4  # Example batch size
    channels = 2048  # Example channels
    height, width = 12, 12  # Spatial dimensions
    features_rgb = torch.rand(batch_size, channels, height, width)
    features_salience = torch.rand(batch_size, channels, height, width)

    additive_layer = AdditiveModulation()
    concat_conv_layer = ConcatConvModulation()
    multiplicative_layer = MultiplicativeModulation()
    saliency_guided_layer = SaliencyGuidedModulation()
    feature_pyramids_layer = FeaturePyramidsModulation()
    squeeze_excitation_layer = SqueezeExcitationModulation()

    additive_output = additive_layer(features_rgb, features_salience)
    concat_conv_output = concat_conv_layer(features_rgb, features_salience)
    multiplicative_output = multiplicative_layer(features_rgb, features_salience)
    saliency_guided_output = saliency_guided_layer(features_rgb, features_salience)
    feature_pyramids_output = feature_pyramids_layer(features_rgb, features_salience)
    squeeze_excitation_output = squeeze_excitation_layer(features_rgb, features_salience)

    output_shapes = {
        "Additive Output Shape": additive_output.shape,
        "Concat Conv Output Shape": concat_conv_output.shape,
        "Multiplicative Output Shape": multiplicative_output.shape,
        "Saliency Guided Output Shape": saliency_guided_output.shape,
        "Feature Pyramids Output Shape": feature_pyramids_output.shape,
        "Squeeze-Excitation Output Shape": squeeze_excitation_output.shape
    }

    print(output_shapes)
