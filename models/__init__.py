"""Model architectures used in the Phase-Aware-IR experiments."""

from .dualhead import DualHeadCNN
from .dualhead_double_channels import DualHeadDoubleChannelsCNN
from .dualhead_double_channels_wide_mlp import DualHeadDoubleChannelsWideMLPCNN
from .dualhead_extra_final_conv import DualHeadExtraFinalConvCNN
from .dualhead_wide_mlp import DualHeadCNNMLP2048
from .patch_transformer import IRPatchTransformerCNN
from .resnet50_1d_base64 import ResNet50_1DBase64CNN
from .resnet50_1d_base128 import ResNet50_1DCNN

__all__ = [
    "DualHeadCNN",
    "DualHeadCNNMLP2048",
    "DualHeadDoubleChannelsCNN",
    "DualHeadDoubleChannelsWideMLPCNN",
    "DualHeadExtraFinalConvCNN",
    "IRPatchTransformerCNN",
    "ResNet50_1DBase64CNN",
    "ResNet50_1DCNN",
]
