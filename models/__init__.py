"""Model architectures used in the Phase-Aware-IR experiments."""

from .aggmapnet import AggMapNetPaperLike
from .dualhead import DualHeadCNN
from .dualhead_double_channels import DualHeadDoubleChannelsCNN
from .dualhead_double_channels_wide_mlp import DualHeadDoubleChannelsWideMLPCNN
from .dualhead_extra_final_conv import DualHeadExtraFinalConvCNN
from .dualhead_wide_mlp import DualHeadCNNMLP2048
from .ircnn import IRCNNPaper
from .patch_transformer import IRPatchTransformerCNN
from .resnet50_1d_base64 import ResNet50_1DBase64CNN
from .resnet50_1d_base128 import ResNet50_1DCNN

__all__ = [
    "AggMapNetPaperLike",
    "DualHeadCNN",
    "DualHeadCNNMLP2048",
    "DualHeadDoubleChannelsCNN",
    "DualHeadDoubleChannelsWideMLPCNN",
    "DualHeadExtraFinalConvCNN",
    "IRCNNPaper",
    "IRPatchTransformerCNN",
    "ResNet50_1DBase64CNN",
    "ResNet50_1DCNN",
]
