"""Some codes adapted from https://github.com/kazuto1011/deeplab-pytorch"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .resnet import Bottleneck, ResNet, model_urls

class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                f"c{i}",
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )
        
        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(ResNet):
    """
    DeepLab v2 with ResNet backbone
    """
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__(
            block, layers, num_classes, zero_init_residual,
            groups, width_per_group, replace_stride_with_dilation, norm_layer,
        )
        self.aspp = ASPP(2048, num_classes, [6, 12, 18, 24])
    
    def _forward_classifier(self, x):
        x = self.aspp(x)
        return x
    
    def forward(self, x):
        input_size = x.shape[-2:]
        out, features = self._forward_backbone(x)
        out = self._forward_classifier(x)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out, features


def _deeplabv2(
    arch,
    block,
    layers,
    pretrained,
    progress,
    num_classes,
    **kwargs,
):
    model = DeepLabV2(block, layers, num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                                progress=progress)
        model.load_statd_dict(state_dict, strict=False)
    return model


def deeplab50(pretrained=False, progress=True, num_classes=19, **kwargs):
    return _deeplabv2("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress,
                      num_classes=num_classes, replace_stride_with_dilation=[False, True, True], **kwargs)


def deeplab101(pretrained=False, progress=True, num_classes=19, **kwargs):
    return _deeplabv2("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress,
                      num_classes=num_classes, replace_stride_with_dilation=[False, True, True], **kwargs)