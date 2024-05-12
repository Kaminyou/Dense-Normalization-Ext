import torch.nn as nn

from models.iin import InterpolatedInstanceNorm, PrefetchInterpolatedInstanceNorm
from models.kin import KernelizedInstanceNorm
from models.tin import ThumbInstanceNorm


def get_normalization_layer(out_channels, normalization='kin', parallelism=False):
    if normalization == 'iin':
        if parallelism:
            return PrefetchInterpolatedInstanceNorm(out_channels=out_channels)
        return InterpolatedInstanceNorm(out_channels=out_channels)
    elif normalization == 'kin':
        return KernelizedInstanceNorm(out_channels=out_channels)
    elif normalization == 'tin':
        return ThumbInstanceNorm(out_channels=out_channels)
    elif normalization == 'in':
        return nn.InstanceNorm2d(out_channels)
    else:
        raise NotImplementedError
