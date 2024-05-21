import torch.nn as nn

from models.dn import DenseInstanceNorm, PrefetchDenseInstanceNorm
from models.kin import KernelizedInstanceNorm
from models.tin import ThumbInstanceNorm


def get_normalization_layer(
    out_channels,
    normalization='kin',
    parallelism=False,
    interpolate_mode='bilinear',
):
    if normalization == 'dn':
        if parallelism:
            return PrefetchDenseInstanceNorm(
                out_channels=out_channels,
                interpolate_mode=interpolate_mode,
            )
        return DenseInstanceNorm(
            out_channels=out_channels,
            interpolate_mode=interpolate_mode,
        )
    elif normalization == 'kin':
        return KernelizedInstanceNorm(out_channels=out_channels)
    elif normalization == 'tin':
        return ThumbInstanceNorm(out_channels=out_channels)
    elif normalization == 'in':
        return nn.InstanceNorm2d(out_channels)
    else:
        raise NotImplementedError
