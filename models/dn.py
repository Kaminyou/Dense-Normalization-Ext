import torch
import torch.nn as nn

from models.interpolation import Interpolation3D


class DenseInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True, device="cuda"):
        super(DenseInstanceNorm, self).__init__()

        # if use normal instance normalization during evaluation mode
        self.normal_instance_normalization = False

        # if collecting instance normalization mean and std
        # during evaluation mode'
        self.collection_mode = False

        self.out_channels = out_channels
        self.device = device
        self.interpolation3d = Interpolation3D(channel=out_channels, device=device)
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)

    def init_collection(self, y_anchor_num, x_anchor_num):
        # TODO: y_anchor_num => grid_height, x_anchor_num => grid_width
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.std_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )

    def pad_table(self, padding=1):
        # modify
        # padded table shape inconsisency
        # TODO: Don't permute the dimensions
        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(
            self.mean_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = pad_func(
            self.std_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def forward_normal(self, x):
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std  # * self.weight + self.bias
        return x

    def forward(self, x, y_anchor=None, x_anchor=None, padding=1):
        # TODO: Do not reply on self.training
        if self.training or self.normal_instance_normalization:
            _, _, h, w = x.shape
            self.interpolation3d.init(size=h)
            return self.forward_normal(x)

        else:
            assert y_anchor is not None
            assert x_anchor is not None

            if self.collection_mode:
                _, _, h, w = x.shape
                self.interpolation3d.init(size=h)
                x_std, x_mean = torch.std_mean(x, dim=(2, 3))  # [B, C]
                # x_anchor, y_anchor = [B], [B]
                # table = [H, W, C]
                # update std and mean to corresponing coordinates
                self.mean_table[y_anchor, x_anchor] = x_mean
                self.std_table[y_anchor, x_anchor] = x_std
                x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
                x_std = x_std.unsqueeze(-1).unsqueeze(-1)

                x = (x - x_mean) / x_std * self.weight + self.bias

            else:

                # currently, could support batch size = 1 for
                # kernelized instance normalization
                assert x.shape[0] == 1

                top = y_anchor
                down = y_anchor + 2 * padding + 1
                left = x_anchor
                right = x_anchor + 2 * padding + 1
                x_mean = self.padded_mean_table[
                    :, :, top:down, left:right
                ]  # 1, C, H, W
                x_std = self.padded_std_table[
                    :, :, top:down, left:right
                ]  # 1, C, H, W

                x_mean = self.interpolation3d.interpolation_mean_table(x_mean[0]).unsqueeze(0)
                x_std = self.interpolation3d.interpolation_std_table_inverse(x_std[0]).unsqueeze(0)


                x = (x - x_mean) * x_std * self.weight + self.bias
            return x


def not_use_dense_instance_norm(model):
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True


def init_dense_instance_norm(
    model, y_anchor_num, x_anchor_num,
):
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(
                y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num
            )


def use_dense_instance_norm(model, padding=1):
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.pad_table(padding=padding)
            layer.collection_mode = False
            layer.normal_instance_normalization = False


class PrefetchDenseInstanceNorm(nn.Module):
    def __init__(self, out_channels=None, affine=True, device="cuda"):
        super(PrefetchDenseInstanceNorm, self).__init__()

        # if use normal instance normalization during evaluation mode

        # if collecting instance normalization mean and std
        # during evaluation mode'

        self.out_channels = out_channels
        self.device = device
        self.interpolation3d = Interpolation3D(channel=out_channels, device=device)
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
        self.pad_func = nn.ReplicationPad2d((1, 1, 1, 1))

    def init_collection(self, y_anchor_num, x_anchor_num):
        # TODO: y_anchor_num => grid_height, x_anchor_num => grid_width
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.std_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.pad_table()

    def pad_table(self):
        # modify
        # padded table shape inconsisency
        # TODO: Don't permute the dimensions
        
        self.padded_mean_table = self.pad_func(
            self.mean_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = self.pad_func(
            self.std_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def forward_normal(self, x):
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std  # * self.weight + self.bias
        return x

    def forward(
        self,
        x,
        y_anchor=None,
        x_anchor=None,
        padding=1,
        pre_y_anchor=None,
        pre_x_anchor=None,
    ):
        real_x, pre_x = torch.chunk(x, 2, dim=0)
        # do caching
        if pre_y_anchor != -1:
            _, _, h, _ = pre_x.shape
            self.interpolation3d.init(size=h)
            pre_x_std, pre_x_mean = torch.std_mean(pre_x, dim=(2, 3))  # [B, C]
            # x_anchor, y_anchor = [B], [B]
            # table = [H, W, C]
            # update std and mean to corresponing coordinates
            self.mean_table[pre_y_anchor, pre_x_anchor] = pre_x_mean
            self.std_table[pre_y_anchor, pre_x_anchor] = pre_x_std

            self.padded_mean_table[:, :, pre_y_anchor + 1, pre_x_anchor + 1]  = pre_x_mean.squeeze(0)
            self.padded_std_table[:, :, pre_y_anchor + 1, pre_x_anchor + 1]  = pre_x_std.squeeze(0)
            pre_x_mean = pre_x_mean.unsqueeze(-1).unsqueeze(-1)
            pre_x_std = pre_x_std.unsqueeze(-1).unsqueeze(-1)

            pre_x = (pre_x - pre_x_mean) / pre_x_std * self.weight + self.bias

        if y_anchor != -1:
            top = y_anchor
            down = y_anchor + 2 * padding + 1
            left = x_anchor
            right = x_anchor + 2 * padding + 1
            x_mean = self.padded_mean_table[
                :, :, top:down, left:right
            ].squeeze(0)  # 1, C, H, W
            
            x_std = self.padded_std_table[
                :, :, top:down, left:right
            ].squeeze(0)

            x_mean_expand = x_mean[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            x_std_expand = x_std[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            x_mean = torch.where(x_mean == 0, x_mean_expand, x_mean)
            x_std = torch.where(x_std == 0, x_std_expand, x_std)

            x_mean = self.interpolation3d.interpolation_mean_table(x_mean).unsqueeze(0)
            x_std = self.interpolation3d.interpolation_std_table_inverse(x_std).unsqueeze(0)

            real_x = (real_x - x_mean) * x_std * self.weight + self.bias
        x = torch.cat((real_x, pre_x), dim=0)
        return x


def init_prefetch_dense_instance_norm(
    model, y_anchor_num, x_anchor_num,
):
    for _, layer in model.named_modules():
        if isinstance(layer, PrefetchDenseInstanceNorm):
            layer.init_collection(
                y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num
            )
