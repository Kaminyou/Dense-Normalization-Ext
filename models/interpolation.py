import torch


class Interpolation3D:
    # input [C, 3, 3]
    # output [C, 512, 512]
    def __init__(self, channel, device="cuda"):
        self.channel = channel
        self.is_init = False
        self.device = device


    def init(self, size):
        if self.is_init:
            return

        self.size = size
        self.half_size = size // 2
        self.eps = 1e-7
        self.init_matrix()
        self.is_init = True

    def init_matrix(self):
        self.small_to_large = torch.arange(0.5, self.size + 0.5, 1).to(self.device) # [0.5, 511.5]
        self.large_to_small = torch.arange(self.size - 0.5, 0, -1).to(self.device) # [511.5, 0.5]
        # self.small_to_large = torch.arange(0.5, self.size + 0.5, 512).to(self.device) # [0.5, 511.5]
        # self.small_to_large = torch.repeat_interleave(self.small_to_large, repeats=512, dim=0)
        # self.large_to_small = torch.arange(self.size - 0.5, 0, -512).to(self.device) # [511.5, 0.5]
        # self.large_to_small = torch.repeat_interleave(self.large_to_small, repeats=512, dim=0)
        # each is [512, 512]
        # import torch.nn as nn
        # img = torch.Tensor([[1, 0], [1, 0]]).unsqueeze(0).unsqueeze(0).to(self.device)
        # img = nn.functional.interpolate(img, [512,512], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        # self.top_left = img
        # img = torch.Tensor([[0, 1], [0, 0]]).unsqueeze(0).unsqueeze(0).to(self.device)
        # img = nn.functional.interpolate(img, [512,512], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        # self.down_left = img
        # img = torch.Tensor([[0, 1], [0, 0]]).unsqueeze(0).unsqueeze(0).to(self.device)
        # img = nn.functional.interpolate(img, [512,512], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        # self.top_right = img
        # img = torch.Tensor([[0, 0], [0, 1]]).unsqueeze(0).unsqueeze(0).to(self.device)
        # img = nn.functional.interpolate(img, [512,512], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        # self.down_right = img

        self.top_left = ((self.large_to_small * self.large_to_small.unsqueeze(0).T) / self.size / self.size).contiguous()
        self.down_left = ((self.large_to_small * self.small_to_large.unsqueeze(0).T) / self.size / self.size).contiguous()
        self.top_right = ((self.small_to_large * self.large_to_small.unsqueeze(0).T) / self.size / self.size).contiguous()
        self.down_right = ((self.small_to_large * self.small_to_large.unsqueeze(0).T) / self.size / self.size).contiguous()
    def cpu_int(self, top_left_value, top_right_value, down_left_value, down_right_value):
        # from scipy.ndimage import zoom
        # import numpy as np
        # original_array = np.array([[float(top_left_value), float(top_right_value)], [float(down_left_value), float(down_right_value)]]) * self.size
        # upscaled_array = zoom(original_array, (self.half_size, self.half_size), order=1)  # order=1 for bilinear
        # upscaled_array /= self.size
        # return torch.Tensor(upscaled_array).to(self.device)
        import torch.nn as nn
        c, _, _ = top_left_value.shape
        img = torch.zeros((1, c, 2, 2))
        img[0, :, 0, 0] = top_left_value.cpu().squeeze()
        img[0, :, 0, 1] = top_right_value.cpu().squeeze()
        img[0, :, 1, 0] = down_left_value.cpu().squeeze()
        img[0, :, 0, 1] = down_right_value.cpu().squeeze()
        img = nn.functional.interpolate(img, [self.size, self.size], mode='bilinear', align_corners=True).squeeze(0).to(self.device)
        # img = torch.zeros((1, c, 2, 2)).to(self.device)
        # img[0, :, 0, 0] = top_left_value.squeeze()
        # img[0, :, 0, 1] = top_right_value.squeeze()
        # img[0, :, 1, 0] = down_left_value.squeeze()
        # img[0, :, 0, 1] = down_right_value.squeeze()
        # img = nn.functional.interpolate(img, [self.size, self.size], mode='bilinear', align_corners=True).squeeze(0)
        return img


    def top_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        # [C, 1, 1] * [512, 512] -> [C, 512, 512]
        # return self.cpu_int(top_left_value, top_right_value, down_left_value, down_right_value)[:, -self.half_size:, -self.half_size:]
        return top_left_value * self.top_left[-self.half_size:, -self.half_size:] + \
                top_right_value * self.top_right[-self.half_size:, -self.half_size:] + \
                down_left_value * self.down_left[-self.half_size:, -self.half_size:] + \
                down_right_value * self.down_right[-self.half_size:, -self.half_size:]

    def top_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        # return self.cpu_int(top_left_value, top_right_value, down_left_value, down_right_value)[:, -self.half_size:, :self.half_size]
        return top_left_value * self.top_left[-self.half_size:, :self.half_size] + \
                top_right_value * self.top_right[-self.half_size:, :self.half_size] + \
                down_left_value * self.down_left[-self.half_size:, :self.half_size] + \
                down_right_value * self.down_right[-self.half_size:, :self.half_size]

    def down_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        # return self.cpu_int(top_left_value, top_right_value, down_left_value, down_right_value)[:, :self.half_size, -self.half_size:]
        return top_left_value * self.top_left[:self.half_size, -self.half_size:] + \
                top_right_value * self.top_right[:self.half_size, -self.half_size:] + \
                down_left_value * self.down_left[:self.half_size, -self.half_size:] + \
                down_right_value * self.down_right[:self.half_size, -self.half_size:]

    def down_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        #return self.cpu_int(top_left_value, top_right_value, down_left_value, down_right_value)[:, :self.half_size, :self.half_size]
        return top_left_value * self.top_left[:self.half_size, :self.half_size] + \
                top_right_value * self.top_right[:self.half_size, :self.half_size] + \
                down_left_value * self.down_left[:self.half_size, :self.half_size] + \
                down_right_value * self.down_right[:self.half_size, :self.half_size]

    def _interpolation_mean_table(self, y0x0, y0x1, y0x2, y1x0, y1x1, y1x2, y2x0, y2x1, y2x2):
        # each input yaxb [C, 1, 1]
        table = torch.zeros((self.channel, self.size, self.size), device=self.device)
        table[:, :self.half_size, :self.half_size] = self.top_left_corner(y0x0, y0x1, y1x0, y1x1)
        table[:, :self.half_size, self.half_size:] = self.top_right_corner(y0x1, y0x2, y1x1, y1x2)
        table[:, self.half_size:, :self.half_size] = self.down_left_corner(y1x0, y1x1, y2x0, y2x1)
        table[:, self.half_size:, self.half_size:] = self.down_right_corner(y1x1, y1x2, y2x1, y2x2)

        return table

    def deal_with_inf(self, matrix_3x3):
        # to fill all the inf and nan with the middle values channel-wisely
        return torch.where(
            torch.logical_or(
                torch.isinf(matrix_3x3),
                torch.isnan(matrix_3x3),
            ),
            matrix_3x3[:, 1:2, 1:2],  # [C, 1, 1]
            matrix_3x3,  # [C, 3, 3]
        )

    def interpolation_mean_table(self, matrix_3x3): # [C, 3, 3] be on the same device
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)  # [C, 3, 3] -> [C, 3, 3, 1, 1]
        # matrix_3x3[:, 0, 0, :, :] => will be [C, 1, 1]
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )

    def interpolation_std_table_inverse(self, matrix_3x3): # [C, 3, 3] be on the same device
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)  # [C, 3, 3] -> [C, 3, 3, 1, 1]
        matrix_3x3 = 1 / (matrix_3x3 + self.eps)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )
