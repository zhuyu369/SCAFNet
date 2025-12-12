"""
Squeeze and Excitation Module
*****************************

Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture

    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_

"""
import math
from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# cSE Spatial Squeeze and Channel Excitation Block
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


# sSE 通道压缩与空间激励阻塞
class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


"""
这个类实现了通道空间并行的挤压-激励操作，通过同时考虑通道维度和空间维度的特征重要性，来增强模型对输入特征的建模能力。
"""


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)  # 通道
        self.sSE = SpatialSELayer(num_channels)  # 空间

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        return output_tensor


class Curvature(nn.Module):
    def __init__(self, ratio):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1 / 16, 5 / 16, -1 / 16], [5 / 16, -1, 5 / 16], [-1 / 16, 5 / 16, -1 / 16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio

    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p = p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio * C), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected


class Entropy_Hist(nn.Module):
    def __init__(self, ratio, win_w=3, win_h=3):
        super(Entropy_Hist, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.ratio = ratio

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-2, end_dim=-1)
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def histc_fork(self, ij):
        BINS = 256
        B, C = ij.shape
        N = 16
        BB = B // N
        min_elem = ij.min()
        max_elem = ij.max()
        ij = ij.view(N, BB, C)

        def f(x):
            with torch.no_grad():
                res = []
                for e in x:
                    res.append(torch.histc(e, bins=BINS, min=min_elem, max=max_elem))
                return res

        futures: List[torch.jit.Future[torch.Tensor]] = []

        for i in range(N):
            futures.append(torch.jit.fork(f, ij[i]))

        results = []
        for future in futures:
            results += torch.jit.wait(future)
        with torch.no_grad():
            out = torch.stack(results)
        return out

    def forward(self, img):
        with torch.no_grad():
            B, C, H, W = img.shape
            ext_x = int(self.win_w / 2)  # 考虑滑动窗口大小，对原图进行扩边，扩展部分长度
            ext_y = int(self.win_h / 2)

            new_width = ext_x + W + ext_x  # 新的图像尺寸
            new_height = ext_y + H + ext_y

            # 使用nn.Unfold依次获取每个滑动窗口的内容
            nn_Unfold = nn.Unfold(kernel_size=(self.win_w, self.win_h), dilation=1, padding=ext_x, stride=1)
            # 能够获取到patch_img，shape=(B,C*K*K,L),L代表的是将每张图片由滑动窗口分割成多少块---->28*28的图像，3*3的滑动窗口，分成了28*28=784块
            x = nn_Unfold(img)  # (B,C*K*K,L)
            x = x.view(B, C, 3, 3, -1).permute(0, 1, 4, 2, 3)  # (B,C*K*K,L) ---> (B,C,L,K,K)
            ij = self.calcIJ_new(x).reshape(B * C,
                                            -1)  # 计算滑动窗口内中心的灰度值和窗口内除了中心像素的灰度均值,(B,C,L,K,K)---> (B,C,L) ---> (B*C,L)
            fij_packed = self.histc_fork(ij)
            p = fij_packed / (new_width * new_height)
            h_tem = -p * torch.log(torch.clamp(p, min=1e-40)) / math.log(2)

            a = torch.sum(h_tem, dim=1)  # 对所有二维熵求和，得到这张图的二维熵
            H = a.reshape(B, C)

            _, index = torch.topk(H, int(self.ratio * C), dim=1)  # Nx3
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected
