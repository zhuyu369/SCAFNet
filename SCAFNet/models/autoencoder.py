import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.fusion_net import Fusion_network, Layer
from .se import ChannelSpatialSELayer


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CSEBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False,
                 act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# image information auxiliary module
class IAM(nn.Module):
    def __init__(self, n_feats=32, n_encoder_res=4):
        super(IAM, self).__init__()
        E1 = [nn.Conv2d(2, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [ResBlock(
            default_conv, n_feats, kernel_size=3
        ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(*E)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class Dense2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Dense2d, self).__init__()
        self.dense_conv = Layer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.dense_conv_2_1 = Layer(in_channels, out_channels, 3, 1)
        self.dense_conv_2_2 = Layer(in_channels, out_channels, 3, 1)
        self.dense_conv_2_3 = Layer(in_channels, out_channels, 3, 1)

    def forward(self, x):
        out_1 = self.dense_conv_2_1(x)
        out_2 = self.dense_conv_2_2(out_1)
        out = self.dense_conv_2_3(x + out_2)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        denseblock = []
        denseblock += [Dense2d(in_channels, out_channels, 3, 1),
                       Dense2d(in_channels + out_channels, out_channels, 3, 1),
                       Dense2d(in_channels + out_channels * 2, out_channels, 3, 1)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class GRDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GRDB, self).__init__()
        self.dense = DenseBlock(in_channels, out_channels)
        self.res = ResnetBlock(in_channels, out_channels)
        self.conv1 = Block(4 * out_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.conv1(x1)
        x2 = self.res(x)
        x3 = x1 + x2
        return x3


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    r"""DPB module

    Use a MLP to predict position bias used in attention.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(
                self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(
                1 - self.window_size[0], self.window_size[0])
            position_bias_w = torch.arange(
                1 - self.window_size[1], self.window_size[1])
            biases = torch.stack(torch.meshgrid(
                [position_bias_h, position_bias_w], indexing='ij'))  # 2, 2Wh-1, 2Ww-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases, persistent=False)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, k_v, mask=None):
        B_, N, C = x.shape
        # k_v = self.kernel(k_v).view(-1, C * 2, 1, 1)
        # k_v1, k_v2 = k_v.chunk(2, dim=1)
        # x = x.view(B_, C, int(math.sqrt(N)), int(math.sqrt(N)))  # 假设H*W是某个数的平方
        # x = x * k_v1 + k_v2
        # x = x.view(B_, -1, C)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.position_bias:
            pos = self.pos(self.biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.window_size[0] *
                self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), which maps query
            y: input features with shape of (num_windows*B, N, C), which maps key and value
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C //
                              self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = q[0], kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, k_v, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            # nW*B, window_size*window_size, C
            attn_windows = self.attn(x_windows, k_v, mask=self.attn_mask)
        else:
            attn_windows = self.attn(
                x_windows, k_v, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Cross_SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_B = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut_A = x
        shortcut_B = y
        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(
                y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # nW*B, window_size, window_size, C
        y_windows = window_partition(shifted_y, self.window_size)
        # nW*B, window_size*window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            # nW*B, window_size*window_size, C
            attn_windows_A = self.attn_A(
                x_windows, y_windows, mask=self.attn_mask)
            # nW*B, window_size*window_size, C
            attn_windows_B = self.attn_B(
                y_windows, x_windows, mask=self.attn_mask)
        else:
            attn_windows_A = self.attn_A(
                x_windows, y_windows, mask=self.calculate_mask(x_size).to(x.device))
            attn_windows_B = self.attn_B(
                y_windows, x_windows, mask=self.calculate_mask(x_size).to(y.device))

        # merge windows
        attn_windows_A = attn_windows_A.view(-1,
                                             self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1,
                                             self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows_A, self.window_size, H, W)  # B H' W' C
        shifted_y = window_reverse(
            attn_windows_B, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        # FFN
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, k_v, x_size):
        for blk in self.blocks:
            x = blk(x, k_v, x_size)
        return x


class Cross_BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            Cross_SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                       num_heads=num_heads, window_size=window_size,
                                       shift_size=0 if (
                                           i % 2 == 0) else window_size // 2,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path[i] if isinstance(
                                           drop_path, list) else drop_path,
                                       norm_layer=norm_layer)
            for i in range(depth)])

    # 以x为主, y只是辅助x的特征提取
    def forward(self, x, y, x_size):
        for blk in self.blocks:
            x, y = blk(x, y, x_size)
        return x, y


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim,
                                   x_size[0], x_size[1])  # B Ph*Pw C
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, num_heads=4, input_resolution=(256, 256), depth=2,
                 window_size=8, img_size=256, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, patch_size=4):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, k_v, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, k_v, x_size), x_size))) + x


class CRSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth=2, num_heads=4, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(CRSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.dim_temp = int(dim / 2)

        self.residual_group = Cross_BasicLayer(dim=dim,
                                               input_resolution=input_resolution,
                                               depth=depth,
                                               num_heads=num_heads,
                                               window_size=window_size,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               drop=drop, attn_drop=attn_drop,
                                               drop_path=drop_path,
                                               norm_layer=norm_layer)
        self.residual_group_A = BasicLayer(dim=dim,
                                           input_resolution=input_resolution,
                                           depth=depth,
                                           num_heads=num_heads,
                                           window_size=window_size,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer)
        self.residual_group_B = BasicLayer(dim=dim,
                                           input_resolution=input_resolution,
                                           depth=depth,
                                           num_heads=num_heads,
                                           window_size=window_size,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, y, x_size):
        x = self.residual_group_A(x, x_size) + x
        y = self.residual_group_B(y, x_size) + y
        x1 = x
        y1 = y
        x, y = self.residual_group(x1, y1, x_size)
        x = x + x1
        y = y + y1
        out = torch.cat([x, y], dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class Block1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Block1, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.batch_norm(out)
        out = F.relu(out, inplace=True)
        return out


class GLBlock(nn.Module):
    def __init__(self, dim, image_size):
        super().__init__()
        self.image_size = image_size
        self.rstb = RSTB(dim)
        self.grdb = GRDB(dim, dim)
        self.seblock = CSEBlock(dim * 2, dim)
        self.kernel = nn.Linear(256, dim * 2, bias=False)

    def forward(self, x, k_v):
        C = x.shape[1]
        k_v = self.kernel(k_v).view(-1, C * 2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x * k_v1 + k_v2
        x1 = torch.flatten(x, 2).permute(0, 2, 1)
        x1 = self.rstb(x1, k_v, (self.image_size, self.image_size))
        x1 = x1.permute(0, 2, 1).view(-1,
                                      x.shape[-3], x.shape[-2], x.shape[-1])
        x1 += x

        x2 = self.grdb(x)
        # x_cat = torch.cat((x1, x2), dim=1)
        # output = self.seblock(x_cat)
        output = x1 + x2

        return output


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hid_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hid_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(hid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        out = self.act(x2)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=64):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        image_size = [256, 128, 64, 32]
        self.encoder_level1 = GLBlock(dim=dim, image_size=image_size[0])

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = GLBlock(
            dim=int(dim * 2 ** 1), image_size=image_size[1])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = GLBlock(
            dim=int(dim * 2 ** 2), image_size=image_size[2])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  # From Level 3 to Level 4
        self.encoder_level4 = GLBlock(
            dim=int(dim * 2 ** 3), image_size=image_size[3])

    def forward(self, x, k_v):
        x_down = []
        inp_enc_level1 = self.init_conv(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, k_v)
        x_down.append(out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, k_v)
        x_down.append(out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, k_v)
        x_down.append(out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4, k_v)
        x_down.append(out_enc_level4)

        return out_enc_level4, x_down


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, bias=False):
        super(Decoder, self).__init__()
        self.up4_3 = Upsample(int(dim * 2 ** 3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = DecoderBlock(
            int(dim * 2 ** 2), int(dim * 2 ** 2))

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = DecoderBlock(
            int(dim * 2 ** 1), int(dim * 2 ** 1))

        # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = DecoderBlock(
            int(dim * 2 ** 1), int(dim * 2 ** 1))

        # self.refinement = DecoderBlock(int(dim * 2 ** 1), int(dim * 2 ** 1))

        modules_tail = [default_conv(int(dim * 2 ** 1), out_channels, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, x_down):
        inp_dec_level3 = self.up4_3(x)
        inp_dec_level3 = torch.cat([inp_dec_level3, x_down[2]], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, x_down[1]], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, x_down[0]], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.tail(out_dec_level1)

        return out_dec_level1


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode1 = Encoder()
        self.encode2 = Encoder()
        self.decode = Decoder()
        self.E = IAM(n_feats=64, n_encoder_res=6)
        self.fusion = Fusion_network(nC=[64, 128, 256, 512])

    def encoder(self, x, x1):
        k_v = self.E(x, x1)
        x, x_down = self.encode1(x, k_v)
        x1, x_down1 = self.encode2(x1, k_v)
        return x, x_down, x1, x_down1

    def decoder(self, x, x_down):
        x = self.decode(x, x_down)
        return x

    def forward(self, x, x1):
        x, x_down, x1, x_down1 = self.encoder(x, x1)
        fusion = self.fusion(x_down, x_down1)
        x = self.decoder(fusion[3], fusion)
        return x


if __name__ == '__main__':
    input = torch.randn(1, 1, 256, 256).cuda()
    input1 = torch.randn(1, 1, 256, 256).cuda()
    en = Autoencoder().cuda()
    output = en(input, input1)
    # down = Upsample(64)
    # output = down(input)
    print(output.shape)
