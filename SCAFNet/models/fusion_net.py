"""
fusion_net：自注意力和交叉注意力的结合
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from .model1 import Attention, PatchEmbed, DePatch, Mlp, Block

EPSILON = 1e-10


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(Layer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = ScConv(in_channels, out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        output = x * output
        return output


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        output = x * output
        return output


class CSAMBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.CAB = ChannelAttention(channel=channel)
        self.SAB = SpatialAttention()
        block = []
        block += [Layer(2 * channel, channel, 1, 1),
                  Layer(channel, channel, 3, 1),
                  Layer(channel, channel, 3, 1)]
        self.blocks = nn.Sequential(*block)

    def forward(self, x1, x2):
        sa_out1 = self.SAB(x1)
        sa_out2 = self.SAB(x2)
        cf1 = sa_out1 + sa_out2
        ca_out1 = self.CAB(x1)
        ca_out2 = self.CAB(x2)
        cf2 = ca_out1 + ca_out2
        out = torch.cat([cf1, cf2], 1)
        out = self.blocks(out)
        return out


class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.body = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


def CMDAF(F_vi, F_ir):
    sub_vi_ir = F_vi - F_ir
    sub_w_vi_ir = F.avg_pool2d(sub_vi_ir, (sub_vi_ir.size(2), sub_vi_ir.size(3)))  # Global Average Pooling
    w_vi_ir = torch.sigmoid(sub_w_vi_ir)

    sub_ir_vi = F_ir - F_vi
    sub_w_ir_vi = F.avg_pool2d(sub_ir_vi, (sub_ir_vi.size(2), sub_ir_vi.size(3)))  # Global Average Pooling
    w_ir_vi = torch.sigmoid(sub_w_ir_vi)

    F_dvi = w_vi_ir * sub_ir_vi
    F_dir = w_ir_vi * sub_vi_ir

    return F_dvi, F_dir


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # ==Dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_list = self.attn(x1)  # x,q,k,v
        attn = attn_list[0]
        x1 = self.drop_path(attn)
        x = x + x1
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SelfAndCrossAttention(nn.Module):
    def __init__(self, patch_size, dim, num_heads, channel, proj_drop, depth, qk_scale, attn_drop):
        super().__init__()

        self.patchembed1 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)
        self.patchembed2 = PatchEmbed(patch_size=patch_size, in_c=channel, embed_dim=dim)

        self.TransformerEncoderBlocks1 = nn.Sequential(*[
            TransformerEncoderBlock(dim=dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.TransformerEncoderBlocks2 = nn.Sequential(*[
            TransformerEncoderBlock(dim=dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.QKV_Block1 = Block(dim=dim, num_heads=num_heads)
        self.QKV_Block2 = Block(dim=dim, num_heads=num_heads)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        self.depatch = DePatch(channel=channel, embed_dim=dim, patch_size=patch_size)

    def forward(self, in_1, in_2):
        # Patch Embeding1
        in_emb1 = self.patchembed1(in_1)
        B, N, C = in_emb1.shape

        # Transformer Encoder1
        in_emb1 = self.TransformerEncoderBlocks1(in_emb1)

        # cross self-attention Feature Extraction
        _, q1, k1, v1 = self.QKV_Block1(in_emb1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        # Patch Embeding2
        in_emb2 = self.patchembed2(in_2)

        # Transformer Encoder2
        in_emb2 = self.TransformerEncoderBlocks2(in_emb2)

        _, q2, k2, v2 = self.QKV_Block2(in_emb2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        # cross attention
        x_attn1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x_attn1 = self.proj1(x_attn1)
        x_attn1 = self.proj_drop1(x_attn1)

        x_attn2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x_attn2 = self.proj2(x_attn2)
        x_attn2 = self.proj_drop2(x_attn2)

        x_attn = (x_attn1 + x_attn2) / 2

        # Patch Rearrange
        ori = in_2.shape  # b,c,h,w
        out1 = self.depatch(x_attn, ori)

        out = in_1 + in_2 + out1

        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = Layer(2 * channels, channels, ws[index], 1)
        self.conv_diff = Layer(channels, channels, ws[index], 1)

        self.conv_ir = Layer(channels, channels, ws[index], 1)
        self.conv_vi = Layer(channels, channels, ws[index], 1)

        self.CSAB = CSAMBlock(channels)
        self.sca1 = SelfAndCrossAttention(patch_size=16, dim=channels, num_heads=8, channel=channels, depth=3,
                                          qk_scale=None,
                                          attn_drop=0., proj_drop=0.)
        self.sca2 = SelfAndCrossAttention(patch_size=16, dim=channels, num_heads=8, channel=channels, depth=3,
                                          qk_scale=None,
                                          attn_drop=0., proj_drop=0.)
        self.self_attn_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=7 // 2, groups=channels, bias=False)
        )
        self.cross_attn_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=7 // 2, groups=channels, bias=False)
        )
        self.ch_attn = Mlp(channels, channels // 4, channels * 2)

    def forward(self, x1, x2):
        f_cat = torch.cat([x1, x2], 1)
        f_init = self.conv_fusion(f_cat)

        d1, d2 = CMDAF(x1, x2)
        x1 = x1 + d2
        x2 = x2 + d1
        # 自注意力
        out_sa = self.CSAB(x1, x2)
        # 交叉注意力
        out_ca = self.sca1(x1, x2)
        sa_attn = self.self_attn_fusion(out_sa)
        ca_attn = self.cross_attn_fusion(out_ca)
        # blocking artifacts
        deep_fusion_feats = sa_attn + ca_attn
        B, C, H, W = deep_fusion_feats.shape
        deep_fusion_feats = F.adaptive_avg_pool2d(deep_fusion_feats, output_size=1)
        deep_fusion_feats = deep_fusion_feats.squeeze(-1).squeeze(-1)
        deep_fusion_feats = self.ch_attn(deep_fusion_feats).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(
            -1).unsqueeze(-1)
        deep_fusion_feats = deep_fusion_feats[0] * sa_attn + deep_fusion_feats[1] * ca_attn
        # out = self.sca2(out_ca, out)

        out = f_init + deep_fusion_feats
        return out


# Fusion network, 4 groups of features
class Fusion_network(nn.Module):
    def __init__(self, nC):
        super(Fusion_network, self).__init__()
        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


if __name__ == '__main__':
    en = []
    a1 = torch.randn(1, 1, 256, 256)
    a2 = torch.randn(1, 1, 256, 256)
    x0 = torch.randn(1, 64, 256, 256)
    x1 = torch.randn(1, 128, 128, 128)
    x2 = torch.randn(1, 256, 64, 64)
    x3 = torch.randn(1, 512, 32, 32)
    en.append(x0)
    en.append(x1)
    en.append(x2)
    en.append(x3)
    imgs = torch.cat([a1, a2], dim=1)
    fs = Fusion_network([64, 128, 256, 512])
    out = fs(en, en, imgs)
    print(out[0].shape)
