import os
import warnings
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention
import pdb


# 多层感知机
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 全连接层1
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 可分离卷积层
        self.dwconv = DWConv(hidden_features)
        # 激活层 
        self.act = act_layer()
        # 全连接层2
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 丢弃层
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    # 输入张量x、高度h、宽度w
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 多头自注意力机制
class Attention(nn.Module):
    # dim输入特征维度
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 测试
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

#  多头自注意力、多层感知机的组合模型。
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    # 它首先通过归一化层 norm1 对输入特征进行归一化，然后通过多头自注意力机制 attn 对输入特征进行注意力加权，
    # 并应用随机深度层 drop_path。接着，它通过另一个归一化层 norm2 对输入特征进行归一化，
    # 然后通过多层感知机模型 mlp 对输入特征进行非线性变换，并再次应用随机深度层 drop_path。最后，它返回输出张量 x。
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

# 实现了一个图像到补丁（patch）的嵌入（embedding）模型
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    # 输入图像 x 和掩码图像 mask（可选）。
    def forward(self, x, mask=None):
        # 它首先通过卷积层 proj 将输入图像转换为嵌入向量，然后根据是否使用掩码图像来决定是否进行掩码操作。
        x = self.proj(x)
        # Do a zero conv to get the mask
        # 如果使用掩码图像，则通过卷积层 mask_proj 将掩码图像转换为嵌入向量，并将嵌入向量与输入图像的嵌入向量相加。
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        # 它将嵌入向量进行展平（flatten）和转置（transpose）操作，以使其形状与后续的模型层兼容。
        x = x.flatten(2).transpose(1, 2)
        # 它通过层归一化层 norm 对嵌入向量进行归一化，并返回归一化的嵌入向量、输入图像在补丁化后的行数和列数。
        x = self.norm(x)

        return x, H, W

# 时间嵌入、一种基于正弦函数的时间步嵌入方法
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    """
    timesteps 是一个一维张量，表示模型在时间维度上的位置信息；
    dim 表示嵌入向量的维度；
    max_period 表示嵌入向量的最大周期。
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# 实现了一个金字塔结构的视觉Transformer模型
class PyramidVisionTransformerImpr(nn.Module):
    """
    输入图像的尺寸，默认为 224x224。补丁的尺寸，默认为 16x16。输入图像的通道数，默认为 3（RGB）。分类任务的类别数，默认为 1000。
    嵌入向量的维度列表，默认为 [64, 128, 256, 512]。自注意力头的数量列表，默认为 [1, 2, 4, 8]。
    多层感知机（MLP）的隐藏层维度与输入特征维度的比值列表，默认为 [4, 4, 4, 4]。
    每个阶段的块（block）数量列表，默认为 [3, 4, 6, 3]。每个阶段的空间降采样缩放因子列表，默认为 [8, 4, 2, 1]。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))
        # 图像到补丁的嵌入
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder 4次
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        """
        创建一个名为 block1 的模块列表对象，用于存储第一个阶段的块（block）子模块。
        该列表包含 depths[0] 个块子模块，每个块子模块都接受以下参数：输入特征维度、自注意力头的数量、
        多层感知机（MLP）的隐藏层维度与输入特征维度的比值、是否为查询、键和值的线性变换添加偏置、
        用于缩放查询和键的内积的缩放因子、丢弃层的丢弃率、用于自注意力概率分布的 dropout 率、
        随机深度（Stochastic Depth）的丢弃率、归一化层的类型和空间降采样缩放因子。
        """
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    # 输入图像x、时间步长、条件图像
    """
    包括将输入图像转换为嵌入向量序列、执行多头自注意力操作、执行多层感知机操作等。
    最后，该方法返回一个包含四个输出张量的列表，每个输出张量表示一个阶段的特征图。
    """
    def forward_features(self, x, timesteps, cond_img):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))
        time_token = time_token.unsqueeze(dim=1)

        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(cond_img, x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, timesteps, cond_img):
        x = self.forward_features(x, timesteps, cond_img)

        #        x = self.head(x[3])

        return x

# 深度可分离卷积
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

