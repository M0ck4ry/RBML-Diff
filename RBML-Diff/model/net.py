import os
import warnings
from functools import partial
import math
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from model.RBML import *
from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention
import pdb
from model.pvt import *
from timm.models.layers import DropPath
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv(nn.Module):
    """
    Linear Embedding
    """
    # 输入维度、嵌入维度、卷积核大小
    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()
        # 一个序列proj
        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        # 卷积操作
        x = self.proj(x)
        # 然后，使用 flatten 方法将特征图从四维形状转换为一个二维形状 (batch size, channels * height * width)。
        # 接着，使用 transpose 方法将特征图从形状 (batch size, channels * height * width) 转换为一个形状 (channels * height * width, batch size)。
        # 最后，返回特征图作为输出。
        x = x.flatten(2).transpose(1, 2)
        return x


# 上采样
class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2        # 翻倍
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            # 像素洗牌层
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            BasicConv2d(in_channels, in_channels // 4), 
            BasicConv2d(in_channels // 4, out_channels)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)


# 解码器
class Decoder(Module):
    # 输入特征维度，4维元组。嵌入向量的维度。分类，掩码通道
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        # 将嵌入向量的维度存储为变量 embedding_dim。
        embedding_dim = dim
        #  将时间嵌入向量的维度存储为变量 time_embed_dim。
        self.time_embed_dim = embedding_dim
        # 定义了一个时间嵌入子模块 time_embed，
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        # 这个子模块是一个残差块（Residual Block），用于构建残差网络（ResNet）模型。
        resnet_block = partial(ResnetBlock, groups=8)

        # 进行下采样操作，输入特征尺寸减半
        self.down = nn.Sequential(
            # 1,256,256
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=2,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=2,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        
        self.ifi1 = IFI(64,64)
        self.ifi2 = IFI(128,128)
        self.ifi3 = IFI(320,320)
        self.ifi4 = IFI(512,512)


        self.ifc21 = IFC21(64)
        self.ifc321 = IFC321(64)
        self.ifc4321 = IFC4321(64)

        self.up4 = Up(768, 320)
        self.up3 = Up(640, 128)
        self.up2 = Up(256, 64)
        self.up1 = Up(128, 64)

        self.rbml1 = RBML(64)
        self.rbml2 = RBML(128)
        self.rbml3 = RBML(320)

        self.out4 = Out(320, 1)
        self.out3 = Out(128, 1)
        self.out2 = Out(64, 1)
        self.out1 = Out(64, 1)

    def forward(self, inputs, timesteps, x, edge_feature):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        c1, c2, c3, c4 = inputs
        _x = [x]
        # 下采样
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)      # 32 256 16 16

        c1 = self.ifi1(c1)  # 32 64 64 64 
        c2 = self.ifi2(c2)  # 32 128 32 32
        c3 = self.ifi3(c3)  # 32 320 16 16
        c4 = self.ifi4(c4)  # 32 512 8 8

        e3 = self.ifc321(c3,c2)
        e4 = self.ifc4321(c4,c3)

        c4 = F.interpolate(e4, size=c3.size()[2:], mode='bilinear', align_corners=True) # 32 512 16 16
        d4 = self.up4(c4, x)        # 320-32-32         
        out4 = self.out4(d4)        # 1-32-32       
        c3 = F.interpolate(e3, size=c2.size()[2:], mode='bilinear', align_corners=True)
        rbml3 = self.rbml3(edge_feature, c3, out4)   # ([32, 320, 32, 32])
        

        d3 = self.up3(d4, rbml3) #([32, 128, 64, 64])
        out3 = self.out3(d3)    # torch.Size([32, 1, 64, 64])
        c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=True) # ([32, 128, 64, 64])
        rbml2 = self.rbml2(edge_feature, c2, out3)        # ([32, 128, 64, 64])

        d2 = self.up2(d3, rbml2) # ([32, 64, 128, 128])
        out2 = self.out2(d2)    # ([32, 1, 128, 128])
        c1 = F.interpolate(c1, size=128, mode='bilinear', align_corners=True)   # ([32, 64, 128, 128])
        rbml1 = self.rbml1(edge_feature, c1, out2)    # ([32, 64, 128, 128])

        d1 = self.up1(d2, rbml1)
        out1 = self.out1(d1)
        return out1, out2, out3, out4


class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        self.backbone = pvt_v2_b4_m(in_chans=3, mask_chans=mask_chans)
        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=class_num, mask_chans=mask_chans)
        self._init_weights()  # load pretrain

    def forward(self, x, timesteps, cond_img):
        grayscale_img = rgb_to_grayscale(cond_img)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        features = self.backbone(x, timesteps, cond_img)
        features, layer1, layer2, layer3= self.decode_head(features, timesteps, x, edge_feature)
        return features

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        model_path = '/opt/data/private/cjl/RBML-Diff/pretrained_weights/pvt_v2_b4_m.pth'
        pretrained_dict = torch.load(model_path) #for save mem
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        return self.forward(x, timesteps, cond_img)

    def extract_features(self, cond_img):
        # do nothing
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass