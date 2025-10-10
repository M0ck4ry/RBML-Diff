import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math, copy

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # return out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # return x
        return self.sigmoid(x)


class TransferConv_m21(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=1),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c, in_c * 2, kernel_size=1, padding=1),
            nn.BatchNorm2d(in_c * 2
                           , momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class TransferConv_h21(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c * 2, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c * 2, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class IFC21(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.ca_1 = ChannelAttention(128)
        self.sa_1 = SpatialAttention()
        self.ca_2 = ChannelAttention(64)
        self.sa_2 = SpatialAttention()


        self.transferconv_h = TransferConv_h21(in_c)
        self.transferconv_m = TransferConv_m21(in_c)
    def reconstructing_procedure(self, f_h, f_m):
        _, c, h, w = f_h.shape      # 16 512 8 8 
        f_m = f_m.view(f_m.size(0), f_m.size(1), -1)    # 16 512 64
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)

        f_m_T = torch.transpose(f_m, 2, 1)              # 16 64 512     16 512 64
        matrix_hm = torch.matmul(f_m_T, f_h)            # 16 64 64      矩阵乘法
        l2_m = torch.norm(matrix_hm)                    # 
        matrix_hm = torch.tanh(matrix_hm / l2_m)        # # 16 64 64
        f_refine_h = torch.matmul(f_m, matrix_hm) + f_h
        return f_refine_h.view(-1, c, h, w)
        # f_refine_h_T = torch.transpose(f_refine_h, 2, 1)
        # matrix_mh = torch.matmul(f_refine_h_T, f_m)
        # l2_h = torch.norm(matrix_mh)
        # matrix_mh = torch.tanh(matrix_mh / l2_h)
        # f_refine_m = torch.matmul(f_refine_h, matrix_mh) + f_m
        # return f_refine_h.view(-1, c, h, w), f_refine_m.view(-1, c, h, w)

    def forward(self, f_h, f_m):
        # f_h  = self.convmix_h(f_h)
        # f_m  = self.convmix_m(f_m)
        # f_h=self.ca_1(f_h)*f_h
        # f_h=self.sa_1(f_h)*f_h

        # f_m=self.ca_2(f_m)*f_m
        # f_m=self.sa_2(f_m)*f_m

        # f_m = self.etmm(f_m)
        # f_h = self.etmh(f_h)

        f_h = self.transferconv_h(f_h)
        f_m = self.transferconv_m(f_m)
        f_refine_h= self.reconstructing_procedure(f_h, f_m)
        return f_refine_h
        
# 321
class TransferConv_m321(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c*2, in_c*2, kernel_size=1, padding=1),
            nn.BatchNorm2d(in_c*2, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c*2 , in_c * 5 , kernel_size=1, padding=1),
            nn.BatchNorm2d(in_c * 5, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class TransferConv_h321(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c * 5, in_c * 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c * 5, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c * 5, in_c * 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c * 5, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class IFC321(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.ca_1 = ChannelAttention(320)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()
    

        self.transferconv_h = TransferConv_h321(in_c)
        self.transferconv_m = TransferConv_m321(in_c)
        

    def reconstructing_procedure(self, f_h, f_m):
        _, c, h, w = f_h.shape      # 16 512 8 8 
        # 重塑
        f_m = f_m.view(f_m.size(0), f_m.size(1), -1)    # 16 512 64
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)

        f_m_T = torch.transpose(f_m, 2, 1)              # 16 64 512     16 512 64
        matrix_hm = torch.matmul(f_m_T, f_h)            # 16 64 64      矩阵乘法
        l2_m = torch.norm(matrix_hm)                    # 
        matrix_hm = torch.tanh(matrix_hm / l2_m)        # # 16 64 64
        f_refine_h = torch.matmul(f_m, matrix_hm) + f_h
        return f_refine_h.view(-1, c, h, w)
        # f_refine_h_T = torch.transpose(f_refine_h, 2, 1)
        # matrix_mh = torch.matmul(f_refine_h_T, f_m)
        # l2_h = torch.norm(matrix_mh)
        # matrix_mh = torch.tanh(matrix_mh / l2_h)
        # f_refine_m = torch.matmul(f_refine_h, matrix_mh) + f_m
        # return f_refine_h.view(-1, c, h, w), f_refine_m.view(-1, c, h, w)

    def forward(self, f_h, f_m):
        # f_m = self.etmm(f_m)
        # f_h = self.etmh(f_h)

        f_h = self.transferconv_h(f_h)
        f_m = self.transferconv_m(f_m)
        
        f_refine_h= self.reconstructing_procedure(f_h, f_m)
        return f_refine_h

# 4321
class TransferConv_m4321(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c*5, in_c*5, kernel_size=1, padding=1),
            nn.BatchNorm2d(in_c*5, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c*5 , in_c * 8 , kernel_size=1, padding=1),
            nn.BatchNorm2d(in_c * 8, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class TransferConv_h4321(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c * 8, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c * 8, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

class IFC4321(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.ca_1 = ChannelAttention(512)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(320)
        self.sa_2 = SpatialAttention()
                         
        self.transferconv_h = TransferConv_h4321(in_c)
        self.transferconv_m = TransferConv_m4321(in_c)
        

    def reconstructing_procedure(self, f_h, f_m):
        _, c, h, w = f_h.shape      # 16 512 8 8 
        f_m = f_m.view(f_m.size(0), f_m.size(1), -1)    # 16 512 64
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)

        f_m_T = torch.transpose(f_m, 2, 1)              # 16 64 512     16 512 64
        matrix_hm = torch.matmul(f_m_T, f_h)            # 16 64 64      矩阵乘法
        l2_m = torch.norm(matrix_hm)                    # 
        matrix_hm = torch.tanh(matrix_hm / l2_m)        # # 16 64 64
        f_refine_h = torch.matmul(f_m, matrix_hm) + f_h
        return f_refine_h.view(-1, c, h, w)
        # f_refine_h_T = torch.transpose(f_refine_h, 2, 1)
        # matrix_mh = torch.matmul(f_refine_h_T, f_m)
        # l2_h = torch.norm(matrix_mh)
        # matrix_mh = torch.tanh(matrix_mh / l2_h)
        # f_refine_m = torch.matmul(f_refine_h, matrix_mh) + f_m
        # return f_refine_h.view(-1, c, h, w), f_refine_m.view(-1, c, h, w)

    def forward(self, f_h, f_m):

        f_h = self.transferconv_h(f_h)
        f_m = self.transferconv_m(f_m)
        
        f_refine_h= self.reconstructing_procedure(f_h, f_m)
        return f_refine_h

class IFI(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(IFI , self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
            
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel*3, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3), dim=1))

        x = self.relu(x0 + x_cat)
        return x

        

class RBML(nn.Module):
    def __init__(self, in_channels):
        super(RBML, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        
        pred = torch.sigmoid(pred)
        
        #reverse attention 
        background_att = 1 - pred
        background_x= x * background_att
        
        #boudary attention
        edge_pred = make_laplace(pred, 1)  
        pred_feature = x * edge_pred

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.ca(out)  * out
        out = self.sa(out) * out
        return out

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

