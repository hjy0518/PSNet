import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from models.mobilenetv2 import mobilenetv2
from models.smt import smt_t
import torch.nn.functional as F
import os
# import onnx

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class PSNet(nn.Module):
    def __init__(self):
        super(PSNet, self).__init__()

        self.rgb = smt_t()
        self.depth = mobilenetv2()
        self.mffm1 = MFFM(64,24)
        self.mffm2 = MFFM(128,32)
        self.mffm3 = MFFM(256,96)
        self.mffm4 = MFFM(512,320)
        self.mffms = [self.mffm1,self.mffm2,self.mffm3,self.mffm4]
        self.decode = Decode(64,128,256,512)
    def forward(self, r,d):

        depth_maps = self.depth(d)
        B = r.shape[0]
        fuses = []
        for i in range(4):
            patch_embed = getattr(self.rgb, f"patch_embed{i + 1}")
            block = getattr(self.rgb, f"block{i + 1}")
            norm = getattr(self.rgb, f"norm{i + 1}")
            r, H, W = patch_embed(r)
            for blk in block:
                r = blk(r, H, W)
            r = norm(r)
            r = r.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            r,d,fuse = self.mffms[i](r,depth_maps[i])
            fuses.append(fuse)
        pred1,pred2,pred3,pred4 = self.decode(fuses[0],fuses[1],fuses[2],fuses[3],384)
        return pred1,pred2,pred3,pred4
    def load_pre(self, pre_model_r,pre_model_d):
        self.rgb.load_state_dict(torch.load(pre_model_r)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model_r}")

        pretrained_dict = torch.load(pre_model_d)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.depth.state_dict()}
        self.depth.load_state_dict(pretrained_dict)
        print(f"Depth PyramidVisionTransformerImpr loading pre_model ${pre_model_d}")

class MFFM(nn.Module):
    def __init__(self, dim_r,dim_d):
        super(MFFM, self).__init__()
        self.DtoR = nn.Conv2d(dim_d,dim_r,kernel_size=1,stride=1)
        self.rca = ChannelAttention(dim_r)
        self.dca = ChannelAttention(dim_r)
        self.sa = SpatialAttention()
        self.catt = CrossAtt(dim_r*2,dim_r)

    def forward(self,r,d):

        d = self.DtoR(d)
        rca = self.rca(r) * r
        dca = self.dca(d) * d
        x = torch.cat((rca,dca),1)
        sa = self.sa(x)
        out_r = rca.mul(sa)
        out_d = dca.mul(sa)
        out = self.catt(torch.cat((out_r,out_d),1))
        return out_r,out_d,out

class CrossAtt(nn.Module):
    def __init__(self, inp, oup, reduction=8):
        super(CrossAtt, self).__init__()
        self.conv_end = ChannelEmbed(inp, oup)

        mip = max(8, inp // reduction)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(oup, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, oup, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.self_SA_Enhance = SpatialAttention()
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1,bias=False)
    def forward(self, rgb):
        x = self.conv_end(rgb)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(max_out) * x
        sa = self.self_SA_Enhance(out)
        sa = self.sa_conv(sa)
        out = out.mul(sa)
        return out


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = nn.Sequential(
            nn.Conv2d(in_channels=in4, out_channels=in3, kernel_size=1, bias=False),
            nn.BatchNorm2d(in3),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=in3*2, out_channels=in2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in2),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=in2*2, out_channels=in1, kernel_size=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in1*2, out_channels=in1, kernel_size=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2
        )

        # self.upb4 = Block(in3)
        self.upb3 = Block(in2)
        self.upb2 = Block(in1)
        self.upb1 = Block(in1)


        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.p2 = nn.Conv2d(in1, 1, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(in2, 1, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(in3, 1, kernel_size=3, padding=1)

    def forward(self,x1,x2,x3,x4,s):
        up4 = self.conv_up4(x4)
        up3 = self.upb3(self.conv_up3(torch.cat((up4,x3),1)))
        up2 = self.upb2(self.conv_up2(torch.cat((up3,x2),1)))
        up1 = self.upb1(self.conv_up1(torch.cat((up2,x1), 1)))

        pred1 = self.p_1(up1)
        pred2 = F.interpolate(self.p2(up2), size=s, mode='bilinear')
        pred3 = F.interpolate(self.p3(up3), size=s, mode='bilinear')
        pred4 = F.interpolate(self.p4(up4), size=s, mode='bilinear')

        return pred1,pred2,pred3,pred4

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = min(8, in_planes // ratio)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.res = nn.Conv2d(dim,dim,kernel_size=1,stride=1,bias=False)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.sa_conv = nn.Conv2d(1,1,kernel_size=1,stride=1,bias=False)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.res(x)
        ca = self.ca(x) * x
        sa = self.sa(ca)
        sa = self.sa_conv(sa)
        x = ca.mul(sa)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    model = PSNet()
    a = torch.randn(1, 3, 384, 384)
    b = torch.randn(1, 3, 384, 384)
    flops, params = profile(model, (a,b))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

