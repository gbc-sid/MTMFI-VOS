import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import conv, relu, interpolate, adaptive_cat


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            assert channel_att_raw is not None
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()
        nc = ic + ic + fc
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(), conv(nc, nc, 3), relu(), conv(nc, oc, 3), relu())

    def forward(self, ft, score1, score2):
        h = adaptive_cat((ft, score1, score2), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h


class UPC(nn.Module):
    def __init__(self, ch_out):
        super(UPC, self).__init__()
        self.up = nn.Sequential(
            conv(2 * ch_out, ch_out, 3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, shallower, deeper):
        deeper = F.interpolate(deeper, shallower.shape[-2:], mode='bilinear', align_corners=False)
        return self.up(torch.cat([shallower, deeper], dim=1))


class RRB(nn.Module):

    def __init__(self, oc, use_bn=False):
        super().__init__()
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block
        self.conv1x1 = conv(oc, oc, 1)

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class PyrUpBicubic2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        def kernel(d):
            x = d + torch.arange(-1, 3, dtype=torch.float32)
            x = torch.abs(x)
            a = -0.75
            f = (x < 1).float() * ((a + 2) * x * x * x - (a + 3) * x * x + 1) + \
                ((x >= 1) * (x < 2)).float() * (a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a)
            W = f.reshape(1, 1, 1, len(x)).float()
            Wt = W.permute(0, 1, 3, 2)
            return W, Wt

        We, We_t = kernel(-0.25)
        Wo, Wo_t = kernel(-0.25 - 0.5)

        # Building non-separable filters for now. It would make sense to
        # have separable filters if it proves to be faster.

        # .contiguous() is needed until a bug is fixed in nn.Conv2d.
        self.W00 = (We_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W01 = (We_t @ Wo).expand(channels, 1, 4, 4).contiguous()
        self.W10 = (Wo_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W11 = (Wo_t @ Wo).expand(channels, 1, 4, 4).contiguous()

    def forward(self, input):
        if input.device != self.W00.device:
            self.W00 = self.W00.to(input.device)
            self.W01 = self.W01.to(input.device)
            self.W10 = self.W10.to(input.device)
            self.W11 = self.W11.to(input.device)

        a = F.pad(input, (2, 2, 2, 2), 'replicate')

        I00 = F.conv2d(a, self.W00, groups=self.channels)
        I01 = F.conv2d(a, self.W01, groups=self.channels)
        I10 = F.conv2d(a, self.W10, groups=self.channels)
        I11 = F.conv2d(a, self.W11, groups=self.channels)

        n, c, h, w = I11.shape

        J0 = torch.stack((I00, I01), dim=-1).view(n, c, h, 2 * w)
        J1 = torch.stack((I10, I11), dim=-1).view(n, c, h, 2 * w)
        out = torch.stack((J0, J1), dim=-2).view(n, c, 2 * h, 2 * w)

        out = F.pad(out, (-1, -1, -1, -1))
        return out


class BackwardCompatibleUpsampler(nn.Module):
    """ Upsampler with bicubic interpolation that works with Pytorch 1.0.1 """

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.up1 = PyrUpBicubic2d(in_channels)
        self.conv2 = conv(in_channels // 2, 1, 3)
        self.up2 = PyrUpBicubic2d(in_channels // 2)

    def forward(self, x, image_size):
        x = self.up1(x)
        x = F.relu(self.conv1(x))
        x = self.up2(x)
        x = F.interpolate(x, image_size[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x


class SegNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.CBAM = nn.ModuleDict()
        self.UPC = nn.ModuleDict()

        ic = in_channels
        oc = out_channels

        for L, fc in self.ft_channels.items():
            # print(L, fc)
            self.TSE[L] = TSE(fc, ic, oc)
            self.RRB1[L] = RRB(oc, use_bn=use_bn)
            self.RRB2[L] = RRB(oc, use_bn=use_bn)
            self.UPC[L] = UPC(oc)
            self.CBAM[L] = CBAM(oc, reduction_ratio=16, pool_types=['avg'])

        # if torch.__version__ == '1.0.1'
        self.project = BackwardCompatibleUpsampler(out_channels)
        # self.project = Upsampler(out_channels)

    def forward(self, scores_l3, scores_l4, features, image_size):

        num_targets = scores_l4.shape[0]
        num_fmaps = features[next(iter(self.ft_channels))].shape[0]
        if num_targets > num_fmaps:
            multi_targets = True
        else:
            multi_targets = False

        x = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            s_l3 = interpolate(scores_l3, ft.shape[-2:])  # Resample scores to match features size
            s_l4 = interpolate(scores_l4, ft.shape[-2:])

            if multi_targets:
                h = self.TSE[L](ft.repeat(num_targets, 1, 1, 1), s_l3, s_l4)
            else:
                h = self.TSE[L](ft, s_l3, s_l4)

            if L == "layer5":
                h = self.RRB1[L](h)
                h = self.CBAM[L](h)
                x = self.RRB2[L](h)
            else:
                h = self.UPC[L](h, x)
                h = self.RRB1[L](h)
                h = self.CBAM[L](h)
                x = self.RRB2[L](h)

        x = self.project(x, image_size)
        return x
