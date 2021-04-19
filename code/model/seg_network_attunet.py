import torch
from torch import nn as nn
from torch.nn import functional as F
from lib.utils import conv, relu, interpolate, adaptive_cat


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()
        nc = ic + ic + fc
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(), conv(nc, nc, 3), relu(), conv(nc, oc, 3), relu())

    def forward(self, ft, score1, score2):
        h = adaptive_cat((ft, score1, score2), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h


class ATB(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ATB, self).__init__()
        self.W_g = nn.Sequential(conv(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(conv(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(conv(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi



class RRB(nn.Module):

    def __init__(self, oc, use_bn=False):
        super().__init__()
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block
        self.conv1x1 = conv(2 * oc, oc, 1)
    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class UPC(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UPC, self).__init__()
        self.up = nn.Sequential(
            conv(ch_in, ch_out, 3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, image_size):
        x = F.interpolate(x, image_size[-2:], mode='bilinear', align_corners=False)
        x = self.up(x)
        return x


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
        self.ATB = nn.ModuleDict()
        self.RRB = nn.ModuleDict()
        self.UPC = nn.ModuleDict()

        ic = in_channels
        oc = out_channels

        for L, fc in self.ft_channels.items():
            self.TSE[L] = TSE(fc, ic, oc)
            self.UPC[L] = UPC(oc, oc)
            self.ATB[L] = ATB(oc, oc, oc)
            self.RRB[L] = RRB(oc, use_bn=use_bn)

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

        ft5 = features["layer5"]
        ft4 = features["layer4"]
        ft3 = features["layer3"]
        ft2 = features["layer2"]

        if multi_targets:
            h5 = self.TSE["layer5"](ft5.repeat(num_targets, 1, 1, 1)
                                    , interpolate(scores_l3, ft5.shape[-2:])
                                    , interpolate(scores_l4, ft5.shape[-2:]))
            h4 = self.TSE["layer4"](ft4.repeat(num_targets, 1, 1, 1)
                                    , interpolate(scores_l3, ft4.shape[-2:])
                                    , interpolate(scores_l4, ft4.shape[-2:]))
            h3 = self.TSE["layer3"](ft3.repeat(num_targets, 1, 1, 1)
                                    , interpolate(scores_l3, ft3.shape[-2:])
                                    , interpolate(scores_l4, ft3.shape[-2:]))
            h2 = self.TSE["layer2"](ft2.repeat(num_targets, 1, 1, 1)
                                    , interpolate(scores_l3, ft2.shape[-2:])
                                    , interpolate(scores_l4, ft2.shape[-2:]))
        else:
            h5 = self.TSE["layer5"](ft5
                                    , interpolate(scores_l3, ft5.shape[-2:])
                                    , interpolate(scores_l4, ft5.shape[-2:]))
            h4 = self.TSE["layer4"](ft4
                                    , interpolate(scores_l3, ft4.shape[-2:])
                                    , interpolate(scores_l4, ft4.shape[-2:]))
            h3 = self.TSE["layer3"](ft3
                                    , interpolate(scores_l3, ft3.shape[-2:])
                                    , interpolate(scores_l4, ft3.shape[-2:]))
            h2 = self.TSE["layer2"](ft2
                                    , interpolate(scores_l3, ft2.shape[-2:])
                                    , interpolate(scores_l4, ft2.shape[-2:]))

        d5 = self.UPC["layer5"](h5, h4.size())
        #print("UPC:",d5.size())
        h4 = self.ATB["layer5"](g=d5, x=h4)
        #print("ATB:", h4.size())
        d5 = torch.cat((h4, d5), dim=1)
        d5 = self.RRB["layer5"](d5)

        d4 = self.UPC["layer4"](d5, h3.size())
        #print("UPC:", d4.size())
        h3 = self.ATB["layer4"](g=d4, x=h3)
        #print("ATB:", h3.size())
        d4 = torch.cat((h3, d4), dim=1)
        d4 = self.RRB["layer4"](d4)

        d3 = self.UPC["layer3"](d4, h2.size())
        #print("UPC:", d3.size())
        h2 = self.ATB["layer3"](g=d3, x=h2)
        #print("ATB:", h2.size())
        d3 = torch.cat((h2, d3), dim=1)
        x = self.RRB["layer3"](d3)

        x = self.project(x, image_size)

        return x
