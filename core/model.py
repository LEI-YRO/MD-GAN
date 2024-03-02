import os

from core.spectral_normalization import SpectralNorm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.wing import FAN

class DenseBlock(nn.Module):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, dim_in,drop_rate):
        super(DenseBlock, self).__init__()
        self.dim_in  = dim_in
        self.drop_rate = drop_rate
        self.norm =nn.InstanceNorm2d(dim_in,affine=True)
        self.norm2 =nn.InstanceNorm2d(int(dim_in+dim_in/2),affine=True)
        self.relu =nn.LeakyReLU(0.2)
        self.conv1 =nn.Conv2d(dim_in, dim_in,kernel_size=1, stride=1, bias=False)
        self.conv2 =nn.Conv2d(dim_in, int(dim_in/2),kernel_size=3, stride=1, padding=1,bias=False)
        self.conv3 = nn.Conv2d(int(dim_in+dim_in/2), int(dim_in+dim_in/2), kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(int(dim_in+dim_in/2),int(dim_in/2), kernel_size=3, stride=1, padding=1, bias=False)
    def _denselayer1(self,x):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out
    def _denselayer2(self,x):
        out = self.norm2(x)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv4(out)
        return out
    def forward(self, x):
        # print(x.size())
        new_features = self._denselayer1(x)
        # print(new_features.size())
        new_features  = torch.cat([x, new_features], 1)
        # print(new_features.size())
        new_features2 = self._denselayer2(new_features)
        new_features2 = torch.cat([new_features, new_features2], 1)
        # print(new_features2.size())
        if self.drop_rate > 0:
            new_features2 = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return  new_features2

class Transition(nn.Module):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, downsample=False,upsample=False):
        super(Transition, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
    def forward(self, x):
        out = x
        if self.downsample:
            self.pool = nn.AvgPool2d(2, stride=2)
            out = self.pool(x)
        if self.upsample:
            out = F.interpolate(x, scale_factor=2, mode='nearest')
        return out




# special ResBlock just for the layer of the discriminator
class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, feature=False):
        super().__init__()
        #
        self.feature = feature
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            # self.norm1 = SwitchNormalization(3)
            # self.norm2 = SwitchNormalization(3)
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        if self.feature:
            return x
        return x / math.sqrt(2)  # unit variance
    
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s=None):
        if s is not None:
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
        else:
            return self.norm(x)


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=16, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=16):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        # self.norm3 = nn.InstanceNorm2d(dim_in, affine=True)
        # self.norm4 = nn.InstanceNorm2d(dim_out, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


    
class Anencoder(nn.Module):
    def __init__(self, img_size=256, style_dim=16, max_conv_dim=256, w_hpf=0):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            # SwitchNormalization(dim_in),
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))


        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            if dim_in != max_conv_dim:
                self.encode.append(DenseBlock(dim_in, 0))
                self.encode.append(Transition(downsample=True))
            dim_in = dim_out
        # bottleneck blocks
        #self.encode.append(Transition(downsample=True))
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))

    def forward(self, x, masks=None):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
            #print(x.shape)
        return x
    
class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=16, max_conv_dim=256, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        # self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        # self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))


        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 6

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            # if dim_in != max_conv_dim:
            #     self.encode.append(DenseBlock(dim_in, 0))
            #     self.encode.append(Transition(downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))
            dim_in = dim_out
        # bottleneck blocks
        # self.encode.append(Transition(downsample=True))
        for _ in range(4):
            # self.encode.append(
            #     ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))


    def forward(self, x, s, masks=None):
        for block in self.decode:
            x = block(x,s)
            #print("decoder:",x.shape)
        return self.to_rgb(x)
    

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=16, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        #
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]


        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        
        self.linear = nn.Linear(dim_out,style_dim)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, 64)]
    def forward(self, x, y=None):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        if y==None:
            m = self.linear(h)
            return m
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)#
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlockDiscriminator(dim_in, dim_out, stride=2)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

class Classifier(nn.Module):
    def __init__(self, input_dim=3, df_dim=64, num_classes=3):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, df_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(df_dim, df_dim*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(df_dim*2, df_dim*4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(df_dim*4, num_classes)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.conv2(h1))
        h3 = F.leaky_relu(self.conv3(h2))
        h4 = self.pool(h3).squeeze()
        h5 = self.fc(h4)
        return h5

def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    # mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    anencoder = nn.DataParallel(Anencoder(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    classifier = nn.DataParallel(Classifier(args.input_dim, args.df_dim))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    anencoder_ema = copy.deepcopy(anencoder)
    classifier_ema = copy.deepcopy(classifier)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
    

    nets = Munch(generator=generator,
                 # mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 anencoder=anencoder,
                 discriminator=discriminator,
                 classifier=classifier)
    nets_ema = Munch(generator=generator_ema,
                     # mapping_network=mapping_network_ema,
                     anencoder=anencoder_ema,
                     style_encoder=style_encoder_ema,
                     classifier=classifier_ema
                    )

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema

