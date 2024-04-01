import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity

### Your model

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class MDBN(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, upscale=2, res_scale=1.0):
        super(MDBN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlock, num_block, num_feat=num_feat, res_scale=res_scale)

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = Upsample(upscale, num_feat)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))

        return x

class ResidualBlock(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.baseblock1 = BaseBlock(num_feat)
        self.baseblock2 = BaseBlock(num_feat)

    def forward(self, x):
        identity = x

        x = self.baseblock1(x)
        x = self.baseblock2(x)

        return identity + x * self.res_scale

class BaseBlock(nn.Module):
    def __init__(self, num_feat):
        super(BaseBlock, self).__init__()
        self.uconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.uconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.uconv2(self.act(self.uconv1(x)))
        x2 = self.dconv(x)
        x = self.act(x1 + x2)
        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


### Settings

dev = torch.device('cuda:0')
path = "experiments/pretrained_models/MDBN_x2.pth"

model = MDBN().to(dev)
state_dict = torch.load(path)
model.load_state_dict(state_dict['params'])


### random input

x = torch.randn((1, 3, 320, 180),device=dev)

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        for _ in range(50):
            model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))