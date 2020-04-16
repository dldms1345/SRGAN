import torch
import torch.nn as nn
from torchvision.models import vgg19


class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.vgg_model = nn.Sequential(
            *list(vgg19(pretrained=True).features)[:31]).eval()
        for param in self.vgg_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.vgg_model(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_features, num_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.Conv2d(num_features, num_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features))
            
    def forward(self, x):
        return x + self.main(x) # skip-connection(elementwise sum)


class UpsamplingBlock(nn.Module):
    def __init__(self):
        super(UpsamplingBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, # in_channels * up_scale ** 2
                      kernel_size=3, stride=1, padding=1),
			nn.PixelShuffle(upscale_factor=2),
			nn.PReLU())

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, num_res_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,   # inchannel=3(BGR)
                      kernel_size=3, stride=1, padding=1),
            nn.PReLU(num_parameters=1, init=0.25))   # default
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(num_features=64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64))

        upsample_blocks = []
        for _ in range(2):
            upsample_blocks.append(UpsamplingBlock())
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4))

    def forward(self, x):
        out1 = self.conv1(x)
        res_out = self.res_blocks(out1)
        out2 = self.conv2(res_out)
        out = out1 + out2
        out = self.upsample_blocks(out)
        out = self.conv3(out)
        return (torch.tanh(out) + 1) / 2    # 0~1


class Discriminator(nn.Module):
    def __init__(self, a=0.2):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride):
            layer = []
            layer.append(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride, padding=1))
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.LeakyReLU(a))
            return layer

        self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(a))
        
        discrim_blocks = []
        discrim_blocks.extend(discriminator_block(64, 64, 2))
        discrim_blocks.extend(discriminator_block(64, 128, 1))
        discrim_blocks.extend(discriminator_block(128, 128, 2))
        discrim_blocks.extend(discriminator_block(128, 256, 1))
        discrim_blocks.extend(discriminator_block(256, 256, 2))
        discrim_blocks.extend(discriminator_block(256, 512, 1))
        discrim_blocks.extend(discriminator_block(512, 512, 2))
        self.conv2 = nn.Sequential(*discrim_blocks)

        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(a),
            nn.Conv2d(1024, 1, kernel_size=1))
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv2(self.conv1(x))
        out = self.dense(out)
        out = out.view(batch_size)
        return torch.sigmoid(out)