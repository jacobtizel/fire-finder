import torch
import torch.nn as nn
import torch.nn.functional as F

#Somehow ARCGIS has a tutorial on this? Ok i guess
#https://developers.arcgis.com/python/latest/guide/how-unet-works/
#Original Paper: https://arxiv.org/pdf/1505.04597

class UNet(nn.Module):
    def __init__(self, inputChannels=1, outputChannels=1):
        #When a UNet is created, initialize each step
        #
        super(UNet, self).__init__()
        #Each convolution block follows: Convolution -> ReLu -> Convolution -> ReLU
        #It makes sense to do that all at once
        #Convolutions are 3x3
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), #Use padding of 1
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        #Create Several useful encoders
        #From 1 to 64 downsample
        self.downsample1to64 = conv_block(inputChannels, 64)
        #From 64 to 128 downsample
        self.downsample64to128 = conv_block(64, 128)
        #From 128 to 256 downsample
        self.downsample128to256 = conv_block(128, 256)
        #From 256 to 512 downsample
        self.downsample256to512 = conv_block(256, 512)

        self.bottleneck = conv_block(512, 1024)# I think this is called a bottleneck?

        self.upconvLevel4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample1024to512 = conv_block(1024, 512)

        self.upconvLevel3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample512to256 = conv_block(512, 256)

        self.upconvLevel2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample256to128 = conv_block(256, 128)

        self.upconvLevel1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample128to64 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, outputChannels, kernel_size=1)

    def forward(self, x):
        #Contracting Path
        encodeLevel1 = self.downsample1to64(x)
        pool1to2 = F.max_pool2d(encodeLevel1, 2)

        encodeLevel2 = self.downsample64to128(pool1to2)
        pool2to3 = F.max_pool2d(encodeLevel2, 2)

        encodeLevel3 = self.downsample128to256(pool2to3)
        pool3to4 = F.max_pool2d(encodeLevel3, 2)

        encodeLevel4 = self.downsample256to512(pool3to4)
        pool4to5 = F.max_pool2d(encodeLevel4, 2)

        # Bottleneck
        b = self.bottleneck(pool4to5)

        #Expansive Path
        upConv5to4 = self.upconvLevel4(b)
        upConv5to4 = torch.cat([upConv5to4, encodeLevel4], dim=1)
        decodeLevel4 = self.upsample1024to512(upConv5to4)

        upConv4to3 = self.upconvLevel3(decodeLevel4)
        upConv4to3 = torch.cat([upConv4to3, encodeLevel3], dim=1)
        decodeLevel3 = self.upsample512to256(upConv4to3)

        upConv3to2 = self.upconvLevel2(decodeLevel3)
        upConv3to2 = torch.cat([upConv3to2, encodeLevel2], dim=1)
        decodeLevel2 = self.upsample256to128(upConv3to2)

        upConv2to1 = self.upconvLevel1(decodeLevel2)
        upConv2to1 = torch.cat([upConv2to1, encodeLevel1], dim=1)
        decodeLevel1 = self.upsample128to64(upConv2to1)

        return torch.sigmoid(self.final_conv(decodeLevel1))