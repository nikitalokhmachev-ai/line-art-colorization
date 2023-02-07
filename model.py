import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate, groups=cardinality, bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        
        if stride != 1:
            self.shortcut.add_module('shortcut', nn.AvgPool2d(2, stride=2))
            
    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        
        return x + bottleneck
    
    
class Generator(nn.Module):
    def __init__(self, ngf=64, feat=True):
        super(Generator, self).__init__()
        self.feat = feat
        if feat:
            add_channels = 512
        else:
            add_channels = 0
        
        self.toH = self._block(4, ngf, kernel_size=7, stride=1, padding=3)
        self.to0 = self._block(1, ngf // 2, kernel_size=3, stride=1, padding=1)
        self.to1 = self._block(ngf // 2, ngf, kernel_size=4, stride=2, padding=1)
        self.to2 = self._block(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.to3 = self._block(ngf * 3, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.to4 = self._block(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
        
        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1) for _ in range(20)])
        
        self.tunnel4 = nn.Sequential(self._block(ngf * 8 + add_channels, ngf * 8, kernel_size=3, stride=1, padding=1),
                                     tunnel4,
                                     nn.Conv2d(ngf * 8, ngf * 16, kernel_size = 3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True))
        
        depth = 2
        
        tunnel = [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2),
                   ResNeXtBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)
        
        self.tunnel3 = nn.Sequential(self._block(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     tunnel3,
                                     nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2), 
                                     nn.LeakyReLU(0.2, True))
        
        tunnel = [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2),
                   ResNeXtBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)
        
        self.tunnel2 = nn.Sequential(self._block(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     tunnel2,
                                     nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2), 
                                     nn.LeakyReLU(0.2, True))
        
        tunnel = [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=2),
                   ResNeXtBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)
        
        self.tunnel1 = nn.Sequential(self._block(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
                                     tunnel1,
                                     nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2), 
                                     nn.LeakyReLU(0.2, True))
        
        self.exit = nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, True)
        )
    
    
    def forward(self, sketch, hint, sketch_feat):
        hint = self.toH(hint)
        
        x0 = self.to0(sketch)
        x1 = self.to1(x0)
        x2 = self.to2(x1)
        x3 = self.to3(torch.cat([x2, hint], 1))
        x4 = self.to4(x3)
        
        if self.feat:
            x = self.tunnel4(torch.cat([x4, sketch_feat], 1))
            x = self.tunnel3(torch.cat([x, x3], 1))
            x = self.tunnel2(torch.cat([x, x2], 1))
            x = self.tunnel1(torch.cat([x, x1], 1))
            x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        else:
            x = self.tunnel4(x4)
            x = self.tunnel3(torch.cat([x, x3], 1))
            x = self.tunnel2(torch.cat([x, x2], 1))
            x = self.tunnel1(torch.cat([x, x1], 1))
            x = torch.tanh(self.exit(torch.cat([x, x0], 1)))
        return x
    

class Discriminator(nn.Module):
    def __init__(self, ndf=64, feat=True):
        super(Discriminator, self).__init__()
        self.feat = feat
        
        if feat:
            add_channels = ndf * 8
            ks = 4
        else:
            add_channels = 0
            ks = 3
            
        self.feed = nn.Sequential(
            self._block(3, ndf, kernel_size=7, stride=1, padding=1),
            self._block(ndf, ndf, kernel_size=4, stride=2, padding=1),
            
            ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),
            self._block(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            
            ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),
            self._block(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0),
            
            ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2)
        )
        
        self.feed2 = nn.Sequential(
            self._block(ndf * 4 + add_channels, ndf * 8, kernel_size=3, stride=1, padding=1),
            
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            
            self._block(ndf * 8, ndf * 8, kernel_size=ks, stride=1, padding=0),
        )
        
        self.out = nn.Linear(512, 1)
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, color, sketch_feat=None):
        x = self.feed(color)
        
        if self.feat:
            x = self.feed2(torch.cat([x, sketch_feat], 1))
        else:
            x = self.feed2(x)
        
        out = self.out(x.view(color.size(0), -1))
        return out