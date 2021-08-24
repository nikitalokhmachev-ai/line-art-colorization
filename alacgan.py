#!/usr/bin/env python
# coding: utf-8
# %%

# %%

import re
import os
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as M

from torchsummary import summary

from torch.utils.data import DataLoader, Dataset

from sklearn.utils import shuffle

import numpy as np

from PIL import Image
import IPython.display as ipd

import requests

from xdog import to_sketch
from data_utils import *

import matplotlib.pyplot as plt

CUDA_VISIBLE_DEVICES = 5
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


# %%


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


# %%


class Generator(nn.Module):
    def __init__(self, ngf=64, feat=True):
        super(Generator, self).__init__()
        self.feat = feat
        if feat:
            add_channels = 512
        else:
            add_channels = 0
        #WHY CONV2D and not CONVTRANSPOSE2D
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


# %%


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


# %%


class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()
        vgg16 = M.vgg16(pretrained=True)
        vgg16.features = nn.Sequential(
            *list(vgg16.features.children())[:9]
        )
        self.model = vgg16.features
        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images.mul(0.5) - self.mean) / self.std)


# %%


def gradient_penalty(critic, real, fake, gp_weight=10, device='cpu'):
    bs, c, h, w = real.shape
    epsilon = torch.rand((bs, 1, 1, 1)).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad = True
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    #gradient = gradient.view(gradient.shape[0], -1)
    gradient_penalty = torch.mean((gradient.norm(2, dim=1) - 1)**2) * gp_weight
    
    return gradient_penalty


# %%


def predict_img(gen, sk, hnt = None):
    #sk = Image.open(sketch_path).convert('L')
    sk = etrans(sk)

    pad_w = 16 - sk.shape[1] % 16 if sk.shape[1] % 16 != 0 else 0
    pad_h = 16 - sk.shape[2] % 16 if sk.shape[2] % 16 != 0 else 0
    pad = nn.ZeroPad2d((pad_h, 0, pad_w, 0))
    sk = pad(sk)

    sk = sk.unsqueeze(0)
    sk = sk.to(device)

    if hnt is None:
        hnt = torch.zeros((1, 4, sk.shape[2]//4, sk.shape[3]//4))

    hnt = hnt.to(device)

    img_gen = gen(sk, hnt, sketch_feat=None).squeeze(0)
    img_gen = denormalize(img_gen) * 255
    img_gen = img_gen.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(img_gen[pad_w:, pad_h:])


# %%


def predict_link(gen, link):
    img_orig = Image.open(requests.get(link, stream=True).raw).convert('RGB')
    sketch_test = to_sketch(img_orig, sigma=0.5, k=5, gamma=0.92, epsilon=-1, phi=10e15, area_min=2)
    return predict_img(gen, sketch_test, hnt = None)


# %%
def get_hint(im_mask):
    pad_w = 16 - im_mask.size[1] % 16 if im_mask.size[1] % 16 != 0 else 0
    pad_h = 16 - im_mask.size[0] % 16 if im_mask.size[0] % 16 != 0 else 0

    im_mask = Image.fromarray(np.pad(np.array(im_mask), [(pad_w, 0), (pad_h, 0), (0, 0)], mode='constant'))
    im_mask = im_mask.resize((im_mask.width//4, im_mask.height//4))

    hint = np.array(im_mask) / 255.
    hint = np.moveaxis(hint, -1, 0)
    mask = np.expand_dims(np.array(im_mask.convert('L')) != 0, 0)
    hint = np.concatenate([hint,mask], 0)
    
    return torch.FloatTensor(hint).unsqueeze(0)

# %%


etrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
])


# %%


TOTAL_ITER = 250000
IMG_SIZE = 512
BS = 4
LR = 1e-4

WORKERS = 4

DITERS = 1

ADVERSARIAL_WEIGHT = 1e-4
GP_WEIGHT = 10
DRIFT = 1e-3

NGF, NDF = 64, 64

IMG_PATH = 'alacgan_data'

OUT_FOLDER = 'alacgan_mdl'
OUT_IMG_FOLDER = 'alacgan_res'

CONTINUE_TRAINING = True


# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


gen = Generator(feat=False).to(device)
critic = Discriminator(feat=False).to(device)
globf = GlobalFeatureExtractor().to(device)

if CONTINUE_TRAINING:
    last_batch = 373000#max([int(re.sub('\D', '', el)) for el in os.listdir('alacgan_mdl') if 'gen' in el])
    gen.load_state_dict(torch.load(os.path.join(OUT_FOLDER, 'gen_' + str(last_batch) + '.pth')))
    critic.load_state_dict(torch.load(os.path.join(OUT_FOLDER, 'critic_' + str(last_batch) + '.pth')))
    print('Continuing from {} batch...'.format(last_batch))

for param in globf.parameters():
    param.requires_grad = False


# %%
## NEED MORE TESTS
im = Image.open('testim.png')
im_mask = Image.open('testim_mask.png').convert('RGB')

img_to_disp = predict_img(gen, im.convert('L'), hnt = get_hint(im_mask)) 

# %%
####################
links = ['https://cdn.vox-cdn.com/thumbor/J2XSqgAqREtpkGAWa6rMhkHA1Y0=/0x0:1600x900/1400x933/filters:focal(672x322:928x578):no_upscale()/cdn.vox-cdn.com/uploads/chorus_image/image/66320060/Tanjiro__Demon_Slayer_.0.png',
    'http://static.demilked.com/wp-content/uploads/2014/03/detailed-black-pen-drawings-kerby-rosanes-thumb640.jpg',
    'https://i.pinimg.com/474x/8c/84/72/8c847264ac638f6b047ae62eddd0d7ab--dragon-sketch-dragon-drawings.jpg',
        'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpfke4P4Tx3CR9bfTGy1nZF0WH37L8olTigw&usqp=CAU',
        'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCw8f0v-AJkLKGSozjt-ZER6fIEew7nk_MFw&usqp=CAU',
         'https://flowers.tn/wp-content/uploads/2018/08/Flowers-Drawings-Inspiration-A-detailed-flower-line-art-feel-free-to-download-and-colour.jpg',
         'https://media.istockphoto.com/vectors/china-detailed-skyline-vector-background-line-illustration-line-art-vector-id515527320',
        'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSStzDp24RHFChp_Sd02xAK1zram70szPBW3w&usqp=CAU',
        'https://mymodernmet.com/wp/wp-content/uploads/2018/03/coloring-book-pages.jpg']

for link in links:
    img_link = Image.open(requests.get(link, stream=True).raw).convert('RGB')
    sketch_tst = to_sketch(img_link, sigma=0.3, k=5.5, gamma=0.98, epsilon=-1, phi=10e15, area_min=2).convert('L')

    img_to_disp = predict_img(gen, sketch_tst, hnt = None)
    ipd.display(img_to_disp)

# %%


criterion_MSE = nn.MSELoss().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.5, 0.9))


# %%


lr_scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gen, TOTAL_ITER, eta_min=1e-7)
lr_scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(opt_critic, TOTAL_ITER, eta_min=1e-7)


# %%


loader = get_dataloader('alacgan_data', img_size=IMG_SIZE, seed=5, total_iter=TOTAL_ITER, bs=BS, diters=DITERS, last_iter=-1)


# %%
if not CONTINUE_TRAINING:
    last_batch = 0

link = 'https://cdn.vox-cdn.com/thumbor/J2XSqgAqREtpkGAWa6rMhkHA1Y0=/0x0:1600x900/1400x933/filters:focal(672x322:928x578):no_upscale()/cdn.vox-cdn.com/uploads/chorus_image/image/66320060/Tanjiro__Demon_Slayer_.0.png'
img_link = Image.open(requests.get(link, stream=True).raw).convert('RGB')
sketch_tst = to_sketch(img_link, sigma=0.4, k=4.5, gamma=0.93, epsilon=-1, phi=10e15, area_min=2).convert('L')

if OUT_FOLDER not in os.listdir():
    os.mkdir(OUT_FOLDER)

for batch_idx, data in enumerate(loader, start=last_batch):
    color, color_down, sketch = data
    mask = mask_gen(IMG_SIZE, BS)
    hint = torch.cat((color_down * mask, mask), 1)
    color, hint, sketch = color.to(device), hint.to(device), sketch.to(device)
    
    lr_scheduler_gen.step(batch_idx)
    lr_scheduler_critic.step(batch_idx)
    
    current_lr = lr_scheduler_gen.get_lr()[0]
    
    if color.shape[0] == BS:
        for p in critic.parameters():
            p.requires_grad = True
        for p in gen.parameters():
            p.requires_grad = False  
            
        for _ in range(DITERS):
            critic.zero_grad()
                       
            with torch.no_grad():
                #sketch_feat = netI(sketch).detach()
                fake_color = gen(sketch, hint, sketch_feat=None).detach()
                
            critic_conf_fake = critic(fake_color)
            critic_conf_fake = critic_conf_fake.mean(0).view(1)
            critic_conf_fake.backward(retain_graph=True)
            
            critic_conf_real = critic(color)
            critic_conf_real = critic_conf_real.mean(0).view(1)
            
            critic_conf_err = critic_conf_real - critic_conf_fake
            
            loss_critic_real = critic_conf_real.pow(2) * DRIFT - critic_conf_real
            loss_critic_real.backward(retain_graph=True)
            
            gp = gradient_penalty(critic, color, fake_color, device=device)
            gp.backward()
            
            opt_critic.step()

        
        for p in critic.parameters():
            p.requires_grad = False
        for p in gen.parameters():
            p.requires_grad = True
            
        gen.zero_grad()
        
        fake_color = gen(sketch, hint, sketch_feat=None)
        
        critic_conf_fake = critic(fake_color)
        adv_loss = - critic_conf_fake.mean() * ADVERSARIAL_WEIGHT
        adv_loss.backward(retain_graph=True)
        
        feat_fake = globf(fake_color)
        with torch.no_grad():
            feat_real = globf(color)
            
        content_loss = criterion_MSE(feat_fake, feat_real)
        content_loss.backward()
        
        opt_gen.step()
        print(f'adv_loss:{adv_loss}, content_loss:{content_loss}, critic_loss:{critic_conf_err.cpu().detach().numpy()[0]}, penalty_loss:{loss_critic_real.cpu().detach().numpy()[0]}')
        '''with torch.no_grad():
            img_to_disp = predict_img(gen, sketch_tst, hnt = None)
            ipd.display(img_to_disp.resize((img_to_disp.width//4, img_to_disp.height//4)))'''
            
            
        if batch_idx % 1000 == 0:
            with torch.no_grad():
                img_to_disp = predict_img(gen, sketch_tst, hnt = None)
                img_to_disp.save(os.path.join(OUT_IMG_FOLDER, f'img_{batch_idx}.jpg'), 'JPEG')
            torch.save(gen.state_dict(), os.path.join(OUT_FOLDER, f'gen_{batch_idx}.pth'))
            torch.save(critic.state_dict(), os.path.join(OUT_FOLDER, f'critic_{batch_idx}.pth'))
            
        ipd.clear_output(wait=True)


# %%


'''links = ['https://cdn.vox-cdn.com/thumbor/HyOhm280EOQO2ubcOZCSONkDGb8=/0x0:1200x675/1200x800/filters:focal(504x242:696x434)/cdn.vox-cdn.com/uploads/chorus_image/image/68567666/Dr._STONE_Season_2_release_date_Episode_24_ending_with_Stone_Wars_Dr._STONE_manga_compared_to_the_anime_Spoilers.0.jpg',
        'https://jw-webmagazine.com/wp-content/uploads/2020/03/Kimetsu-no-YaibaDemon-Slayer.jpg',
        'https://dthezntil550i.cloudfront.net/00resources/images/page/banner/2f/2fb10643-cd06-461e-8538-2ed6823833ec.jpg',
        'https://live-production.wcms.abc-cdn.net.au/b481c9acb8e5e283f276dfcd7889b593?impolicy=wcms_crop_resize&cropH=576&cropW=863&xPos=80&yPos=0&width=862&height=575',
        'https://cdn.mos.cms.futurecdn.net/eVyt9jnUrLBSvSwW6pScj9-320-80.jpg',
        'https://i.pinimg.com/564x/9a/ba/60/9aba6040f5c0af8c93b388f5df24c121.jpg',
        'https://assets.puzzlefactory.pl/puzzle/258/817/original.jpg']
get_data(links, img_path='alacgan_data', line_widths=[0.3, 0.5])'''

