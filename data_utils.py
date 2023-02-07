# +
import os
import math
import random
import numbers
import requests
import shutil
import numpy as np
import scipy.stats as stats
from PIL import Image
from tqdm.auto import tqdm

from xdog import to_sketch
# -

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop

mu, sigma = 1, 0.005
X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

denormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                  transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                       std = [ 1., 1., 1. ]),])

etrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_img(gen, sk, hnt = None):
    #sk = Image.open(sketch_path).convert('L')
    sk = etrans(sk)

    pad_w = 16 - sk.shape[1] % 16 if sk.shape[1] % 16 != 0 else 0
    pad_h = 16 - sk.shape[2] % 16 if sk.shape[2] % 16 != 0 else 0
    pad = nn.ZeroPad2d((pad_h, 0, pad_w, 0))
    sk = pad(sk)

    sk = sk.unsqueeze(0)
    sk = sk.to(device)

    if hnt == None:
        hnt = torch.zeros((1, 4, sk.shape[2]//4, sk.shape[3]//4))

    hnt = hnt.to(device)

    img_gen = gen(sk, hnt, sketch_feat=None).squeeze(0)
    img_gen = denormalize(img_gen) * 255
    img_gen = img_gen.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    #return img_gen[pad_w:, pad_h:]
    return Image.fromarray(img_gen[pad_w:, pad_h:])

def files(img_path, img_size=512):
    img_path = os.path.abspath(img_path)
    line_widths = sorted([el for el in os.listdir(os.path.join(img_path, 'pics_sketch')) if el != '.ipynb_checkpoints'])
    images_names = sorted([el for el in os.listdir(os.path.join(img_path, 'pics_sketch', line_widths[0])) if '.jpg' in el])
    
    images_names = [el for el in images_names if np.all(np.array(Image.open(os.path.join(img_path, 'pics', el)).size) >= np.array([img_size, img_size]))]
    
    images_color = [os.path.join(img_path, 'pics', el) for el in images_names]
    images_sketch = {line_width:[os.path.join(img_path, 'pics_sketch', line_width, el) for el in images_names] for line_width in line_widths}
    return images_color, images_sketch

def mask_gen(img_size=512, bs=4):
    maskS = img_size // 4

    mask1 = torch.cat([torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(bs // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(bs // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)
    return mask

def jitter(x):
    ran = random.uniform(0.7, 1)
    return x * ran + 1 - ran

def make_trans(img_size):
    vtrans = transforms.Compose([
            RandomSizedCrop(img_size // 4, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ctrans = transforms.Compose([
            transforms.Resize(img_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    strans = transforms.Compose([
            transforms.Resize(img_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(jitter),
            transforms.Normalize((0.5), (0.5))
    ])
    
    return vtrans, ctrans, strans

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1, img2

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        Resize = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(Resize(img))


class ImageFolder(data.Dataset):
    def __init__(self, img_path, img_size):
        
        self.images_color, self.images_sketch = files(img_path, img_size)
        if (any([self.images_sketch[key] == 0 for key in self.images_sketch])) or (len(self.images_color) == 0):
            raise (RuntimeError("Found 0 images in one of the folders."))
        if any([len(self.images_sketch[key]) != len(self.images_color) for key in self.images_sketch]):
            raise (RuntimeError("The number of sketches is not equal to the number of colorized images."))
        self.img_path = img_path
        self.img_size = img_size
        self.vtrans, self.ctrans, self.strans = make_trans(img_size)

    def __getitem__(self, index):
        color = Image.open(self.images_color[index]).convert('RGB')
        
        random_line_width = random.choice(list(self.images_sketch.keys()))
        sketch = Image.open(self.images_sketch[random_line_width][index]).convert('L')
        #the image can be smaller than img_size, fix!
        color, sketch = RandomCrop(self.img_size)(color, sketch)
        if random.random() < 0.5:
            color, sketch = color.transpose(Image.FLIP_LEFT_RIGHT), sketch.transpose(Image.FLIP_LEFT_RIGHT)
            
        color, color_down, sketch = self.ctrans(color), self.vtrans(color), self.strans(sketch)
                
        return color, color_down, sketch

    def __len__(self):
        return len(self.images_color)


class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, diter, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.diter = diter
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size * (self.diter + 1)

        self.indices = self.gen_new_list()
        self.call = 0
        

    def __iter__(self):
        #if self.call == 0:
            #self.call = 1
        return iter(self.indices[(self.last_iter + 1) * self.batch_size * (self.diter + 1):])
        #else:
        #    raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        # each process shuffle all list with same seed
        np.random.seed(0)

        indices = np.arange(len(self.dataset))
        indices = indices[:self.total_size]
        num_repeat = (self.total_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_size]

        np.random.shuffle(indices)
        assert len(indices) == self.total_size
        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

def get_dataloader(img_path, img_size=512, seed=0, total_iter=250000, bs=4, diters=1, last_iter=-1):
    
    random.seed(seed)
    
    train_dataset = ImageFolder(img_path=img_path, img_size=img_size)

    train_sampler = GivenIterationSampler(train_dataset, total_iter, bs, diters, last_iter=last_iter)

    return data.DataLoader(train_dataset, batch_size=bs, shuffle=False, pin_memory=True, num_workers=4, sampler=train_sampler)


def get_data(links, img_path='alacgan_data', line_widths=[0.3, 0.5]):
    c = 0
    
    for line_width in line_widths:
        lw = str(line_width)
        if lw not in os.listdir(os.path.join(img_path, 'pics_sketch')):
            os.mkdir(os.path.join(img_path, 'pics_sketch', lw))
        else:
            shutil.rmtree(os.path.join(img_path, 'pics_sketch', lw))
            os.mkdir(os.path.join(img_path, 'pics_sketch', lw))
            
    for link in tqdm(links):
        img_orig = Image.open(requests.get(link, stream=True).raw).convert('RGB')
        img_orig.save(os.path.join(img_path, 'pics', str(c) + '.jpg'), 'JPEG')
        for line_width in line_widths:
            sketch_test = to_sketch(img_orig, sigma=line_width, k=5, gamma=0.96, epsilon=-1, phi=10e15, area_min=2)
            sketch_test.save(os.path.join(img_path, 'pics_sketch', str(line_width), str(c) + '.jpg'), 'JPEG')
            
        c += 1
