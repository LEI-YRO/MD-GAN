"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random
from munch import Munch
from PIL import Image
import numpy as np
import argparse
import torch
import tensorflow as tf
from keras import backend as K
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder




def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    fnames=sorted(fnames)
    print(fnames)
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)#取出所有图片
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列（下标，数据）
        for idx, domain in enumerate(sorted(domains)):  # [(0,'Flair'),(1,'T1'),(2,'T1ce'),(3,'T2')]
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames  # ['10001.png','10002.png'..]
            # 随机截取列表指定长度（相当于打乱图片重新保存）
            fnames2 += random.sample(cls_fnames, len(cls_fnames))  # ['20003.png','40002.png'..]
            # [0][1][2]..
            labels += [idx] * len(cls_fnames)  # [[0][0]..[0][1][1]..[1][2][2]..[2][3][3]..[3]](每个标签有19238个相当于一个名单)
        return list(zip(fnames, fnames2)), labels  # [('10001.png','20003.png'),('10002.png','40002.png'),....]

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label,str(fname)


    def __len__(self):
        return len(self.targets)

class SourceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        global proot
        proot=root
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        self.domains = os.listdir(root)
        fnames, labels = [], []
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列（下标，数据）
        for idx, domain in enumerate(sorted(self.domains)):#[(0,'Flair'),(1,'T1'),(2,'T1ce'),(3,'T2')]
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames#['10001.png','10002.png'..]
            # 随机截取列表指定长度（相当于打乱图片重新保存）
            labels += [idx] * len(cls_fnames)#[[0][0]..[0][1][1]..[1][2][2]..[2][3][3]..[3]](每个标签有19238个相当于一个名单)
        return fnames, labels#[('10001.png','20003.png'),('10002.png','40002.png'),....]

    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(fname)

    def getroot(self,root):
        return root


    def __len__(self):
        return len(self.targets)
    
# class TestDataset(data.Dataset):
#     def __init__(self, root, transform=None):
#         global proot
#         proot=root
#         self.samples, self.targets = self._make_dataset(root)
#         self.transform = transform

#     def _make_dataset(self, root):
#         self.domains = os.listdir(root)
#         fnames, labels = [], []
#         # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列（下标，数据）
#         for idx, domain in enumerate(sorted(self.domains)):#[(0,'Flair'),(1,'T1'),(2,'T1ce'),(3,'T2')]
#             class_dir = os.path.join(root, domain)
#             cls_fnames = listdir(class_dir)
#             fnames += cls_fnames#['10001.png','10002.png'..]
#             # 随机截取列表指定长度（相当于打乱图片重新保存）
#             labels += [idx] * len(cls_fnames)#[[0][0]..[0][1][1]..[1][2][2]..[2][3][3]..[3]](每个标签有19238个相当于一个名单)
#         return fnames, labels#[('10001.png','20003.png'),('10002.png','40002.png'),....]

#     def __getitem__(self, index):
#         fname = self.samples[index]
#         label = self.targets[index]
#         img = Image.open(fname).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label, str(fname)

#     def getroot(self,root):
#         return root


#     def __len__(self):
#         return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)#统计数值出现次数[19238,19238,19238,19238],（0.1.2.3每个出现19238次）
    class_weights = 1. / class_counts#[1./19238,1./19238,1./19238,1./19238]
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_ref_img(img_size=256, src_path=None, x_label=None, y_label=None):
    x_domain = ''
    y_domain = ''
    i=0
    img = list()
    root = proot
    src_path = src_path
    domains = os.listdir(root)
    transform = transforms.Compose([
        # 重建分辨率为256×256
        transforms.Resize([img_size, img_size]),
        # # 依据概率p（默认0.5）对图像进行水平翻转
        # transforms.RandomHorizontalFlip(),
        # 将图像转为tensor并归一化到0-1（直接除以255），输出是chw
        transforms.ToTensor(),
        # 逐通道进行标准化为正态分布【-1,1】：output = (input - mean) / std
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    for path in src_path:
        if i<len(x_label):
            for x, domain in enumerate(sorted(domains)):
                if x == x_label[i]:
                    x_domain = str(domain)
            for y, domian in enumerate(sorted(domains)):
                if y == y_label[i]:
                    y_domain = str(domian)
            i = i + 1
        y_path = path.replace(x_domain, y_domain)
        # print(path)
        # print(y_path)
        y_img = Image.open(y_path).convert('RGB')
        y_img = transform(y_img)
        img.append(y_img)
    yimg = torch.stack(img, 0)
    return  yimg



def get_train_loader(root, which='source', img_size=256,
                     batch_size=32, prob=0.5, num_workers=8):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    # 将图片随机裁剪为不同大小和宽高比后输出为img_size大小
    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.0])
    # 随机决定是否裁剪缩放图片，如果随机数小于0.5则裁剪，否则不裁剪
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)
    # 对图片的各种转换操作用compose进行组合
    transform = transforms.Compose([
        # rand_crop,
        # 重建分辨率为256×256
        transforms.Resize([img_size, img_size]),
        # # 依据概率p（默认0.5）对图像进行水平翻转
        # transforms.RandomHorizontalFlip(),
        # 将图像转为tensor并归一化到0-1（直接除以255），输出是chw
        transforms.ToTensor(),
        # 逐通道进行标准化为正态分布【-1,1】：output = (input - mean) / std
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        # ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名:0开始的序号
        # 按root寻找图片，对图片进行transform操作
        dataset = SourceDataset(root, transform)#['10001.png',0]
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)#[('10001.png','20004.png')][0]
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           # shuffle=shuffle,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)



def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=False, num_workers=8):
    print('Preparing DataLoader for the generation test phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = SourceDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y, path= next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, path = next(self.iter)
        return x, y, path

    def _fetch_refs(self):
        try:
            x, x2, y, path= next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y , path= next(self.iter_ref)
        return x, x2, y, path

    def __next__(self):
        x, y ,path= self._fetch_inputs()
        # y_src = get_ref_img()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref, ypath = self._fetch_refs()
            y_img = get_ref_img(src_path=path, x_label=y.numpy(), y_label=y_ref.numpy())
            x_img = get_ref_img(src_path=ypath, x_label=y_ref.numpy(), y_label=y.numpy())
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,y_img = y_img,x_img = x_img,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref,fpath = self._fetch_inputs()
            y_img = get_ref_img(src_path=path, x_label=y.numpy(), y_label=y_ref.numpy())
            inputs = Munch(x_src=x, y_src=y,y_img = y_img,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})