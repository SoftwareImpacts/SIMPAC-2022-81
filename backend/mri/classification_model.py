import math
import os
import shutil
import time
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from nilearn.image import crop_img
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value


from medcam import medcam
import nibabel as nib

import logging
import numpy as np
from collections import defaultdict,deque

import sys
from torch.utils.data import Dataset, DataLoader
import random
import warnings
from torch.optim import lr_scheduler
from tqdm import tqdm_notebook

from openpyxl import load_workbook
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

import torchvision.transforms as transforms
from nilearn.image import crop_img
from scipy.ndimage import zoom

from backend.mri.datasets import *
from backend.mri.config import *

class ClassificationBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(ClassificationBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool3d(out, 2)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_planes)
        self.conv2 = nn.Conv3d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DenseNet3(nn.Module):
    def __init__(self, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = [6, 12, 32, 32]
        # n = [6, 12, 24, 16]
        if bottleneck == True:
            block = BottleneckBlock
        else:
            block = ClassificationBasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv3d(4, in_planes, kernel_size=7, stride=2,
                               padding=3, bias=True)
        # 1st block
        self.block1 = DenseBlock(n[0], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n[0]*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))

        # 2nd block
        self.block2 = DenseBlock(n[1], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n[1]*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))

        # 3rd block
        self.block3 = DenseBlock(n[2], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n[2]*growth_rate)
        self.trans3 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))

        # 4th block
        self.block4 = DenseBlock(n[3], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n[3]*growth_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(1920, num_classes)
        self.fc = nn.Linear(2688, num_classes)
        self.in_planes = in_planes

        self.maxPool = nn.MaxPool3d(kernel_size= 3, padding=1, stride=2)
        self.avgPool = nn.AvgPool3d(kernel_size = (3,4,3))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] *m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.modelSequential = nn.Sequential(
            self.conv1,
            self.maxPool,
            self.block1,
            self.trans1,
            self.block2,
            self.trans2,
            self.block3,
            self.trans3,
            self.block4,
            self.bn1,
            self.relu,
            self.avgPool,
            nn.Flatten(start_dim=1, end_dim=-1),
            self.fc
        )
  
    def forward(self, x):
        return self.modelSequential(x)