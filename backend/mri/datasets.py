
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

from backend.mri.config import *

class MriSegmentationDataset(Dataset):

    def __init__(self, patient_list, data_dir):
        self.data_dir = data_dir
        self.data_files = patient_list

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        volumes = nib.load(os.path.join(self.data_dir, self.data_files[index])).get_fdata()
        volume = self.aug_sample(volumes)
        
        # ============== mri classification dataset ==========
        # input = torch.rot90(input,2,(2,3))
        crop_size = DATASET_INPUT_SHAPE
        height, width, depth = volume.shape[1:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = volume[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return (torch.tensor(crop_volume.copy(), dtype=torch.float), torch.tensor(volume.copy(), dtype=torch.float))


    def aug_sample(self, volumes):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]
        """
        volumes = np.moveaxis(volumes, 3, 0)    # [N, H, W, D]
        x = []

        for volume in volumes:
          vol = self.normalize(volume)
          x.append(vol)
        x = np.stack(x, axis=0)
        return x

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())


class MRIClassificationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_files = os.listdir(root_dir)

    def __len__(self):
        return (len(self.data_files))

    def __getitem__(self, idx):
      image_stack = []
      fileName = os.path.join(self.root_dir, self.data_files[idx])
      img = crop_img(fileName, copy=False, pad=False).get_fdata()
      for i in [3,1,0,2]:
        modality = img[:,:,:,i]
        x,y,z = modality.shape
        modality = zoom(modality, (120.0/x, 148.0/y, 120.0/z))
        modality = (modality-np.mean(modality))/np.std(modality)
        modality = np.rot90(modality,2,(0,1))
        image_stack.append(modality)
      image = np.stack(image_stack, axis=0)
      image = torch.from_numpy(image).float()

      return (image, self.data_files[idx])