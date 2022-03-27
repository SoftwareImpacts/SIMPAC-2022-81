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


DATASET_USE_MODES = ( "flair", "t1", "t1ce", "t2")
DATASET_INPUT_SHAPE = (160, 192, 128)

DATALOADER_BATCH_SIZE = 1
DATALOADER_NUM_WORKERS = 6

MODEL_NAME = 'unet-vae'
MODEL_INIT_CHANNELS = 16
MODEL_DROPOUT = 0.2
MODEL_LOSS_WEIGHT = 0.1

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')