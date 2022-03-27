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
from backend.mri.segmentation_model import *
from backend.mri.classification_model import *
from backend.mri.datasets import *
from backend.mri.mri_visualize import *

#******************** MRI Segmentation ********************

def make_data_loaders_segmenetation(image_val):
        patients_list=sorted(os.listdir(image_val))
        dataset = MriSegmentationDataset(patients_list, image_val)
        batch_size = 1
        loaders = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False)
        return loaders


def build_model_segmentation(file_checkpoint, cuda_available = False):  
        model_segmentation = UnetVAE3D(DATASET_INPUT_SHAPE,
                                in_channels=len(DATASET_USE_MODES),
                                out_channels=3,
                                init_channels=MODEL_INIT_CHANNELS,
                                p=MODEL_DROPOUT)
        if cuda_available & torch.cuda.is_available():
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          model_segmentation.cuda()
          checkpoint = torch.load(file_checkpoint)
        else:
          checkpoint = torch.load(file_checkpoint, map_location=torch.device('cpu'))
        model_segmentation.load_state_dict(checkpoint['model'])
        return model_segmentation


#******************** MRI Classification ********************

def make_data_loaders_classification(image_val):
      dataset = MRIClassificationDataset(image_val)
      batch_size = 1
      loaders = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
      return loaders

def build_model_classification(file_checkpoint, cuda_available = False): 
    growth =  32
    droprate = 0
    bottleneck = False
    reduce = 1

    model_classification = DenseNet3(3, growth, reduction=reduce, bottleneck=bottleneck, dropRate=droprate)
    if cuda_available & torch.cuda.is_available():
      model_classification.cuda()
      checkpoint = torch.load(file_checkpoint)
    else:
      checkpoint = torch.load(file_checkpoint, map_location=torch.device('cpu'))
    model_classification.load_state_dict(checkpoint['state_dict'])   

    # Inject model with M3d-CAM
    model_classification = medcam.inject(model_classification, output_dir="attention_maps", save_maps=True)
    model_classification.eval()

    return model_classification


def calculate_class_probablity(output_classi):
        output_classi = torch.exp(output_classi.cpu())
        classi_class = torch.argmax(output_classi)
        classi_prob = torch.max(output_classi)/torch.sum(output_classi)
        return classi_class, classi_prob

def main():    
    init_env('1') 
    is_cuda_run = False

    file_checkpoint_seg = 'backend/mri/model_parameters/seg_params.pth'
    file_checkpoint_classi = 'backend/mri/model_parameters/classi_params.pth.tar'    
    data_path = 'static/upload/mri'
    gradCAM_save_path = "static/upload/result/"


    # file_checkpoint_seg = '/content/drive/MyDrive/Unet-Densenet pipeline/best_model.pth'
    # file_checkpoint_classi = '/content/drive/MyDrive/densenet_169_dataclean/runs/classification_train/model_best.pth.tar'
    # data_path = "/content/drive/MyDrive/MRI Segmentation data/imagesTr" 
    # gradCAM_save_path = "/content/drive/MyDrive/Unet-Densenet pipeline/Grad CAM for segmentation set/" 

    model_seg = build_model_segmentation(file_checkpoint_seg, cuda_available = is_cuda_run)
    model_classi = build_model_classification(file_checkpoint_classi, cuda_available = is_cuda_run)

 
    data_loader_seg = make_data_loaders_segmenetation(data_path)
    data_loader_classi = make_data_loaders_classification(data_path)
    
    iterator_seg = iter(data_loader_seg)
    iterator_classi = iter(data_loader_classi)
  
    for i in range (len(data_loader_seg.dataset)):  
      print(i)
      # ************ Segmentation ************
      crop_volume, input_seg = iterator_seg.next()
      data_seg = crop_volume
      original_x = input_seg

      # ========== Cuda Available ==========
      if torch.cuda.is_available() & is_cuda_run:
        data_seg= data_seg.cuda(non_blocking=True)
      output_seg,_ = model_seg.unet(data_seg)
      pred_output = output_seg.squeeze()
      output_seg[output_seg>=0.5] = 1
      output_seg[output_seg<0.5] = 0
      output_seg = pred_output.cpu().detach().numpy() 

      coronal, transversal, sagittal = format_segmented_mri(output_seg, original_x)
      # visualise_segemnted_mri(coronal, transversal, sagittal)

      # ************ Classification ************
      data_classi, file_name = iterator_classi.next()

      # ========== Cuda Available ==========
      if torch.cuda.is_available() & is_cuda_run:
        data_classi= data_classi.cuda(non_blocking=True)
      output_classi = model_classi(data_classi)
      classification_class, probabaility = calculate_class_probablity(output_classi)
      print(classification_class.item(), round(probabaility.item(), 2))

      # ************ Grad CAMs ************
      new_file_name = visualize_grad_cam(model_classi, transversal, data_classi, gradCAM_save_path, file_name, i)

      return classification_class.item(), math.trunc(probabaility.item()*100), new_file_name



