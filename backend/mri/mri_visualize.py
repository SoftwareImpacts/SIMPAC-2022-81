
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

def format_segmented_mri(output, original_x):
        new_output = []
        for channel in range(3):
            full_slices = []
            for depth in range(128):
                buidling_block = np.repeat(0.0, 192).reshape(1,192)
                up_zeros = np.repeat(0.0, 192).reshape(1,192)
                for _ in range(39):
                    buidling_block = np.concatenate((buidling_block,up_zeros),axis = 0)
                
                buidling_block = np.concatenate((buidling_block,output[channel,:,:,depth]))

                

                
                for _ in range(40):
                    buidling_block = np.concatenate((buidling_block,up_zeros)) 

                
                side_zeros = np.repeat(0.0, 240).reshape(240,1)
                for _ in range(23):
                    side_zeros = np.concatenate((side_zeros,np.repeat(0.0, 240).reshape(240,1)),axis=1)    
                      
                side_zeros = np.concatenate((side_zeros,buidling_block),axis=1)

                for _ in range(24):
                    side_zeros = np.concatenate((side_zeros,np.repeat(0.0, 240).reshape(240,1)),axis=1)
                

                full_slices.append(side_zeros)
                
            full_slices = np.stack(full_slices)
            channel_output = np.concatenate((np.zeros((14,240,240)),full_slices,np.zeros((13,240,240))),axis=0)
            channel_output = np.moveaxis(channel_output,0,2)

            new_output.append(channel_output)

        new_output=np.stack(new_output) 
        tumor_background = np.zeros(new_output.shape[1:])
        tumor_background = np.expand_dims(np.logical_or(new_output[2], np.logical_or(new_output[0],new_output[1])),0)
        new_output = np.concatenate((new_output,tumor_background))
        # np.where(new_output[1,:,:,:]==1,2,0)
        # np.where(new_output[2,:,:,:]==1,3,0)

        new_output = np.moveaxis(new_output,0,3)
        new_output.shape

        image = original_x.squeeze().numpy()
        image = np.moveaxis(image,0,3) 

        segmented_output = new_output
        segmented_output[:, :, :, 3] = np.where(segmented_output[:, :, :, 3]==1,0,1)

        image_norm = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        image_norm.shape

        labeled_image = np.zeros_like(segmented_output[:, :, :, 0:3])

        # remove tumor part from image
        labeled_image[:, :, :, 0] = image_norm * (segmented_output[:, :, :, 3])
        labeled_image[:, :, :, 1] = image_norm * (segmented_output[:, :, :, 3])
        labeled_image[:, :, :, 2] = image_norm * (segmented_output[:, :, :, 3])

        # color labels
        labeled_image += segmented_output[:, :, :, 0:3] * 255

        labeled_image.shape

        data_all = []
        data_all.append(labeled_image)
        np.array(data_all).shape

        # coronal plane
        coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
        coronal = np.rot90(coronal, 1)
        
        # transversal plane
        transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
        transversal = np.rot90(transversal, 2)

        # sagittal plane
        sagittal = np.transpose(data_all, [2, 3, 1, 4, 0])
        sagittal = np.rot90(sagittal, 1)    

        image = original_x.squeeze().numpy()

        return (coronal, transversal, sagittal) 

def visualise_segemnted_mri(coronal, transversal, sagittal):
        fig, ax = plt.subplots(3, 6, figsize=[16, 9])

        for i in range(6):
            n = np.random.randint(coronal.shape[2])
            ax[0][i].imshow(np.squeeze(coronal[:, :, n, :]).astype('uint32'))
            ax[0][i].set_xticks([])
            ax[0][i].set_yticks([])
            title = 'depth = '+ str(n)
            ax[0][i].title.set_text(title)
            if i == 0:
                ax[0][i].set_ylabel('Coronal', fontsize=15)

        for i in range(6):
            n = np.random.randint(transversal.shape[2])
            ax[1][i].imshow(np.squeeze(transversal[:, :, n, :]).astype('uint32'))
            ax[1][i].set_xticks([])
            ax[1][i].set_yticks([])
            title = 'depth = '+ str(n)
            ax[1][i].title.set_text(title)
            if i == 0:
                ax[1][i].set_ylabel('Transversal', fontsize=15)

        for i in range(6):
            n = np.random.randint(sagittal.shape[2])
            ax[2][i].imshow(np.squeeze(sagittal[:, :, n, :]).astype('uint32'))
            ax[2][i].set_xticks([])
            ax[2][i].set_yticks([])
            title = 'depth = '+ str(n)
            ax[2][i].title.set_text(title)
            if i == 0:
                ax[2][i].set_ylabel('Sagittal', fontsize=15)

        fig.subplots_adjust(wspace=0, hspace=0)
        # fig.savefig('static/upload/result/result.png')   # save the figure to file
        # plt.close(fig)

def visualize_grad_cam(model_classification, transversal, input, gradCAM_save_path, file_name, index):
    cmap = plt.get_cmap('RdYlBu_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    map = model_classification.get_attention_map()
    heatmap = torch.from_numpy(map)
    heatmap = nn.functional.interpolate(heatmap,size=(input.shape[2], input.shape[3], input.shape[4]), mode='trilinear')
    heatmap = heatmap.squeeze()

    input = torch.rot90(input,3,(2,3))
    heatmap = torch.rot90(heatmap,3,(0,1))

    fig, ax = plt.subplots(16,4, figsize=[6, 30], squeeze=False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    fig.subplots_adjust(wspace=0.05, hspace=0)

    k = 0

    for depth in range(20,55,10):
        n_trans = round((140/120)*depth)
        segment_transversal = np.squeeze(transversal[:, :, n_trans, :])
        segment = segment_transversal
        # only for classification dataset
        segment = np.flip(segment_transversal,1) 
        crop_size = (160,140)
        height, width, _ = segment.shape
        sx = ((height - crop_size[0] - 1) // 2)+10
        sy = (width - crop_size[1] - 1) // 2
        segment = segment[ sx:sx + crop_size[0], sy:sy + crop_size[1],:]

    
        display = 'depth = '+str(depth)
        ax[k][0].set_ylabel(display, fontsize=15)
        ax[k][0].imshow(segment.astype('uint32'))
        ax[k][1].imshow(heatmap[:,:,depth], cmap='RdYlBu_r')
        ax[k][2].imshow(input[0,2,:,:,depth].cpu().detach().numpy(), cmap='gray')
        ax[k][3].imshow(input[0,2,:,:,depth].cpu().detach().numpy(), cmap='gray')
        ax[k][3].imshow(heatmap[:,:,depth], cmap='RdYlBu_r', alpha=0.6)
        k+=1

    for depth in range(60,120,5):
        n_trans = round((140/120)*depth)
        segment_transversal = np.squeeze(transversal[:, :, n_trans, :])
        segment = segment_transversal
        
        # only for classification dataset
        segment = np.flip(segment_transversal,1)
        crop_size = (160,140)
        height, width, _ = segment.shape
        sx = ((height - crop_size[0] - 1) // 2)+10
        sy = (width - crop_size[1] - 1) // 2
        segment = segment[ sx:sx + crop_size[0], sy:sy + crop_size[1],:]

        display = 'depth = '+str(depth)
        ax[k][0].set_ylabel(display, fontsize=15)
        ax[k][0].imshow(segment.astype('uint32'))
        ax[k][1].imshow(heatmap[:,:,depth], cmap='RdYlBu_r')
        ax[k][2].imshow(input[0,2,:,:,depth].cpu().detach().numpy(), cmap='gray')
        ax[k][3].imshow(input[0,2,:,:,depth].cpu().detach().numpy(), cmap='gray')
        ax[k][3].imshow(heatmap[:,:,depth], cmap='RdYlBu_r', alpha=0.6)
        k+=1


    new_file_name = gradCAM_save_path+str(index+1)+'_'+file_name[0]+'.png'   # save the figure to file
    fig.savefig(new_file_name)   # save the figure to file
    plt.close(fig)
    return str(index+1)+'_'+file_name[0]+'.png'