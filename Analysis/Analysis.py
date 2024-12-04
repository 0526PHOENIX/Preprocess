"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import os
import sys
if os.path.isdir("/home/ccy/Data"):
    # Lab Server R740
    sys.path.append("/home/ccy/Data")
elif os.path.isdir("/home/phoenix/Data"):
    # Lab Server Supermicro
    sys.path.append("/home/phoenix/Data")
else:
    # Lab Local
    sys.path.append("C:/Users/user/Desktop/Data")

from tqdm import tqdm

import numpy as np
import nibabel as nib
import SimpleITK as sitk

import matplotlib.pyplot as plt

from scipy import io
from scipy import ndimage

import ants
from antspynet.utilities import brain_extraction

import torch
import torch.nn.functional as F

from Utils import *


"""
========================================================================================================================
Global Constant
========================================================================================================================
"""
MR_RAW = ""
CT_RAW = ""

MR = "C:/Users/user/Desktop/Data/Data/MR"
CT = "C:/Users/user/Desktop/Data/Data/CT"
HM = "C:/Users/user/Desktop/Data/Data/HM"
BR = "C:/Users/user/Desktop/Data/Data/BR"
SK = "C:/Users/user/Desktop/Data/Data/SK"
VS = "C:/Users/user/Desktop/Data/Data/VS"

TP = "C:/Users/user/Desktop/Data/Test/Test"

DATA_2D = "C:/Users/user/Desktop/Data/Data_2D"


METRICS = 11

LOSS_PIX = 0
LOSS_GDL = 1
LOSS_SIM = 2
LOSS_PER = 3

METRICS_HEAD_MAE    = 4
METRICS_HEAD_PSNR   = 5
METRICS_HEAD_SSIM   = 6
METRICS_BONE_MAE    = 7
METRICS_BONE_PSNR   = 8
METRICS_BONE_SSIM   = 9
METRICS_BONE_DICE   = 10


"""
========================================================================================================================
Analysis
========================================================================================================================
"""
class Analysis():

    """
    ====================================================================================================================
    Critical Parameters
    ====================================================================================================================
    """
    def __init__(self) -> None:

        print()
        print('=======================================================================================================')
        print('Initialization')
        print('=======================================================================================================')
        print()

        # Get File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)
        self.hmasks = os.listdir(HM)
        self.brains = os.listdir(BR)
        self.skulls = os.listdir(SK)

        # Check File Number
        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        # Get File Number
        self.len = 26

        # Loss
        self.loss = Loss(torch.device('cuda'))
        self.metrics = Metrics(torch.device('cuda'))

        # Log
        print('Done !')
        print()

        return

    """
    ====================================================================================================================
    Main Process
    ====================================================================================================================
    """
    def main(self) -> None:

        self.histogram_overall(5000)

        # self.histogram_subject(5000)

        # self.loss_and_metrics()

        return

    """
    ====================================================================================================================
    Plot Historgram of All Data
    ====================================================================================================================
    """
    def histogram_overall(self, bins: int = 5000) -> None:

        print()
        print('=======================================================================================================')
        print('Plot Historgram')
        print('=======================================================================================================')
        print()

        # Clear File Name List
        self.images.clear()
        self.labels.clear()
        self.hmasks.clear()
        self.brains.clear()
        self.skulls.clear()

        # Open File of Specifice Order
        with open(os.path.join(DATA_2D, 'Slice.txt'), 'r') as file:
            lines = file.readlines()

        # Get Specific Order
        for line in lines:

            # Split Out Numerical Part
            nums = line.split()

            # Form New File Name List with Specific Order
            for num in nums:
                if num.isdigit():
                    self.images.append('MR' + str(num) + '.nii')
                    self.labels.append('CT' + str(num) + '.nii')
                    self.hmasks.append('HM' + str(num) + '.nii')

        # Initialize cumulative histogram bins
        hist_image_train = np.zeros(bins)
        hist_label_train = np.zeros(bins)
        hist_image_val = np.zeros(bins)
        hist_label_val = np.zeros(bins)
        hist_image_test = np.zeros(bins)
        hist_label_test = np.zeros(bins)
        image_min, image_max = -1, 1
        label_min, label_max = -1000, 3000

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            image = image[hmask]
            label = label[hmask]

            temp_image = np.histogram(image, bins = bins, range = (image_min, image_max))[0].astype('float32')
            temp_label = np.histogram(label, bins = bins, range = (label_min, label_max))[0].astype('float32')

            if i < 20:
                hist_image_train += temp_image
                hist_label_train += temp_label
            elif i < 24:
                hist_image_val += temp_image
                hist_label_val += temp_label
            elif i < 26:
                hist_image_test += temp_image
                hist_label_test += temp_label
        print()

        hist_image = hist_image_train + hist_image_val + hist_image_test
        hist_label = hist_label_train + hist_label_val + hist_label_test
        
        hist_image /= hist_image.sum()
        hist_label /= hist_label.sum()
        hist_image_train /= hist_image_train.sum()
        hist_label_train /= hist_label_train.sum()
        hist_image_val /= hist_image_val.sum()
        hist_label_val /= hist_label_val.sum()
        hist_image_test /= hist_image_test.sum()
        hist_label_test /= hist_label_test.sum()

        # Plot histograms
        fig, axs = plt.subplots(4, 2)

        for ax in axs.flat:
            ax.set_yticks([])
            ax.set_ylim(0, 1 / bins * 4)

        # MR image histogram
        ax = axs[0][0]
        ax.bar(np.linspace(image_min, image_max, bins), hist_image, width = (image_max - image_min) / bins)
        ax.set_ylabel('Overall MR')

        # CT label histogram
        ax = axs[0][1]
        ax.bar(np.linspace(label_min, label_max, bins), hist_label, width = (label_max - label_min) / bins)
        ax.set_ylabel('Overall CT')
        
        # MR train image histogram
        ax = axs[1][0]
        ax.bar(np.linspace(image_min, image_max, bins), hist_image_train, width = (image_max - image_min) / bins)
        ax.set_ylabel('Training Set MR')

        # CT train label histogram
        ax = axs[1][1]
        ax.bar(np.linspace(label_min, label_max, bins), hist_label_train, width = (label_max - label_min) / bins)
        ax.set_ylabel('Training Set CT')

        # MR val image histogram
        ax = axs[2][0]
        ax.bar(np.linspace(image_min, image_max, bins), hist_image_val, width = (image_max - image_min) / bins)
        ax.set_ylabel('Validataion Set MR')

        # CT val label histogram
        ax = axs[2][1]
        ax.bar(np.linspace(label_min, label_max, bins), hist_label_val, width = (label_max - label_min) / bins)
        ax.set_ylabel('Validataion Set CT')

        # MR test image histogram
        ax = axs[3][0]
        ax.bar(np.linspace(image_min, image_max, bins), hist_image_test, width = (image_max - image_min) / bins)
        ax.set_ylabel('Testing Set MR')

        # CT test label histogram
        ax = axs[3][1]
        ax.bar(np.linspace(label_min, label_max, bins), hist_label_test, width = (label_max - label_min) / bins)
        ax.set_ylabel('Testing Set CT')

        plt.show()

        # Ascending Sort File Name List
        self.images.sort()
        self.labels.sort()
        self.hmasks.sort()

        return

    """
    ====================================================================================================================
    Plot Historgram of Each Subject
    ====================================================================================================================
    """
    def histogram_subject(self, bins: int = 5000) -> None:

        print()
        print('=======================================================================================================')
        print('Plot Historgram of Each Subject')
        print('=======================================================================================================')
        print()

        # Initialize cumulative histogram bins
        image_min, image_max = -1, 1
        label_min, label_max = -1000, 3000

        # Plot histograms
        fig_image, axs_image = plt.subplots(4, 7)
        fig_label, axs_label = plt.subplots(4, 7)

        for ax_image, ax_label in zip(axs_image.flat, axs_label.flat):
            ax_image.set_yticks([])
            ax_label.set_yticks([])
            ax_image.set_ylim(0, 1 / bins * 4)
            ax_label.set_ylim(0, 1 / bins * 4)

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            image = image[hmask]
            label = label[hmask]

            hist_image_subject = np.histogram(image, bins = bins, range = (image_min, image_max))[0].astype('float32')
            hist_label_subject = np.histogram(label, bins = bins, range = (label_min, label_max))[0].astype('float32')

            hist_image_subject /= hist_image_subject.sum()
            hist_label_subject /= hist_label_subject.sum()
            
            ax_image = axs_image[i // 7][i % 7]
            ax_image.bar(np.linspace(image_min, image_max, bins), hist_image_subject, width = (image_max - image_min) / bins)
            ax_image.set_ylabel(self.images[i])

            ax_label = axs_label[i // 7][i % 7]
            ax_label.bar(np.linspace(label_min, label_max, bins), hist_label_subject, width = (label_max - label_min) / bins)
            ax_label.set_ylabel(self.labels[i])
        print()

        plt.show()

        return

    """
    ====================================================================================================================
    
    ====================================================================================================================
    """
    def loss_and_metrics(self) -> None:

        print()
        print('=======================================================================================================')
        print('Compute Loss and Metrics')
        print('=======================================================================================================')
        print()

        # Buffer for Metrics
        metrics = torch.zeros(METRICS, self.len, device = torch.device('cuda'))

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('bool')

            # [-1, 1]
            label -= -1000
            label /= 4000
            label = (label * 2) - 1

            pix, gdl, sim, per = self.loss.eva_head(image, label)

            image = ((image + 1) * 2000) - 1000
            label = ((label + 1) * 2000) - 1000

            head_mae, head_psnr, head_ssim = self.metrics.eva_head(image, label, hmask)
            bone_mae, bone_psnr, bone_ssim, bone_dice = self.metrics.eva_bone(image, skull, hmask, brain)

            # Save Metrics
            metrics[LOSS_PIX, i] = pix
            metrics[LOSS_GDL, i] = gdl
            metrics[LOSS_SIM, i] = sim
            metrics[LOSS_PER, i] = per
            # Save Metrics
            metrics[METRICS_HEAD_MAE, i]  = head_mae
            metrics[METRICS_HEAD_PSNR, i] = head_psnr
            metrics[METRICS_HEAD_SSIM, i] = head_ssim
            metrics[METRICS_BONE_MAE, i]  = bone_mae
            metrics[METRICS_BONE_PSNR, i] = bone_psnr
            metrics[METRICS_BONE_SSIM, i] = bone_ssim
            metrics[METRICS_BONE_DICE, i] = bone_dice
        print()

        # Torch Tensor to Numpy Array
        metrics = metrics.to('cpu').detach().numpy()

        # Output Format
        title = "{:<35}{:<35}{:<35}"
        space = "{:<35}{:<35.2f}{:<35.2f}"

        # Title
        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('Loss', 'Mean', 'STD'))
        print('-------------------------------------------------------------------------------------------------------')

        # Print Metrics
        print(space.format('PIX', metrics[LOSS_PIX].mean(), metrics[LOSS_PIX ].std()))
        print(space.format('GDL', metrics[LOSS_GDL].mean(), metrics[LOSS_GDL ].std()))
        print(space.format('SIM', metrics[LOSS_SIM].mean(), metrics[LOSS_SIM ].std()))
        print(space.format('PER', metrics[LOSS_PER].mean(), metrics[LOSS_PER ].std()))
        print()

        # Title
        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('Region Metrics', 'Mean', 'STD'))
        print('-------------------------------------------------------------------------------------------------------')

        # Print Metrics
        print(space.format('Head MAE', metrics[METRICS_HEAD_MAE].mean(), metrics[METRICS_HEAD_MAE].std()))
        print(space.format('Head PSNR', metrics[METRICS_HEAD_PSNR].mean(), metrics[METRICS_HEAD_PSNR].std()))
        print(space.format('Head SSIM', metrics[METRICS_HEAD_SSIM].mean(), metrics[METRICS_HEAD_SSIM].std()))
        print()
        print(space.format('Bone MAE', metrics[METRICS_BONE_MAE].mean(), metrics[METRICS_BONE_MAE].std()))
        print(space.format('Bone PSNR', metrics[METRICS_BONE_PSNR].mean(), metrics[METRICS_BONE_PSNR].std()))
        print(space.format('Bone SSIM', metrics[METRICS_BONE_SSIM].mean(), metrics[METRICS_BONE_SSIM].std()))
        print(space.format('Bone DICE', metrics[METRICS_BONE_DICE].mean(), metrics[METRICS_BONE_DICE].std()))
        print()

        return

    
"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    analysis = Analysis()
    analysis.main()