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
sys.path.append("C:/Users/user/Desktop/Data")

from tqdm import tqdm
from typing import Literal

from scipy import io
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch

from Utils import *


"""
========================================================================================================================
Global Constant
========================================================================================================================
"""
# Data File Path
MR_RAW = "C:/Users/user/Desktop/Data/Data_Raw/MR"
CT_RAW = "C:/Users/user/Desktop/Data/Data_Raw/CT"

MR = "C:/Users/user/Desktop/Data/Data/MR"
CT = "C:/Users/user/Desktop/Data/Data/CT"
HM = "C:/Users/user/Desktop/Data/Data/HM"
BR = "C:/Users/user/Desktop/Data/Data/BR"
SK = "C:/Users/user/Desktop/Data/Data/SK"

DATA_2D = "C:/Users/user/Desktop/Data/Data_2D"

# Statistic Buffer
STATIST = 4

STATIST_MR_MEAN = 0
STATIST_CT_MEAN = 1
STATIST_MR_STD  = 2
STATIST_CT_STD  = 3

# Metrics Buffer
METRICS = 10

METRICS_HEAD_MAE    = 0
METRICS_HEAD_RMSE   = 1
METRICS_HEAD_PSNR   = 2
METRICS_HEAD_SSIM   = 3
METRICS_HEAD_LPIPS  = 4
METRICS_BONE_MAE    = 5
METRICS_BONE_RMSE   = 6
METRICS_BONE_PSNR   = 7
METRICS_BONE_SSIM   = 8
METRICS_BONE_DICE   = 9

# Plot
HT = "C:/Users/user/Desktop/Data/Analysis/Histogram"
CK = "C:/Users/user/Desktop/Data/Analysis/Checker"
PF = "C:/Users/user/Desktop/Data/Analysis/Profile"
BP = "C:/Users/user/Desktop/Data/Analysis/Boxplot"

PATH = [HT, CK, PF, BP]

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
        print('=' * 110)
        print('Initialization')
        print('=' * 110)
        print()

        # Get File Name
        self.images = sorted(os.listdir(MR))
        self.labels = sorted(os.listdir(CT))
        self.hmasks = sorted(os.listdir(HM))
        self.brains = sorted(os.listdir(BR))
        self.skulls = sorted(os.listdir(SK))

        # Check File Number
        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        # Get File Number
        self.len = 26

        # Metrics
        self.get_metrics = Metrics(torch.device('cuda'))

        # Check Path
        for path in PATH:
            if not os.path.exists(path):
                os.makedirs(path)

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

        # self.print_rawdata()
        self.print_statist()
        # self.print_metrics()
        # self.print_stissue()

        # self.histogram()
        # self.checker()
        # self.profile()
        # self.boxplot()

        return

    """
    ====================================================================================================================
    Print Raw Data Statistic
    ====================================================================================================================
    """
    def print_rawdata(self) -> None:

        print()
        print('=' * 110)
        print('Print Raw Data Statistic')
        print('=' * 110)
        print()

        # Raw File Name
        self.images = os.listdir(MR_RAW)
        self.labels = os.listdir(CT_RAW)

        # Output Format
        title = "{:^17}|{:^17}|{:^17}|{:^17}|{:^17}|{:^20}"
        space = "{:^17}|{:^17.3f}|{:^17.3f}|{:^17.3f}|{:^17.3f}|{:^20}"

        for i in range(self.len):

            # Load Data
            image = io.loadmat(os.path.join(MR_RAW, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT_RAW, self.labels[i]))['CT'].astype('float32')

            # Data Shape
            shape = str(image.shape)

            # Title
            print('-' * 110)
            print(title.format('File Name', 'Mean', 'STD', 'Min', 'Max', 'Shape'))
            print('-' * 110)

            # Result
            print(space.format(self.images[i], image.mean(), image.std(), image.min(), image.max(), shape))
            print(space.format(self.labels[i], label.mean(), label.std(), label.min(), label.max(), shape))
            print()
        print()

        # New File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)

        return

    """
    ====================================================================================================================
    Print Statistic
    ====================================================================================================================
    """
    def print_statist(self) -> None:

        print()
        print('=' * 110)
        print('Print Statistic')
        print('=' * 110)
        print()

        # Output Format
        title = "{:^17}|{:^17}|{:^17}|{:^17}|{:^17}|{:^20}"
        space = "{:^17}|{:^17.3f}|{:^17.3f}|{:^17.3f}|{:^17.3f}|{:^20}"

        # Buffer for Statistic
        statist = np.zeros((STATIST, self.len))

        for i in range(self.len):

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Original Data Shape
            shape = str(image.shape)

            # Remove Air Region
            image = image[hmask]
            label = label[hmask]

            # Title
            print('-' * 110)
            print(title.format('File Name', 'Mean', 'STD', 'Min', 'Max', 'Shape'))
            print('-' * 110)

            # Result
            print(space.format(self.images[i], image.mean(), image.std(), image.min(), image.max(), shape))
            print(space.format(self.labels[i], label.mean(), label.std(), label.min(), label.max(), shape))
            print()

            # Save Statistic
            statist[STATIST_MR_MEAN, i]  = image.mean()
            statist[STATIST_CT_MEAN, i]  = label.mean()
            statist[STATIST_MR_STD, i]   = image.std()
            statist[STATIST_CT_STD, i]   = label.std()
        print()

        # Rearrange Statistic
        image_mean_mean  = round(statist[STATIST_MR_MEAN].mean().item(), 3)
        label_mean_mean  = round(statist[STATIST_CT_MEAN].mean().item(), 3)
        image_mean_std   = round(statist[STATIST_MR_MEAN].std().item(), 3)
        label_mean_std   = round(statist[STATIST_CT_MEAN].std().item(), 3)
        image_std_mean   = round(statist[STATIST_MR_STD].mean().item(), 3)
        label_std_mean   = round(statist[STATIST_CT_STD].mean().item(), 3)
        image_std_std    = round(statist[STATIST_MR_STD].std().item(), 3)
        label_std_std    = round(statist[STATIST_CT_STD].std().item(), 3) 
        
        # Title
        print('-' * 110)
        print(title.format('', 'Mean Mean', 'Mean STD', 'STD Mean', 'STD STD', '---'))
        print('-' * 110)

        # Result
        print(space.format('MR Mean & STD', image_mean_mean, image_mean_std, image_std_mean, image_std_std, '---'))
        print(space.format('CT Mean & STD', label_mean_mean, label_mean_std, label_std_mean, label_std_std, '---'))
        print()

        return
    
    """
    ====================================================================================================================
    Print Metrics
    ====================================================================================================================
    """
    def print_metrics(self) -> None:

        print()
        print('=' * 110)
        print('Print Metrics')
        print('=' * 110)
        print()

        # Output Format
        space = "{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}"

        # Buffer for Metrics
        metrics = torch.zeros(METRICS, self.len, device = torch.device('cuda'))

        for i in range(self.len):

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('bool')

            # [-1000, 3000]
            image = ((image + 1) * 2000) - 1000

            # Head MAE, PSNR, SSIM, LPIPS
            head_mae, head_rmse, head_psnr, head_ssim, head_lpips = self.get_metrics.eva_head(image, label, hmask, 'region')

            # Bone MAE, PSNR, SSIM, DICE
            bone_mae, bone_rmse, bone_psnr, bone_ssim, bone_dice = self.get_metrics.eva_bone(image, skull, hmask, brain, 'region')

            # Save Metrics
            metrics[METRICS_HEAD_MAE, i]   = head_mae
            metrics[METRICS_HEAD_RMSE, i]  = head_rmse
            metrics[METRICS_HEAD_PSNR, i]  = head_psnr
            metrics[METRICS_HEAD_SSIM, i]  = head_ssim
            metrics[METRICS_HEAD_LPIPS, i] = head_lpips
            metrics[METRICS_BONE_MAE, i]   = bone_mae
            metrics[METRICS_BONE_RMSE, i]  = bone_rmse
            metrics[METRICS_BONE_PSNR, i]  = bone_psnr
            metrics[METRICS_BONE_SSIM, i]  = bone_ssim
            metrics[METRICS_BONE_DICE, i]  = bone_dice

            # Title
            print('-' * 110)
            print(space.format('File Region', 'MAE', 'RMSE', 'PSNR', 'SSIM', 'LPIPS', 'DICE'))
            print('-' * 110)

            # Result
            print(space.format(str(i + 1) + ' Head', head_mae, head_rmse, head_psnr, head_ssim, head_lpips, '---'))
            print(space.format(str(i + 1) + ' Bone', bone_mae, bone_rmse, bone_psnr, bone_ssim, '---', bone_dice))
            print()
        print()

        # Torch Tensor to Numpy Array
        metrics = metrics.to('cpu').detach().numpy()

        # Rearrange Metrics
        head_mae    = round(metrics[METRICS_HEAD_MAE].mean().item(), 3)
        head_rmse   = round(metrics[METRICS_HEAD_RMSE].mean().item(), 3)
        head_psnr   = round(metrics[METRICS_HEAD_PSNR].mean().item(), 3)
        head_ssim   = round(metrics[METRICS_HEAD_SSIM].mean().item(), 3)
        head_lpips  = round(metrics[METRICS_HEAD_LPIPS].mean().item(), 3)
        bone_mae    = round(metrics[METRICS_BONE_MAE].mean().item(), 3)
        bone_rmse   = round(metrics[METRICS_BONE_RMSE].mean().item(), 3)
        bone_psnr   = round(metrics[METRICS_BONE_PSNR].mean().item(), 3)
        bone_ssim   = round(metrics[METRICS_BONE_SSIM].mean().item(), 3)
        bone_dice   = round(metrics[METRICS_BONE_DICE].mean().item(), 3)

        # Title
        print('-' * 110)
        print(space.format('File Region', 'MAE', 'RMSE', 'PSNR', 'SSIM', 'LPIPS', 'DICE'))
        print('-' * 110)

        # Result
        print(space.format('Overall Head', head_mae, head_rmse, head_psnr, head_ssim, head_lpips, '---'))
        print(space.format('Overall Bone', bone_mae, bone_rmse, bone_psnr, bone_ssim, '---', bone_dice))
        print()

        return
    
    """
    ====================================================================================================================
    Print CT Soft Tissue Behavior
    ====================================================================================================================
    """
    def print_stissue(self) -> None:

        print()
        print('=' * 110)
        print('Check CT Behavior')
        print('=' * 110)
        print()

        # Output Format
        title = "{:^35}|{:^35}|{:^35}"
        space = "{:^35}|{:^35.3f}|{:^35.3f}"

        for i in range(self.len):

            # Load Data
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Soft Tissue Intensity
            tissue = label[label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 3 * 2]

            # Remove Air Region
            label = label[hmask]

            # Title
            print('-' * 110)
            print(title.format('File Name', 'Mean Value of Head', 'Soft Tissue Intensity'))
            print('-' * 110)

            # Result
            print(space.format(self.labels[i], label.mean(), tissue))
            print()
        print()

        return

    """
    ====================================================================================================================
    Plot Historgram
    ====================================================================================================================
    """
    def histogram(self, bins: int = 5000) -> None:

        print()
        print('=' * 110)
        print('Plot Historgram')
        print('=' * 110)
        print()

        # Clear File Name List
        self.images.clear()
        self.labels.clear()
        self.hmasks.clear()

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

        # Cumulative Histogram Bin
        hist_image = np.array([np.zeros(bins) for _ in range(4)])
        hist_label = np.array([np.zeros(bins) for _ in range(4)])

        # Data Range
        image_min, image_max = -1, 1
        label_min, label_max = -1000, 3000

        # Plot histograms
        fig, axs = plt.subplots(4, 2, figsize = (15, 7.5))

        # Remove Redundancy
        for ax in axs.flat:
            ax.set_yticks([])
            ax.set_ylim(0, 1 / bins * 4)

        # Plot histograms
        fig_image, axs_image = plt.subplots(4, 7, figsize = (15, 7.5))
        fig_label, axs_label = plt.subplots(4, 7, figsize = (15, 7.5))

        # Remove Redundancy
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

            # Remove Background + Flatten Image
            image = image[hmask]
            label = label[hmask]

            # Histogram
            hist_image_subject = np.histogram(image, bins = bins, range = (image_min, image_max))[0].astype('float32')
            hist_label_subject = np.histogram(label, bins = bins, range = (label_min, label_max))[0].astype('float32')

            # Split Set
            if i < 20:
                hist_image[1] += hist_image_subject
                hist_label[1] += hist_label_subject
            elif i < 24:
                hist_image[2] += hist_image_subject
                hist_label[2] += hist_label_subject
            elif i < 26:
                hist_image[3] += hist_image_subject
                hist_label[3] += hist_label_subject

            # [0, 1]
            hist_image_subject /= hist_image_subject.sum()
            hist_label_subject /= hist_label_subject.sum()

            # Get Real Index
            index = int(self.images[i][2:4]) - 1

            # MR Histogram
            ax_image = axs_image[index // 7][index % 7]
            ax_image.bar(np.linspace(image_min, image_max, bins), hist_image_subject, width = (image_max - image_min) / bins)
            ax_image.set_ylabel(self.images[i])

            # CT Histogram
            ax_label = axs_label[index // 7][index % 7]
            ax_label.bar(np.linspace(label_min, label_max, bins), hist_label_subject, width = (label_max - label_min) / bins)
            ax_label.set_ylabel(self.labels[i])
        print()

        # Overall Histogram
        hist_image[0] = hist_image.sum(axis = 0)
        hist_label[0] = hist_label.sum(axis = 0)
        
        # [0, 1]
        for i in range(4):
            hist_image[i] /= hist_image[i].sum()
            hist_label[i] /= hist_label[i].sum()

        # Specify Dataset
        dataset = {0: 'All', 1: 'Training', 2: 'Validation', 3: 'Testing'}

        # Histogram
        for i in range(4):
            # MR
            ax = axs[i][0]
            ax.bar(np.linspace(image_min, image_max, bins), hist_image[i], width = (image_max - image_min) / bins)
            ax.set_ylabel('MR' + ' ' + dataset[i])
            # CT
            ax = axs[i][1]
            ax.bar(np.linspace(label_min, label_max, bins), hist_label[i], width = (label_max - label_min) / bins)
            ax.set_ylabel('CT' + ' ' + dataset[i])

        # Save Histogram
        fig.tight_layout()
        fig.savefig(os.path.join(HT, 'Overall.png'), format = 'png', dpi = 300)
        plt.close(fig)

        fig_image.tight_layout()
        fig_image.savefig(os.path.join(HT, 'MR.png'), format = 'png', dpi = 300)
        plt.close(fig_image)

        fig_label.tight_layout()
        fig_label.savefig(os.path.join(HT, 'CT.png'), format = 'png', dpi = 300)
        plt.close(fig_label)

        # Ascending Sort File Name List
        self.images.sort()
        self.labels.sort()
        self.hmasks.sort()

        return

    """
    ====================================================================================================================
    Plot Checkerboard
    ====================================================================================================================
    """
    def checker(self, index: int = None, check: int = 8) -> None:

        print()
        print('=' * 110)
        print('Plot Checkerboard')
        print('=' * 110)
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # [-1, 1]
            label -= -1000
            label /= 4000
            label = (label * 2) - 1

            # Target Index and Buffer for Checkerboard
            index = index or (image.shape[2] // 3 * 2)
            board = np.zeros((image.shape[0], image.shape[1]), dtype = 'float32')

            # Plot Checkerboard
            for j in range(board.shape[0]):
                for k in range(board.shape[1]):
                    if (j // (image.shape[0] // check) % 2) ^ (k // (image.shape[1] // check) % 2):
                        board[j, k] = image[j, k, index]
                    else:
                        board[j, k] = label[j, k, index]

            # Rotate
            board = np.rot90(board, k = 1)

            # Figure
            plt.figure()

            # Plot Checkerboard
            plt.imshow(board, cmap = 'gray')
            plt.axis('off')

            # Save Checkerboard
            plt.tight_layout()
            plt.savefig(os.path.join(CK, self.images[i][2:4] + '.png'), format = 'png', dpi = 300)
            plt.close()
        print()

        return

    """
    ====================================================================================================================
    Plot Profile
    ====================================================================================================================
    """
    def profile(self, index: int = None, mode: str | Literal['Horizontal', 'Vertical'] = 'Horizontal') -> None:

        print()
        print('=' * 110)
        print('Plot Profile')
        print('=' * 110)
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # Rotate
            image = np.rot90(image, k = 1, axes = (0, 1))
            label = np.rot90(label, k = 1, axes = (0, 1))

            # [-1000, 3000]
            image = ((image + 1) * 2000) - 1000

            # Target Line Index
            index = index or (image.shape[0] // 2)

            # Figure
            _, axs = plt.subplots(1, 2, figsize = (15, 7.5))

            # Plot CT
            axs[0].imshow(label[:, :, label.shape[2] // 3 * 2], cmap = 'gray')
            axs[0].axis('off')

            # Target Line
            if mode == 'Horizontal':
                # Plot Target Line
                axs[0].axhline(index, color = 'cyan', linewidth = 2)
                # Target Line Index
                image = image[index, :, image.shape[2] // 3 * 2]
                label = label[index, :, label.shape[2] // 3 * 2]
            elif mode == 'Vertical':
                # Plot Target Line
                axs[0].axvline(index, color = 'cyan', linewidth = 2)
                # Target Line Index
                image = image[:, index, image.shape[2] // 3 * 2]
                label = label[:, index, label.shape[2] // 3 * 2]
            else:
                raise TypeError('Invalid Mode')
            
            # Plot Target Line Profile
            axs[1].plot(range(0, 256), image, label = 'MR', color = 'green')
            axs[1].plot(range(0, 256), label, label = 'CT', color = 'red')
            axs[1].set_xlim(0, 255)
            axs[1].set_ylim(-1000, 3000)
            axs[1].legend(loc = 'upper left', fontsize = 12)
            
            # Save Profile
            plt.tight_layout()
            plt.savefig(os.path.join(PF, self.images[i][2:4] + '.png'), format = 'png', dpi = 300)
            plt.close()
        print()

        return
    
    """
    ====================================================================================================================
    Plot Boxplot
    ====================================================================================================================
    """
    def boxplot(self) -> None:

        print()
        print('=' * 110)
        print('Plot Boxplot')
        print('=' * 110)
        print()

        # Buffer for Metrics
        metrics_mrct = torch.zeros(METRICS, self.len, device = torch.device('cuda'))
        metrics_ctct = torch.zeros(METRICS, self.len, device = torch.device('cuda'))

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('bool')

            # [-1000, 3000]
            image = ((image + 1) * 2000) - 1000

            # Head MAE, PSNR, SSIM, LPIPS
            head_mae, head_rmse, head_psnr, head_ssim, head_lpips = self.get_metrics.eva_head(image, label, hmask, 'region')

            # Bone MAE, PSNR, SSIM, DICE
            bone_mae, bone_rmse, bone_psnr, bone_ssim, bone_dice = self.get_metrics.eva_bone(image, skull, hmask, brain, 'region')

            # Save Metrics
            metrics_mrct[METRICS_HEAD_MAE, i]   = head_mae
            metrics_mrct[METRICS_HEAD_RMSE, i]  = head_rmse
            metrics_mrct[METRICS_HEAD_PSNR, i]  = head_psnr
            metrics_mrct[METRICS_HEAD_SSIM, i]  = head_ssim
            metrics_mrct[METRICS_HEAD_LPIPS, i] = head_lpips
            metrics_mrct[METRICS_BONE_MAE, i]   = bone_mae
            metrics_mrct[METRICS_BONE_RMSE, i]  = bone_rmse
            metrics_mrct[METRICS_BONE_PSNR, i]  = bone_psnr
            metrics_mrct[METRICS_BONE_SSIM, i]  = bone_ssim
            metrics_mrct[METRICS_BONE_DICE, i]  = bone_dice

            # Head MAE, PSNR, SSIM, LPIPS
            head_mae, head_rmse, head_psnr, head_ssim, head_lpips = self.get_metrics.eva_head(label, label, hmask, 'region')

            # Bone MAE, PSNR, SSIM, DICE
            bone_mae, bone_rmse, bone_psnr, bone_ssim, bone_dice = self.get_metrics.eva_bone(label, skull, hmask, brain, 'region')

            # Save Metrics
            metrics_ctct[METRICS_HEAD_MAE, i]   = head_mae
            metrics_ctct[METRICS_HEAD_RMSE, i]  = head_rmse
            metrics_ctct[METRICS_HEAD_PSNR, i]  = head_psnr
            metrics_ctct[METRICS_HEAD_SSIM, i]  = head_ssim
            metrics_ctct[METRICS_HEAD_LPIPS, i] = head_lpips
            metrics_ctct[METRICS_BONE_MAE, i]   = bone_mae
            metrics_ctct[METRICS_BONE_RMSE, i]  = bone_rmse
            metrics_ctct[METRICS_BONE_PSNR, i]  = bone_psnr
            metrics_ctct[METRICS_BONE_SSIM, i]  = bone_ssim
            metrics_ctct[METRICS_BONE_DICE, i]  = bone_dice
        print()

        # Torch Tensor to Numpy Array
        metrics_mrct = metrics_mrct.to('cpu').detach().numpy()
        metrics_ctct = metrics_ctct.to('cpu').detach().numpy()

        # Create boxplot
        fig_head, axs_head = plt.subplots(2, 3, figsize = (8, 6))
        fig_bone, axs_bone = plt.subplots(2, 3, figsize = (8, 6))

        # Format Buffer
        categories = ['MR vs CT', 'CT vs CT']
        list_head = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'LPIPS']
        list_bone = ['MAE', 'RMSE', 'PSNR', 'SSIM', 'DICE']
        for i in range(5):
            
            # Get Subplot Index
            ax_head = axs_head[i // 3][i % 3]
            ax_bone = axs_bone[i // 3][i % 3]

            # Extract Data from Numpy Array
            data_head = [metrics_mrct[i], metrics_ctct[i]]
            data_bone = [metrics_mrct[i + 5], metrics_ctct[i + 5]]

            # Boxplot
            box_head = ax_head.boxplot(data_head, patch_artist = True, widths = 0.6, showfliers = False)
            box_bone = ax_bone.boxplot(data_bone, patch_artist = True, widths = 0.6, showfliers = False)

            # Set Boxplot Color
            colors = ['salmon', 'lightgreen']
            for patch_head, patch_bone, color in zip(box_head['boxes'], box_bone['boxes'], colors):
                patch_head.set_facecolor(color)
                patch_bone.set_facecolor(color)
                patch_head.set_alpha(0.2)
                patch_bone.set_alpha(0.2)

            # Scatter Point
            for j, (values_head, values_bone) in enumerate(zip(data_head, data_bone)):
                ax_head.scatter(np.random.normal(j + 1, 0.04, size = len(values_head)), values_head, alpha = 0.7, color = colors[j])
                ax_bone.scatter(np.random.normal(j + 1, 0.04, size = len(values_bone)), values_bone, alpha = 0.7, color = colors[j])

            # Plot Format
            ax_head.set_xticks(range(1, len(categories) + 1))
            ax_bone.set_xticks(range(1, len(categories) + 1))
            ax_head.set_xticklabels(categories)
            ax_bone.set_xticklabels(categories)
            ax_head.set_xlabel(list_head[i])
            ax_bone.set_xlabel(list_bone[i])
        
        # Save Head Boxplot
        fig_head.tight_layout()
        fig_head.savefig(os.path.join(BP, 'Head_Metrics.png'), format = 'png', dpi = 300)
        plt.close(fig_head)

        # Save Bone Boxplot
        fig_bone.tight_layout()
        fig_bone.savefig(os.path.join(BP, 'Bone_Metrics.png'), format = 'png', dpi = 300)
        plt.close(fig_bone)

        return

    
"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    analysis = Analysis()
    analysis.main()