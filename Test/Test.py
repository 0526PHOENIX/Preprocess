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
Test
========================================================================================================================
"""
class Test():

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

        self.histogram_equalization()

        return

    """
    ====================================================================================================================
    
    ====================================================================================================================
    """
    def histogram_equalization(self, bins: int = 5000) -> None:
        
        print()
        print('=======================================================================================================')
        print('')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Remove Background + Flatten Image
            flat = image[hmask]
            
            # Histogram
            hist, edge = np.histogram(flat, bins = bins)

            # Cumulative Distribution Function
            cdf = np.cumsum(hist)

            # Normalize to [-1, 1]
            cdf = cdf / cdf[-1]
            cdf = cdf * 2 - 1
            
            # Map Intensity
            flat = np.interp(flat, edge[:-1], cdf)

            # Reconstruct
            equal = image.copy()
            equal[hmask] = flat
            
            # Save Data
            equal = nib.Nifti1Image(equal, np.eye(4))
            nib.save(equal, os.path.join("./Test", 'Test' + self.images[i][2:]))

        return


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    test = Test()
    test.main()
