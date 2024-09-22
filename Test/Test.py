"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random

from tqdm import tqdm
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from scipy import io
from scipy import ndimage

import ants
from antspynet.utilities import brain_extraction

import torch
import torch.nn.functional as F


"""
========================================================================================================================
Global Constant
========================================================================================================================
"""
MR_RAW = "C:/Users/user/Desktop/Data/Data_Raw/MR"
CT_RAW = "C:/Users/user/Desktop/Data/Data_Raw/CT"

MR = "C:/Users/user/Desktop/Data/Test/Data/MR"
CT = "C:/Users/user/Desktop/Data/Test/Data/CT"

PATH_LIST = [MR, CT]


"""
========================================================================================================================
Preprocess
========================================================================================================================
"""
class Preprocess():

    """
    ====================================================================================================================
    Critical Parameters
    ====================================================================================================================
    """
    def __init__(self) -> None:

        # Check File Path
        for path in PATH_LIST:
            if not os.path.exists(path):
                os.makedirs(path)

        # Get File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)

        # Check File Number
        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        # Get File Number
        self.len = 26

        # Problem Case
        self.direction = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.artifacts = [5, 6, 7, 8, 9, 10]
        self.highvalue = [2, 5, 6, 7, 8, 9, 10]

        return

    """
    ====================================================================================================================
    Main Process
    ====================================================================================================================
    """
    def main(self) -> None:

        # # Change File Format
        # self.mat2nii()

        # Crop Image
        self.cropping()

        # Check Statistic
        self.statistic()

        return
    
    """
    ====================================================================================================================
    Change File Format
    ====================================================================================================================
    """
    def mat2nii(self) -> None:
        
        print()
        print('=======================================================================================================')
        print('Change File Format')
        print('=======================================================================================================')
        print()

        # Get Raw File Name
        self.images = os.listdir(MR_RAW)
        self.labels = os.listdir(CT_RAW)

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = io.loadmat(os.path.join(MR_RAW, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT_RAW, self.labels[i]))['CT'].astype('float32')

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i].strip('.mat') + '.nii'))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i].strip('.mat') + '.nii'))
        print()

        # Get New File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)

        return
    
    """
    ====================================================================================================================
    Crop Image
    ====================================================================================================================
    """
    def cropping(self) -> None:

        print()
        print('=======================================================================================================')
        print('Crop Image')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            if (image.shape[0] > 192 and image.shape[1] > 192 and image.shape[2] > 192):

                mid_x = image.shape[0] // 2
                mid_y = image.shape[1] // 2
                mid_z = image.shape[2] // 2

                image = image[mid_x - 96 : mid_x + 96, mid_y - 96 : mid_y + 96, mid_z - 96 : mid_z + 96]
                label = label[mid_x - 96 : mid_x + 96, mid_y - 96 : mid_y + 96, mid_z - 96 : mid_z + 96]

            # # Numpy Array to Troch Tensor
            # image = torch.from_numpy(image)
            # label = torch.from_numpy(label)

            # # Trilinear Interpolation: (192, 192, 192)
            # image = F.interpolate(image[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]
            # label = F.interpolate(label[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]

            # # Troch Tensor to Numpy Array
            # image = image.numpy()
            # label = label.numpy()

            # Rotate
            if (i + 1) not in self.direction:
                image = np.rot90(image, k = 3, axes = (0, 1))
                label = np.rot90(label, k = 3, axes = (0, 1))

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i]))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i]))
        print()
        
        return
    
    """
    ====================================================================================================================
    Check Statistic
    ====================================================================================================================
    """
    def statistic(self) -> None:

        print()
        print('=======================================================================================================')
        print('Check Statistic')
        print('=======================================================================================================')
        print()

        # Output Format
        title = "{:<40}"
        space = "{:<40}"

        # Title
        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('File Name', 'Size'))
        print('-------------------------------------------------------------------------------------------------------')

        for i in range(self.len):

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # Check Statistics
            print(space.format(self.images[i]), image.shape)
            print(space.format(self.labels[i]), label.shape)
            print()
    
        return


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    pre = Preprocess()
    pre.main()