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
MR_RAW = ""
CT_RAW = ""

MR = ""
CT = ""
HM = ""
BR = ""
SK = ""
VS = ""

DATA_2D = ""

PATH_LIST = [MR, CT, HM, BR, SK, VS, DATA_2D]


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

        # Data_2D File Path
        for dataset in ['Train', 'Val', 'Test']:
            for data in ['MR', 'CT', 'HM']:
                path = os.path.join(os.path.join(DATA_2D, dataset, data))
                if not os.path.exists(path):
                    os.makedirs(path)

        # Get File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)

        if os.listdir(HM):
            self.hmasks = os.listdir(HM)
        if os.listdir(BR):
            self.brains = os.listdir(BR)
        if os.listdir(SK):
            self.skulls = os.listdir(SK)

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

        """
        ================================================================================================================
        File Format
        ================================================================================================================
        """
        # Change File Format
        self.mat2nii()

        """
        ================================================================================================================
        Background
        ================================================================================================================
        """
        # Interpolate + Rotate
        self.transform()
        # Remove Background
        self.background()
        # Process Intensity
        self.intensity()

        """
        ================================================================================================================
        Extract Brain Region + Remove Useless Region
        ================================================================================================================
        """
        # N4 Bias Correction
        self.n4bias()
        # Extract Brain Region
        self.strip()
        # Fill Holes in Brain Mask
        self.fillhole()
        # Remove Useless Area
        self.remove()

        """
        ================================================================================================================
        Extract Skull Region
        ================================================================================================================
        """
        # Extract Slull Region
        self.extract()

        """
        ================================================================================================================
        Normalize + Slice
        ================================================================================================================
        """
        # MR Normalize
        self.normalize()
        # Slice
        self.slice()
        # Slice with Specific Order
        self.specific()

        """
        ================================================================================================================
        Check Data Behavior
        ================================================================================================================
        """
        # Check Statistic
        self.statistic()
        # Check CT Behavior
        self.checkct()
        # Visulize Brain and Skull Extraction Result
        self.visualize()

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
    Interpolate + Rotate
    ====================================================================================================================
    """
    def transform(self) -> None:

        print()
        print('=======================================================================================================')
        print('Interpolate + Rotate')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # Numpy Array to Troch Tensor
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)

            # Trilinear Interpolation: (192, 192, 192)
            image = F.interpolate(image[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]
            label = F.interpolate(label[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]

            # Troch Tensor to Numpy Array
            image = image.numpy()
            label = label.numpy()

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
    Remove Background
    ====================================================================================================================
    """
    def background(self) -> None:

        print()
        print('=======================================================================================================')
        print('Remove Background')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # Flatten MR Data
            flat = image.flatten()

            # Sort in Ascending Order
            sorted = np.sort(flat)

            # Get Cumulative Distribution
            dis = np.cumsum(sorted)
            dis = dis / dis[-1]

            # Get Threshold
            if (i + 1) in self.artifacts:
                # Specific Case
                index = np.where(dis <= 0.200)[0][-1]
                value = sorted[index]
            else:
                # General Case
                index = np.where(dis <= 0.125)[0][-1]
                value = sorted[index]

            # Thresholding
            binary = (image > value)

            # Get Connective Component
            components, features = ndimage.label(binary)

            # Compute Size of Each Component
            sizes = ndimage.sum(binary, components, range(1, features + 1))

            # Find Largest Component
            largest = np.argmax(sizes) + 1

            # Slect Largest Component
            hmask = (components == largest)

            # Fill Holes in Mask
            hmask = ndimage.binary_dilation(hmask, np.ones((25, 25, 25)))
            hmask = ndimage.binary_erosion(hmask, np.ones((25, 25, 25)))

            # Apply Mask
            image = np.where(hmask, image, 0)
            label = np.where(hmask, label, -1000)
            hmask = np.where(hmask, 1, 0)

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i]))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i]))

            hmask = nib.Nifti1Image(hmask, np.eye(4))
            nib.save(hmask, os.path.join(HM, 'HM' + self.images[i][2:]))
        print()

        # Get New File Name
        self.hmasks = os.listdir(HM)

        return

    """
    ====================================================================================================================
    Process Intensity
    ====================================================================================================================
    """
    def intensity(self) -> None:

        print()
        print('=======================================================================================================')
        print('Process Intensity')
        print('=======================================================================================================')
        print()

        print('-------------------------------------------------------------------------------------------------------')
        print('Clip MR14 Intensity')
        print('-------------------------------------------------------------------------------------------------------')
        
        # Buffer for Maximum Value of MR
        mean = 0

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:
            
            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            
            # Summarize MR Maximum Value
            if (i + 1) != 14:
                mean += image.max()
        print()

        # Load MR14
        image = nib.load(os.path.join(MR, self.images[13])).get_fdata().astype('float32')

        # Mean Value of Maximum Value
        mean /= (self.len - 1)

        # Clip Intensity
        image = np.clip(image, 0, mean)

        # Save Data
        image = nib.Nifti1Image(image, np.eye(4))
        nib.save(image, os.path.join(MR, self.images[13]))

        print('-------------------------------------------------------------------------------------------------------')
        print('Clip CT Intensity + Deal With Extreme Case')
        print('-------------------------------------------------------------------------------------------------------')

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            
            # Deal With Extreme Case: Shift -1000
            if (i + 1) in self.highvalue:
                label -= 1000

            # Clip Intensity
            label = np.clip(label, -1000, 3000)

            # Save Data
            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i]))
        print()

        return
    
    """
    ====================================================================================================================
    N4 Bias Correction
    ====================================================================================================================
    """
    def n4bias(self) -> None:

        print()
        print('=======================================================================================================')
        print('N4 Bias Correction')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = sitk.ReadImage(os.path.join(MR, self.images[i]))
            image = sitk.Cast(image, sitk.sitkFloat32)

            # N4 Bias Correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            image = corrector.Execute(image)

            # Save Data
            sitk.WriteImage(image, os.path.join(MR, self.images[i]))
        print()

        return

    """
    ====================================================================================================================
    Extract Brain Region
    ====================================================================================================================
    """
    def strip(self) -> None:

        print()
        print('=======================================================================================================')
        print('Extract Brain Region')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = ants.image_read(os.path.join(MR, self.images[i]))

            # Brain Extraction
            brain = brain_extraction(image, modality = 't1')

            # Save Data
            brain.to_file(os.path.join(BR, 'BR' + self.images[i][2:]))
        print()

        # Get New File Name
        self.brains = os.listdir(BR)

        return
    
    """
    ====================================================================================================================
    Fill Holes in Brain Mask
    ====================================================================================================================
    """
    def fillhole(self) -> None:

        print()
        print('=======================================================================================================')
        print('Fill Holes in Brain Mask')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:
            
            # Load Data
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Apply Mask
            brain = np.where(hmask, brain, 0)
            brain = np.where(brain > 1e-6, 1, 0)

            # Filling Holes and Removing Small Component
            masks = []
            for j in range(192):

                # Process Exception Case
                if np.all(brain[:, :, j] == 0):
                    masks.append(np.zeros((192, 192)))
                    continue

                # Thresholding
                binary = (brain[:, :, j] > 0)

                # Get Connective Components
                components, features = ndimage.label(binary)

                # Compute Size of Each Component
                sizes = ndimage.sum(binary, components, range(1, features + 1))

                # Find Largest Component
                largest = np.argmax(sizes) + 1

                # Select Largest Component
                mask = (components == largest)

                # Fill Holes in Mask
                mask = ndimage.binary_dilation(mask, np.ones((15, 15)))
                mask = ndimage.binary_erosion(mask, np.ones((15, 15)))

                # Background
                masks.append(np.where(mask, 1, 0))

            # Stack + Transpose
            masks = np.stack(masks, axis = 0)
            masks = masks.transpose(1, 2, 0)

            # Save Brain Mask
            masks = nib.Nifti1Image(masks, np.eye(4))
            nib.save(masks, os.path.join(BR, self.brains[i]))
        print()

        return
    
    """
    ====================================================================================================================
    Remove Useless Area
    ====================================================================================================================
    """
    def remove(self) -> None:

        print()
        print('=======================================================================================================')
        print('Remove Useless Area')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')

            # Buffer for Cutting Point
            point = np.zeros((192), dtype = 'int')

            # Create Buffer List for Cutting Point
            for j in range(192):
                for k in range(192):

                    if brain[:, j, k].max() != 0:
                        point[j] = k
                        break

            # Remove Zero Value
            point[point == 0] = 9999
            min_point = point.min()
            min_index = point.argmin()

            # Find Appropriate Cutting Point
            for j in range(192):
                
                if j < min_index:
                    point[j] = min_point
                    continue

                if point[j] == 9999:
                    point[j] = point[j - 1]
                    continue

            # Remove Useless Area
            for j in range(192):
                    
                image[:, j, :point[j]] = 0.0
                label[:, j, :point[j]] = -1000.0
                hmask[:, j, :point[j]] = 0.0

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i]))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i]))

            hmask = nib.Nifti1Image(hmask, np.eye(4))
            nib.save(hmask, os.path.join(HM, self.hmasks[i]))
        print()

        return

    """
    ====================================================================================================================
    Extract Skull Region
    ====================================================================================================================
    """
    def extract(self) -> None:

        print()
        print('=======================================================================================================')
        print('Extract Skull Region')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:
            
            # Load Data
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')

            # Apply Erosion Mask
            brain = ndimage.binary_erosion(brain, np.ones((13, 13, 13)))
            label = np.where(brain, -1000, label)

            # Find Threshold
            value = (label[label != -1000].mean() * 1) + (label[label != -1000].std() * 0)

            # Thresholding
            binary = (label > value)

            # Get Connective Component
            components, features = ndimage.label(binary)

            # Compute Size of Each Component
            sizes = ndimage.sum(binary, components, range(1, features + 1))

            # Find Largest Component
            largest = np.argmax(sizes) + 1

            # Slect Largest Component
            smask = (components == largest)

            # Fill Holes in Mask
            smask = ndimage.binary_dilation(smask, np.ones((5, 5, 5)))
            smask = ndimage.binary_erosion(smask, np.ones((5, 5, 5)))

            # Apply Mask
            label = np.where(smask, label, -1000)

            # Save Data
            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(SK, 'SK' + self.labels[i][2:]))
        print()

        return

    """
    ====================================================================================================================
    Normalize MR
    ====================================================================================================================
    """
    def normalize(self) -> None:

        print()
        print('=======================================================================================================')
        print('Normalize MR')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')

            # Z-Score
            image -= image.mean()
            image /= image.std()

            # [0, 1]
            image -= image.min()
            image /= image.max()

            # [-1, 1]
            image = (image * 2) - 1

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i]))
        print()
        
        return

    """
    ====================================================================================================================
    Slice
    ====================================================================================================================
    """
    def slice(self) -> None:

        print()
        print('=======================================================================================================')
        print('Slice')
        print('=======================================================================================================')
        print()

        # Combine File Name List
        buffer = list(zip(self.images, self.labels, self.hmasks))

        # Random Shuffle Simultaneously
        random.shuffle(buffer)

        # Separate File Name List
        self.images, self.labels, self.hmasks = map(list, zip(*buffer))

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            if i < 20:
                dataset = 'Train'
            elif i < 24:
                dataset = 'Val'
            elif i < 26:
                dataset = 'Test'

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')

            # Find Blank Slice Index
            lower = -1
            upper = -1
            for j in range(192):
                
                # Ratio of Head Region to Whole Slice
                ratio = hmask[:, :, j].sum() / (192 * 192)

                # Lower Bound
                if (ratio > 0.075) and (lower == -1):
                    lower = j
                    continue
                # Upper Bound
                if (ratio < 0.075) and (lower != -1) and (upper == -1):
                    upper = j
                    break

            # Slice
            for j in range(lower + 3, upper - 3):
                
                # (192, 192, 7) and (192, 192, 1)
                mr = image[:, :, j - 3 : j + 3 + 1]
                ct = label[:, :, j : j + 1]
                hm = hmask[:, :, j : j + 1]

                # Transpose (Z, X, Y) + Rotate
                mr = np.rot90(mr.transpose(2, 0, 1), k = 1, axes = (1, 2))
                ct = np.rot90(ct.transpose(2, 0, 1), k = 1, axes = (1, 2))
                hm = np.rot90(hm.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                mr = nib.Nifti1Image(mr, np.eye(4))
                nib.save(mr, os.path.join(DATA_2D, dataset, 'MR', self.images[i][:-4] + '_' + str(j) + '.nii'))

                ct = nib.Nifti1Image(ct, np.eye(4))
                nib.save(ct, os.path.join(DATA_2D, dataset, 'CT', self.labels[i][:-4] + '_' + str(j) + '.nii'))
                
                hm = nib.Nifti1Image(hm, np.eye(4))
                nib.save(hm, os.path.join(DATA_2D, dataset, 'HM', self.hmasks[i][:-4] + '_' + str(j) + '.nii'))
        print()

        # Check Training, Validation, Testing Set
        print('-------------------------------------------------------------------------------------------------------')
        print('Train')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.images[:20]]))
        print()
        print('-------------------------------------------------------------------------------------------------------')
        print('Val')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.images[20:24]]))
        print()
        print('-------------------------------------------------------------------------------------------------------')
        print('Test')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.images[24:]]))
        print()

        # Save Slicing Information
        with open(os.path.join(DATA_2D, 'Slice.txt'), 'w') as f:
            print('Train:', '\t', *sorted([file[2:4] for file in self.images[:20]]), file = f)
            print('Val:', '\t', *sorted([file[2:4] for file in self.images[20:24]]), file = f)
            print('Test:', '\t', *sorted([file[2:4] for file in self.images[24:]]), file = f)

        # Ascending Sort File Name List
        self.images.sort()
        self.labels.sort()
        self.hmasks.sort()

        return

    """
    ====================================================================================================================
    Slice with Specific Order
    ====================================================================================================================
    """ 
    def specific(self) -> None:

        print()
        print('=======================================================================================================')
        print('Slice with Specific Order')
        print('=======================================================================================================')
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

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            if i < 20:
                dataset = 'Train'
            elif i < 24:
                dataset = 'Val'
            elif i < 26:
                dataset = 'Test'

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')

            # Find Blank Slice Index
            lower = -1
            upper = -1
            for j in range(192):
                
                # Ratio of Head Region to Whole Slice
                ratio = hmask[:, :, j].sum() / (192 * 192)

                # Lower Bound
                if (ratio > 0.075) and (lower == -1):
                    lower = j
                    continue
                # Upper Bound
                if (ratio < 0.075) and (lower != -1) and (upper == -1):
                    upper = j
                    break

            # Slice
            for j in range(lower + 3, upper - 3):
                
                # (192, 192, 7) and (192, 192, 1)
                mr = image[:, :, j - 3 : j + 3 + 1]
                ct = label[:, :, j : j + 1]
                hm = hmask[:, :, j : j + 1]

                # Transpose (Z, X, Y) + Rotate
                mr = np.rot90(mr.transpose(2, 0, 1), k = 1, axes = (1, 2))
                ct = np.rot90(ct.transpose(2, 0, 1), k = 1, axes = (1, 2))
                hm = np.rot90(hm.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                mr = nib.Nifti1Image(mr, np.eye(4))
                nib.save(mr, os.path.join(DATA_2D, dataset, 'MR', self.images[i][:-4] + '_' + str(j) + '.nii'))

                ct = nib.Nifti1Image(ct, np.eye(4))
                nib.save(ct, os.path.join(DATA_2D, dataset, 'CT', self.labels[i][:-4] + '_' + str(j) + '.nii'))
                
                hm = nib.Nifti1Image(hm, np.eye(4))
                nib.save(hm, os.path.join(DATA_2D, dataset, 'HM', self.hmasks[i][:-4] + '_' + str(j) + '.nii'))
        print()

        # Check Training, Validation, Testing Set
        print('-------------------------------------------------------------------------------------------------------')
        print('Train')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.images[:20]]))
        print()
        print('-------------------------------------------------------------------------------------------------------')
        print('Val')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.images[20:24]]))
        print()
        print('-------------------------------------------------------------------------------------------------------')
        print('Test')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.images[24:]]))
        print()

        # Ascending Sort File Name List
        self.images.sort()
        self.labels.sort()
        self.hmasks.sort()

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
        title = "{: <20}{: <20}{: <20}{: <20}{: <20}"
        space = "{: <20}{: <20.2f}{: <20.2f}{: <20.2f}{: <20.2f}"

        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('File Name', 'Mean', 'STD', 'Min', 'Max'))
        print('-------------------------------------------------------------------------------------------------------')

        # List of Mean and STD of MR and CT
        mr_mean = []
        ct_mean = []
        mr_std = []
        ct_std = []
        for i in range(self.len):

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32').flatten()
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32').flatten()

            # Remove Air Region
            image = image[image > -1]
            label = label[label > -1000]
            
            # Save Mean and STD
            mr_mean.append(image.mean())
            ct_mean.append(label.mean())
            mr_std.append(image.std())
            ct_std.append(label.std())

            # Check Statistics
            print(space.format(self.images[i], image.mean(), image.std(), image.min(), image.max()))
            print(space.format(self.labels[i], label.mean(), label.std(), label.min(), label.max()))
            print()
        
        # Check Mean and STD Behavior
        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('', 'Mean Mean', 'Mean STD', 'STD Mean', 'STD STD'))
        print('-------------------------------------------------------------------------------------------------------')

        # Python List to Numpy Array
        mr_mean = np.array(mr_mean)
        ct_mean = np.array(ct_mean)
        mr_std = np.array(mr_std)
        ct_std = np.array(ct_std)

        # Check Mean and STD Behavior
        print(space.format('MR Mean & STD', mr_mean.mean(), mr_mean.std(), mr_std.mean(), mr_std.std()))
        print(space.format('CT Mean & STD', ct_mean.mean(), ct_mean.std(), ct_std.mean(), ct_std.std()))
        print()

        return
    
    """
    ====================================================================================================================
    Check CT Behavior
    ====================================================================================================================
    """
    def checkct(self) -> None:

        print()
        print('=======================================================================================================')
        print('Check CT Behavior')
        print('=======================================================================================================')
        print()

        # Output Format
        title = "{: <20}{: <40}{: <40}"
        space = "{: <20}{: <40.2f}{: <40.2f}"

        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('File Name', 'Mean Value of Non-Air Region', 'Soft Tissue Intensity'))
        print('-------------------------------------------------------------------------------------------------------')

        for i in range(self.len):

            # Load Data
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            tissue = label[96, 96, 144]

            label = label.flatten()
            label = label[label > -1000.0]

            # Check Statistics
            print(space.format(self.labels[i], label.mean(), tissue))
            print()
        print()

        return

    """
    ====================================================================================================================
    Visulize Brain and Skull Extraction Result
    ====================================================================================================================
    """
    def visualize(self) -> None:

        print()
        print('=======================================================================================================')
        print('Visulize Brain and Skull Extraction Result')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')

            # Binary Thresholding
            brain = np.where(brain > 0, 1, 0)
            skull = np.where(skull > -1000, 1, 0)

            # Overlap Original Data and Brain or Skull Region
            image = image + np.abs(brain * image.mean() * 5)
            label = label + np.abs(skull * label.mean() * 3)

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(VS, self.images[i]))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(VS, self.labels[i]))
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