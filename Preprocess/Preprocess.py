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

from typing import Literal
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

        print()
        print('=======================================================================================================')
        print('Initialization')
        print('=======================================================================================================')
        print()

        # Check File Path
        for path in PATH_LIST:
            if not os.path.exists(path):
                os.makedirs(path)

        # Data_2D File Path
        for dataset in ['Train', 'Val', 'Test']:
            for data in ['MR', 'CT', 'HM', 'BR', 'SK']:
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
        self.highvalue = [2, 5, 6, 7, 8, 9, 10]

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

        """
        ================================================================================================================
        Fundamental Image Process
        ================================================================================================================
        """
        # Convert File Format
        self.convert_format()
        # Apply Transformation
        self.apply_transformation(mode = 'interpolate')
        # Remove Background
        self.remove_background(otsu = False)
        # Clip Intensity
        self.clip_intensity()

        """
        ================================================================================================================
        Medical Image Process
        ================================================================================================================
        """
        # MR N4 Bias Correction
        self.correct_bias()
        # Extract Brain Region
        self.extract_brain()
        # Fill Holes in Brain Mask
        self.fill_hole()
        # Remove Useless Area
        self.remove_uselessness()
        # MR Intensity Normalize
        self.apply_normalization()
        # Extract Slull Region
        self.extract_skull()

        """
        ================================================================================================================
        Slice
        ================================================================================================================
        """
        # Slice with Random Order
        self.slice_random(threshold = 0.075)
        # Slice with Specific Order
        self.slice_ordered(threshold = 0.075)

        """
        ================================================================================================================
        Check Data Behavior
        ================================================================================================================
        """
        # Check Statistic
        self.compute_statistic()
        # Check CT Behavior
        self.check_ct()
        # Visulize Brain and Skull Extraction Result
        self.visualize_extraction()

        return
    
    """
    ====================================================================================================================
    Convert File Format
    ====================================================================================================================
    """
    def convert_format(self) -> None:
        
        print()
        print('=======================================================================================================')
        print('Convert File Format from .mat to .nii')
        print('=======================================================================================================')
        print()

        # Raw File Name
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

        # New File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)

        return

    """
    ====================================================================================================================
    Apply Transformation
    ====================================================================================================================
    """
    def apply_transformation(self, mode: str | Literal['interpolate', 'padding'] = 'interpolate') -> None:

        print()
        print('=======================================================================================================')
        print('Rotate + Shift Intensity + Interpolate or Padding')
        print('=======================================================================================================')
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # Rotate
            if (i + 1) not in self.direction:
                image = np.rot90(image, k = 3, axes = (0, 1))
                label = np.rot90(label, k = 3, axes = (0, 1))

            # Deal With CT Extreme Case
            if (i + 1) in self.highvalue:
                # Shift -1000
                label -= 1000

            if mode == 'interpolate':

                # Numpy Array to Troch Tensor
                image = torch.from_numpy(image.copy())
                label = torch.from_numpy(label.copy())

                # Get Z-Axis Size
                size_z = round(image.shape[2] * (256 / image.shape[0]))

                # Trilinear Interpolation: (256, 256, Original Z * Scale Factor)
                image = F.interpolate(image[None, None, ...], size = (256, 256, size_z), mode = 'trilinear')[0, 0, ...]
                label = F.interpolate(label[None, None, ...], size = (256, 256, size_z), mode = 'trilinear')[0, 0, ...]

                # Troch Tensor to Numpy Array
                image = image.numpy()
                label = label.numpy()

            else:

                # Calculate Number of Padding Pixel
                x_axis = max(256 - image.shape[0], 0)
                y_axis = max(256 - image.shape[1], 0)

                # Apply Padding
                image = np.pad(image,
                            ((x_axis // 2, x_axis - x_axis // 2), (y_axis // 2, y_axis - y_axis // 2), (0, 0)),
                            mode = 'constant',
                            constant_values = 0)

                label = np.pad(label,
                            ((x_axis // 2, x_axis - x_axis // 2), (y_axis // 2, y_axis - y_axis // 2), (0, 0)),
                            mode = 'constant',
                            constant_values = -1000)

                # Crop
                image = image[: 256, : 256, :]
                label = label[: 256, : 256, :]

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
    def remove_background(self, otsu: bool = False) -> None:

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

            # Remove Rough Background
            label = np.where(label > -250, label, -1000)

            # Flatten CT Data
            flat = label.flatten()

            # Sort in Ascending Order
            sorted = np.sort(flat)

            # Cumulative Distribution
            dis = np.cumsum(sorted)
            dis = dis / dis[-1]

            # Use Otsu's Algorithm
            if otsu:

                # Get Criteria
                criteria = []
                threshold_range = range(10, 100)
                for j in threshold_range:

                    # Get Threshold
                    index = np.where(dis <= j / 800)[0][-1]
                    value = sorted[index]

                    # Thresholding
                    binary = (label > value)

                    # Compute Weight
                    weight_1 = binary.sum() / label.size
                    weight_0 = 1 - weight_1

                    # Extrene Case
                    if weight_1 == 0 or weight_0 == 0:
                        criteria.append(np.inf)
                        continue

                    # Compute Variance
                    var_1 = label[binary == 1].var() if label[binary == 1].size > 0 else 0
                    var_0 = label[binary == 0].var() if label[binary == 0].size > 0 else 0

                    # Save Criteria to Buffer
                    criteria.append(weight_0 * var_0 + weight_1 * var_1)

                # Python List to Numpy Array
                criteria = np.array(criteria)

                # Get Best Threshold in All Criteria
                index = np.where(dis <= threshold_range[criteria.argmin()] / 400)[0][-1]
                value = sorted[index]

            else:

                # Get Threshold
                index = np.where(dis <= 0.025)[0][-1]
                value = sorted[index]

            # Thresholding
            binary = (label > value)

            # Get Connective Component
            components, features = ndimage.label(binary)

            # Compute Size of Each Component
            sizes = ndimage.sum(binary, components, range(1, features + 1))

            # Find Largest Component
            largest = np.argmax(sizes) + 1

            # Slect Largest Component
            hmask = (components == largest)

            # Head Mask Buffer
            mask = hmask.copy()

            # Closing Element Structure
            struct = int(hmask.shape[0] / 8 // 2) * 2 + 1
            while struct >= 3:

                # Fill Holes in Mask (Along Z-Axis)
                for j in range(hmask.shape[2]):
                    hmask[:, :, j] = ndimage.binary_closing(hmask[:, :, j], np.ones((struct, struct)))
                
                # Narrow Down Element Structure
                struct = int(struct / 3 // 2) * 2 + 1

                # Element-Wise Or Operation of Refined Mask with Original Mask
                hmask |= mask
                    
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
    Clip Intensity
    ====================================================================================================================
    """
    def clip_intensity(self) -> None:

        print()
        print('=======================================================================================================')
        print('Clip Intensity')
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
        print('Clip CT Intensity')
        print('-------------------------------------------------------------------------------------------------------')

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

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
    def correct_bias(self) -> None:

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
    def extract_brain(self) -> None:

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
    def fill_hole(self) -> None:

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

            # Filling Holes Along Z-Axis
            masks = []
            for k in range(brain.shape[2]):

                # Process Exception Case
                if np.all(brain[:, :, k] == 0):
                    masks.append(np.zeros((brain.shape[0], brain.shape[1])))
                    continue

                # Thresholding
                binary = (brain[:, :, k] > 0)

                # Get Connective Components
                components, features = ndimage.label(binary)

                # Compute Size of Each Component
                sizes = ndimage.sum(binary, components, range(1, features + 1))

                # Find Largest Component
                largest = np.argmax(sizes) + 1

                # Select Largest Component
                mask = (components == largest)

                # Fill Holes in Mask
                mask = ndimage.binary_closing(mask, np.ones((15, 15)))

                # Background
                masks.append(mask)

            # Stack + Transpose
            masks = np.stack(masks, axis = 0)
            masks = masks.transpose(1, 2, 0)

            # Apply Mask
            brain = np.where(masks, 1, 0)

            # Thresholding
            binary = (brain > 0)

            # Get Connective Component
            components, features = ndimage.label(binary)

            # Compute Size of Each Component
            sizes = ndimage.sum(binary, components, range(1, features + 1))

            # Find Largest Component
            largest = np.argmax(sizes) + 1

            # Slect Largest Component
            brain = (components == largest)

            # Apply Mask
            brain = np.where(brain, 1, 0)

            # Save Brain Mask
            brain = nib.Nifti1Image(brain, np.eye(4))
            nib.save(brain, os.path.join(BR, self.brains[i]))
        print()

        return
    
    """
    ====================================================================================================================
    Remove Useless Area
    ====================================================================================================================
    """
    def remove_uselessness(self) -> None:

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
            point = np.zeros((image.shape[1]), dtype = 'int')

            # Create Buffer List for Cutting Point
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):

                    if brain[:, j, k].max() != 0:
                        point[j] = k
                        break

            # Remove Zero Value
            point[point == 0] = 9999
            min_point = point.min()
            min_index = point.argmin()

            # Find Appropriate Cutting Point
            for j in range(image.shape[1]):
                
                if j < min_index:
                    point[j] = min_point
                    continue

                if point[j] == 9999:
                    point[j] = point[j - 1]
                    continue

            # Remove Useless Area
            for j in range(image.shape[1]):
                    
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
    MR Intensity Normalize
    ====================================================================================================================
    """
    def apply_normalization(self) -> None:

        print()
        print('=======================================================================================================')
        print('MR Intensity Normalize')
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
    Extract Skull Region
    ====================================================================================================================
    """
    def extract_skull(self) -> None:

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

            # Thresholding
            binary = (label > 250)

            # Get Connective Component
            components, features = ndimage.label(binary)

            # Compute Size of Each Component
            sizes = ndimage.sum(binary, components, range(1, features + 1))

            # Find Largest Component
            largest = np.argmax(sizes) + 1

            # Slect Largest Component
            skull = (components == largest)

            # Head Mask Buffer
            mask = skull.copy()

            # Closing Element Structure
            struct = int(skull.shape[0] / 20 // 2) * 2 + 1
            while struct >= 3:

                # Fill Holes in Mask (Along Z-Axis)
                for j in range(skull.shape[2]):
                    skull[:, :, j] = ndimage.binary_closing(skull[:, :, j], np.ones((struct, struct)))
                
                # Narrow Down Element Structure
                struct = int(struct / 3 // 2) * 2 + 1

                # Element-Wise Or Operation of Refined Mask with Original Mask
                skull |= mask

            # Apply Mask
            skull = np.where(skull, label, -1000)

            # Save Data
            skull = nib.Nifti1Image(skull, np.eye(4))
            nib.save(skull, os.path.join(SK, 'SK' + self.labels[i][2:]))
        print()

        # Get New File Name
        self.skulls = os.listdir(SK)

        return

    """
    ====================================================================================================================
    Slice with Random Seed
    ====================================================================================================================
    """
    def slice_random(self, threshold: float = 0.075) -> None:

        print()
        print('=======================================================================================================')
        print('Slice with Random Seed')
        print('=======================================================================================================')
        print()

        # Combine File Name List
        buffer = list(zip(self.images, self.labels, self.hmasks, self.brains, self.skulls))

        # Random Shuffle Simultaneously
        random.shuffle(buffer)

        # Separate File Name List
        self.images, self.labels, self.hmasks, self.brains, self.skulls = map(list, zip(*buffer))

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            if i < 20:
                dataset = 'Train'
            elif i < 24:
                dataset = 'Val'
            elif i < 26:
                dataset = 'Test'

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')

            # Find Blank Slice Index
            lower_overall = -1
            upper_overall = -1
            for k in range(hmask.shape[2]):
                
                # Ratio of Head Region to Whole Slice
                ratio = hmask[:, :, k].sum() / (hmask.shape[0] * hmask.shape[1])

                # Lower Bound
                if (ratio > 0) and (lower_overall == -1):
                    lower_overall = k
                    continue
                # Upper Bound
                if (ratio <= 0) and (lower_overall != -1) and (upper_overall == -1):
                    upper_overall = k
                    break

            # Extreme Case
            if upper_overall == -1:
                upper_overall = hmask.shape[2] - 1

            # Temporal Mask
            tmask = np.where(image > -0.99, 1, 0)
                
            # Find Blank Slice Index
            lower = -1
            upper = -1
            for k in range(tmask.shape[2]):
                
                # Ratio of Head Region to Whole Slice
                ratio = tmask[:, :, k].sum() / (tmask.shape[0] * tmask.shape[1])

                # Lower Bound
                if (ratio > threshold) and (lower == -1):
                    lower = k
                    continue
                # Upper Bound
                if (ratio <= threshold) and (lower != -1) and (upper == -1):
                    upper = k
                    break

            # Extreme Case
            if upper == -1:
                upper = tmask.shape[2] - 1

            # Slice
            for k in range(lower + 3, upper - 3):
                
                # (256, 256, 7) and (256, 256, 1)
                mr = image[:, :, k - 3 : k + 3 + 1]
                ct = label[:, :, k : k + 1]
                hm = hmask[:, :, k : k + 1]
                br = brain[:, :, k : k + 1]
                sk = skull[:, :, k : k + 1]

                # Transpose (Z, X, Y) + Rotate
                mr = np.rot90(mr.transpose(2, 0, 1), k = 1, axes = (1, 2))
                ct = np.rot90(ct.transpose(2, 0, 1), k = 1, axes = (1, 2))
                hm = np.rot90(hm.transpose(2, 0, 1), k = 1, axes = (1, 2))
                br = np.rot90(br.transpose(2, 0, 1), k = 1, axes = (1, 2))
                sk = np.rot90(sk.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                mr = nib.Nifti1Image(mr, np.eye(4))
                nib.save(mr, os.path.join(DATA_2D, dataset, 'MR', self.images[i][:-4] + '_' + str(k) + '.nii'))
                ct = nib.Nifti1Image(ct, np.eye(4))
                nib.save(ct, os.path.join(DATA_2D, dataset, 'CT', self.labels[i][:-4] + '_' + str(k) + '.nii'))
                hm = nib.Nifti1Image(hm, np.eye(4))
                nib.save(hm, os.path.join(DATA_2D, dataset, 'HM', self.hmasks[i][:-4] + '_' + str(k) + '.nii'))
                br = nib.Nifti1Image(br, np.eye(4))
                nib.save(br, os.path.join(DATA_2D, dataset, 'BR', self.brains[i][:-4] + '_' + str(k) + '.nii'))
                sk = nib.Nifti1Image(sk, np.eye(4))
                nib.save(sk, os.path.join(DATA_2D, dataset, 'SK', self.skulls[i][:-4] + '_' + str(k) + '.nii'))

            # Remove Redundant Area + Save Data
            image = nib.Nifti1Image(image[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i]))
            label = nib.Nifti1Image(label[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i]))
            hmask = nib.Nifti1Image(hmask[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(hmask, os.path.join(HM, self.hmasks[i]))
            brain = nib.Nifti1Image(brain[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(brain, os.path.join(BR, self.brains[i]))
            skull = nib.Nifti1Image(skull[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(skull, os.path.join(SK, self.skulls[i]))
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
        self.brains.sort()
        self.skulls.sort()

        return

    """
    ====================================================================================================================
    Slice with Specific Order + Remove Redundant Area
    ====================================================================================================================
    """ 
    def slice_ordered(self, threshold: float = 0.075) -> None:

        print()
        print('=======================================================================================================')
        print('Slice with Specific Order + Remove Redundant Area')
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
                    self.brains.append('BR' + str(num) + '.nii')
                    self.skulls.append('SK' + str(num) + '.nii')

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
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')
            
            # Find Blank Slice Index
            lower_overall = -1
            upper_overall = -1
            for k in range(hmask.shape[2]):
                
                # Ratio of Head Region to Whole Slice
                ratio = hmask[:, :, k].sum() / (hmask.shape[0] * hmask.shape[1])

                # Lower Bound
                if (ratio > 0) and (lower_overall == -1):
                    lower_overall = k
                    continue
                # Upper Bound
                if (ratio <= 0) and (lower_overall != -1) and (upper_overall == -1):
                    upper_overall = k
                    break

            # Extreme Case
            if upper_overall == -1:
                upper_overall = hmask.shape[2] - 1

            # Temporal Mask
            tmask = np.where(image > -0.99, 1, 0)
                
            # Find Blank Slice Index
            lower = -1
            upper = -1
            for k in range(tmask.shape[2]):
                
                # Ratio of Head Region to Whole Slice
                ratio = tmask[:, :, k].sum() / (tmask.shape[0] * tmask.shape[1])

                # Lower Bound
                if (ratio > threshold) and (lower == -1):
                    lower = k
                    continue
                # Upper Bound
                if (ratio <= threshold) and (lower != -1) and (upper == -1):
                    upper = k
                    break

            # Extreme Case
            if upper == -1:
                upper = tmask.shape[2] - 1

            # Slice
            for k in range(lower + 3, upper - 3):
                
                # (256, 256, 7) and (256, 256, 1)
                mr = image[:, :, k - 3 : k + 3 + 1]
                ct = label[:, :, k : k + 1]
                hm = hmask[:, :, k : k + 1]
                br = brain[:, :, k : k + 1]
                sk = skull[:, :, k : k + 1]

                # Transpose (Z, X, Y) + Rotate
                mr = np.rot90(mr.transpose(2, 0, 1), k = 1, axes = (1, 2))
                ct = np.rot90(ct.transpose(2, 0, 1), k = 1, axes = (1, 2))
                hm = np.rot90(hm.transpose(2, 0, 1), k = 1, axes = (1, 2))
                br = np.rot90(br.transpose(2, 0, 1), k = 1, axes = (1, 2))
                sk = np.rot90(sk.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                mr = nib.Nifti1Image(mr, np.eye(4))
                nib.save(mr, os.path.join(DATA_2D, dataset, 'MR', self.images[i][:-4] + '_' + str(k) + '.nii'))
                ct = nib.Nifti1Image(ct, np.eye(4))
                nib.save(ct, os.path.join(DATA_2D, dataset, 'CT', self.labels[i][:-4] + '_' + str(k) + '.nii'))
                hm = nib.Nifti1Image(hm, np.eye(4))
                nib.save(hm, os.path.join(DATA_2D, dataset, 'HM', self.hmasks[i][:-4] + '_' + str(k) + '.nii'))
                br = nib.Nifti1Image(br, np.eye(4))
                nib.save(br, os.path.join(DATA_2D, dataset, 'BR', self.brains[i][:-4] + '_' + str(k) + '.nii'))
                sk = nib.Nifti1Image(sk, np.eye(4))
                nib.save(sk, os.path.join(DATA_2D, dataset, 'SK', self.skulls[i][:-4] + '_' + str(k) + '.nii'))

            # Remove Redundant Area + Save Data
            image = nib.Nifti1Image(image[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(image, os.path.join(MR, self.images[i]))
            label = nib.Nifti1Image(label[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(label, os.path.join(CT, self.labels[i]))
            hmask = nib.Nifti1Image(hmask[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(hmask, os.path.join(HM, self.hmasks[i]))
            brain = nib.Nifti1Image(brain[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(brain, os.path.join(BR, self.brains[i]))
            skull = nib.Nifti1Image(skull[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(skull, os.path.join(SK, self.skulls[i]))
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
        self.brains.sort()
        self.skulls.sort()

        return

    """
    ====================================================================================================================
    Check Statistic
    ====================================================================================================================
    """
    def compute_statistic(self) -> None:

        print()
        print('=======================================================================================================')
        print('Check Statistic')
        print('=======================================================================================================')
        print()

        # Output Format
        title = "{:^20}|{:^20}|{:^20}|{:^20}|{:^20}"
        space = "{:^20}|{:^20.2f}|{:^20.2f}|{:^20.2f}|{:^20.2f}"

        # Title
        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('File Name', 'Mean', 'STD', 'Min', 'Max'))
        print('-------------------------------------------------------------------------------------------------------')

        # Statistic Buffer
        mr_mean = []
        ct_mean = []
        mr_std = []
        ct_std = []

        for i in range(self.len):

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32').flatten()
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32').flatten()
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool').flatten()

            # # Remove Air Region
            # image = np.where(hmask, image, 0)
            # label = np.where(hmask, label, -250)

            # image = image[image > 0]
            # label = label[label > -250]

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
    def check_ct(self) -> None:

        print()
        print('=======================================================================================================')
        print('Check CT Behavior')
        print('=======================================================================================================')
        print()

        # Output Format
        title = "{:^25}|{:^40}|{:^40}"
        space = "{:^25}|{:^40.2f}|{:^40.2f}"

        # Title
        print('-------------------------------------------------------------------------------------------------------')
        print(title.format('File Name', 'Mean Value of Non-Air Region', 'Soft Tissue Intensity'))
        print('-------------------------------------------------------------------------------------------------------')

        for i in range(self.len):

            # Load Data
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Soft Tissue Intensity
            tissue = label[label.shape[0] // 2, label.shape[1] // 2, label.shape[2] // 3 * 2]

            # Remove Air Region
            label = np.where(hmask, label, -250)
            label = label.flatten()
            label = label[label > -250]

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
    def visualize_extraction(self) -> None:

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