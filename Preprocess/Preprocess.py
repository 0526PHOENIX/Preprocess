"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import datetime
import random

from tqdm import tqdm

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from scipy import io
from scipy import ndimage

import ants
from antspynet.utilities import brain_extraction


"""
========================================================================================================================
Global Constant
========================================================================================================================
"""
MR_RAW = "C:/Users/user/Desktop/Data/Data_Raw/MR"
CT_RAW = "C:/Users/user/Desktop/Data/Data_Raw/CT"

MR = "C:/Users/user/Desktop/Data/Data/MR"
CT = "C:/Users/user/Desktop/Data/Data/CT"
HM = "C:/Users/user/Desktop/Data/Data/HM"
BR = "C:/Users/user/Desktop/Data/Data/BR"
SK = "C:/Users/user/Desktop/Data/Data/SK"
EQ = "C:/Users/user/Desktop/Data/Data/EQ"

DATA_2D = "C:/Users/user/Desktop/Data/Data_2D"


PATH_LIST = [MR, CT, HM, BR, SK, EQ, DATA_2D]

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
        print('=' * 110)
        print('Initialization')
        print('=' * 110)
        print()

        # Check File Path
        for path in PATH_LIST:
            if not os.path.exists(path):
                os.makedirs(path)

        # Data_2D File Path
        for dataset in ['Train', 'Val', 'Test']:
            for data in ['MR', 'CT', 'HM', 'BR', 'SK', 'EQ']:
                path = os.path.join(os.path.join(DATA_2D, dataset, data))
                if not os.path.exists(path):
                    os.makedirs(path)

        # Get File Name
        self.images = sorted(os.listdir(MR))
        self.labels = sorted(os.listdir(CT))

        if os.listdir(HM):
            self.hmasks = sorted(os.listdir(HM))
        if os.listdir(BR):
            self.brains = sorted(os.listdir(BR))
        if os.listdir(SK):
            self.skulls = sorted(os.listdir(SK))
        if os.listdir(EQ):
            self.equals = sorted(os.listdir(EQ))

        # Check File Number
        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        # Get File Number
        self.len = 26

        # Problem Case
        self.direction = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.highvalue = [2, 5, 6, 7, 8, 9, 10]

        # Log Timestamp
        print('Timestamp: ' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
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
        # # Convert File Format
        # self.convert_format()
        # # Apply Transformation
        # self.apply_transformation()
        # # Remove Background
        # self.remove_background(otsu = False)

        """
        ================================================================================================================
        Intensity Manipulation
        ================================================================================================================
        """
        # # MR N4 Bias Correction
        # self.correct_bias()
        # # Clip Intensity
        # self.clip_intensity()

        """
        ================================================================================================================
        Medical Image Process
        ================================================================================================================
        """
        # # Extract Brain Region
        # self.extract_brain()
        # # Fill Holes in Brain Mask
        # self.fill_hole()
        # # Remove Useless Area
        # self.remove_uselessness()
        # # Extract Slull Region
        # self.extract_skull()

        """
        ================================================================================================================
        Normalization
        ================================================================================================================
        """
        # # MR Intensity Normalize + Histogram Normalization
        # self.apply_normalization()

        """
        ================================================================================================================
        Slice
        ================================================================================================================
        """
        # # Slice with Random Order
        # self.slice_random(threshold = 0.075)
        # Slice with Specific Order
        self.slice_ordered(threshold = 0.075)

        return
    
    """
    ====================================================================================================================
    Convert File Format
    ====================================================================================================================
    """
    def convert_format(self) -> None:
        
        print()
        print('=' * 110)
        print('Convert File Format from .mat to .nii')
        print('=' * 110)
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
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR, self.images[i].strip('.mat') + '.nii'))
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(CT, self.labels[i].strip('.mat') + '.nii'))
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
    def apply_transformation(self) -> None:

        print()
        print('=' * 110)
        print('Rotate + Shift Intensity + Padding')
        print('=' * 110)
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

            # Number of Padding Pixel
            x_axis = max(256 - image.shape[0], 0)
            y_axis = max(256 - image.shape[1], 0)

            # Padding
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
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(CT, self.labels[i]))
        print()
        
        return 
    
    """
    ====================================================================================================================
    Remove Background
    ====================================================================================================================
    """
    def remove_background(self, otsu: bool = False) -> None:

        print()
        print('=' * 110)
        print('Remove Background')
        print('=' * 110)
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')

            # Remove Rough Background
            label = np.where(label > -250, label, -1000)

            # Sort in Ascending Order
            sorted = np.sort(label.flatten())

            # Cumulative Distribution
            cdf = np.cumsum(sorted) / np.sum(sorted)

            # Use Otsu's Algorithm
            if otsu:

                # Get Criteria
                criteria = []
                threshold_range = range(10, 100)
                for j in threshold_range:

                    # Get Threshold
                    index = np.searchsorted(cdf, j / 800)
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
                index = np.searchsorted(cdf, threshold_range[criteria.argmin()] / 400)
                value = sorted[index]

            else:

                # Get Threshold
                index = np.searchsorted(cdf, 0.025)
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
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(CT, self.labels[i]))
            nib.save(nib.Nifti1Image(hmask, np.eye(4)), os.path.join(HM, 'HM' + self.images[i][2:]))
        print()

        # Get New File Name
        self.hmasks = os.listdir(HM)

        return

    """
    ====================================================================================================================
    N4 Bias Correction
    ====================================================================================================================
    """
    def correct_bias(self) -> None:

        print()
        print('=' * 110)
        print('N4 Bias Correction')
        print('=' * 110)
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
    Clip Intensity
    ====================================================================================================================
    """
    def clip_intensity(self) -> None:

        print()
        print('=' * 110)
        print('Clip Intensity')
        print('=' * 110)
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:
            
            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Clip MR Intensity 0 ~ 99.5% Intensity
            image = np.clip(image, 0, np.percentile(image[hmask], 99.5))

            # Clip CT Intensity -1000 ~ 3000
            label = np.clip(label, -1000, 3000)

            # Save Data
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(CT, self.labels[i]))
        print()

        return
    
    """
    ====================================================================================================================
    Extract Brain Region
    ====================================================================================================================
    """
    def extract_brain(self) -> None:

        print()
        print('=' * 110)
        print('Extract Brain Region')
        print('=' * 110)
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
        print('=' * 110)
        print('Fill Holes in Brain Mask')
        print('=' * 110)
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
            nib.save(nib.Nifti1Image(brain, np.eye(4)), os.path.join(BR, self.brains[i]))
        print()

        return
    
    """
    ====================================================================================================================
    Remove Useless Area
    ====================================================================================================================
    """
    def remove_uselessness(self) -> None:

        print()
        print('=' * 110)
        print('Remove Useless Area')
        print('=' * 110)
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
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(CT, self.labels[i]))
            nib.save(nib.Nifti1Image(hmask, np.eye(4)), os.path.join(HM, self.hmasks[i]))
        print()

        return

    """
    ====================================================================================================================
    MR Intensity Normalize + Histogram Normalization
    ====================================================================================================================
    """
    def apply_normalization(self) -> None:

        print()
        print('=' * 110)
        print('MR Intensity Normalize + Histogram Normalization')
        print('=' * 110)
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('bool')

            # Foreground Z-Score
            fore = image[hmask]
            image -= fore.mean()
            image /= fore.std()

            # [0, 1]
            image -= image.min()
            image /= image.max()

            # [-1, 1]
            image = (image * 2) - 1

            # Remove Background + Flatten Image
            flat = image[hmask]
            
            # Histogram
            hist, edge = np.histogram(flat, bins = 5000)

            # Cumulative Distribution Function
            cdf = np.cumsum(hist)

            # [-1, 1]
            cdf = cdf / cdf[-1]
            cdf = cdf * 2 - 1
            
            # Map Intensity
            flat = np.interp(flat, edge[:-1], cdf)

            # Reconstruct
            equal = image.copy()
            equal[hmask] = flat
            
            # Save Data
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(equal, np.eye(4)), os.path.join(EQ, 'EQ' + self.images[i][2:]))
        print()
        
        return

    """
    ====================================================================================================================
    Extract Skull Region
    ====================================================================================================================
    """
    def extract_skull(self) -> None:

        print()
        print('=' * 110)
        print('Extract Skull Region')
        print('=' * 110)
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
            nib.save(nib.Nifti1Image(skull, np.eye(4)), os.path.join(SK, 'SK' + self.labels[i][2:]))
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
        print('=' * 110)
        print('Slice with Random Seed')
        print('=' * 110)
        print()

        # Combine File Name List
        buffer = list(zip(self.images, self.labels, self.hmasks, self.brains, self.skulls, self.equals))

        # Random Shuffle Simultaneously
        random.shuffle(buffer)

        # Separate File Name List
        self.images, self.labels, self.hmasks, self.brains, self.skulls, self.equals = map(list, zip(*buffer))

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            if i < 20:
                dataset = 'Train'
            elif i < 22:
                dataset = 'Val'
            elif i < 26:
                dataset = 'Test'

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')
            equal = nib.load(os.path.join(EQ, self.equals[i])).get_fdata().astype('float32')

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
                eq = equal[:, :, k - 3 : k + 3 + 1]

                # Transpose (Z, X, Y) + Rotate
                mr = np.rot90(mr.transpose(2, 0, 1), k = 1, axes = (1, 2))
                ct = np.rot90(ct.transpose(2, 0, 1), k = 1, axes = (1, 2))
                hm = np.rot90(hm.transpose(2, 0, 1), k = 1, axes = (1, 2))
                br = np.rot90(br.transpose(2, 0, 1), k = 1, axes = (1, 2))
                sk = np.rot90(sk.transpose(2, 0, 1), k = 1, axes = (1, 2))
                eq = np.rot90(eq.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                nib.save(nib.Nifti1Image(mr, np.eye(4)), os.path.join(DATA_2D, dataset, 'MR', self.images[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(ct, np.eye(4)), os.path.join(DATA_2D, dataset, 'CT', self.labels[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(hm, np.eye(4)), os.path.join(DATA_2D, dataset, 'HM', self.hmasks[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(br, np.eye(4)), os.path.join(DATA_2D, dataset, 'BR', self.brains[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(sk, np.eye(4)), os.path.join(DATA_2D, dataset, 'SK', self.skulls[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(eq, np.eye(4)), os.path.join(DATA_2D, dataset, 'EQ', self.equals[i][:-4] + '_' + str(k) + '.nii'))

            # Remove Redundant Area + Save Data
            nib.save(nib.Nifti1Image(image[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(label[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(CT, self.labels[i]))
            nib.save(nib.Nifti1Image(hmask[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(HM, self.hmasks[i]))
            nib.save(nib.Nifti1Image(brain[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(BR, self.brains[i]))
            nib.save(nib.Nifti1Image(skull[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(SK, self.skulls[i]))
            nib.save(nib.Nifti1Image(equal[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(EQ, self.equals[i]))
        print()

        # Check Training, Validation, Testing Set
        print('-' * 110)
        print('Train')
        print('-' * 110)
        print(*sorted([file[2:4] for file in self.images[:20]]))
        print()
        print('-' * 110)
        print('Val')
        print('-' * 110)
        print(*sorted([file[2:4] for file in self.images[20:24]]))
        print()
        print('-' * 110)
        print('Test')
        print('-' * 110)
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
        self.equals.sort()

        return

    """
    ====================================================================================================================
    Slice with Specific Order + Remove Redundant Area
    ====================================================================================================================
    """ 
    def slice_ordered(self, threshold: float = 0.075) -> None:

        print()
        print('=' * 110)
        print('Slice with Specific Order + Remove Redundant Area')
        print('=' * 110)
        print()

        # Clear File Name List
        self.images.clear()
        self.labels.clear()
        self.hmasks.clear()
        self.brains.clear()
        self.skulls.clear()
        self.equals.clear()

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
                    self.equals.append('EQ' + str(num) + '.nii')

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            if i < 20:
                dataset = 'Train'
            elif i < 22:
                dataset = 'Val'
            elif i < 26:
                dataset = 'Test'

            # Load Data and Backgrond
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT, self.labels[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')
            brain = nib.load(os.path.join(BR, self.brains[i])).get_fdata().astype('float32')
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')
            equal = nib.load(os.path.join(EQ, self.equals[i])).get_fdata().astype('float32')
            
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
                eq = equal[:, :, k - 3 : k + 3 + 1]

                # Transpose (Z, X, Y) + Rotate
                mr = np.rot90(mr.transpose(2, 0, 1), k = 1, axes = (1, 2))
                ct = np.rot90(ct.transpose(2, 0, 1), k = 1, axes = (1, 2))
                hm = np.rot90(hm.transpose(2, 0, 1), k = 1, axes = (1, 2))
                br = np.rot90(br.transpose(2, 0, 1), k = 1, axes = (1, 2))
                sk = np.rot90(sk.transpose(2, 0, 1), k = 1, axes = (1, 2))
                eq = np.rot90(eq.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                nib.save(nib.Nifti1Image(mr, np.eye(4)), os.path.join(DATA_2D, dataset, 'MR', self.images[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(ct, np.eye(4)), os.path.join(DATA_2D, dataset, 'CT', self.labels[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(hm, np.eye(4)), os.path.join(DATA_2D, dataset, 'HM', self.hmasks[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(br, np.eye(4)), os.path.join(DATA_2D, dataset, 'BR', self.brains[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(sk, np.eye(4)), os.path.join(DATA_2D, dataset, 'SK', self.skulls[i][:-4] + '_' + str(k) + '.nii'))
                nib.save(nib.Nifti1Image(eq, np.eye(4)), os.path.join(DATA_2D, dataset, 'EQ', self.equals[i][:-4] + '_' + str(k) + '.nii'))

            # Remove Redundant Area + Save Data
            nib.save(nib.Nifti1Image(image[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(MR, self.images[i]))
            nib.save(nib.Nifti1Image(label[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(CT, self.labels[i]))
            nib.save(nib.Nifti1Image(hmask[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(HM, self.hmasks[i]))
            nib.save(nib.Nifti1Image(brain[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(BR, self.brains[i]))
            nib.save(nib.Nifti1Image(skull[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(SK, self.skulls[i]))
            nib.save(nib.Nifti1Image(equal[:, :, lower_overall : upper_overall], np.eye(4)), os.path.join(EQ, self.equals[i]))
        print()

        # Check Training, Validation, Testing Set
        print('-' * 110)
        print('Train')
        print('-' * 110)
        print(*sorted([file[2:4] for file in self.images[:20]]))
        print()
        print('-' * 110)
        print('Val')
        print('-' * 110)
        print(*sorted([file[2:4] for file in self.images[20:22]]))
        print()
        print('-' * 110)
        print('Test')
        print('-' * 110)
        print(*sorted([file[2:4] for file in self.images[22:]]))
        print()

        # Ascending Sort File Name List
        self.images.sort()
        self.labels.sort()
        self.hmasks.sort()
        self.brains.sort()
        self.skulls.sort()
        self.equals.sort()

        return

    
"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    pre = Preprocess()
    pre.main()