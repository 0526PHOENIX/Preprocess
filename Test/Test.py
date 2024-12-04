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

EQ = "C:/Users/user/Desktop/Data/Data/EQ"

DATA_2D = "C:/Users/user/Desktop/Data/Data_2D"


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

        # Data_2D File Path
        for dataset in ['Train', 'Val', 'Test']:
            for data in ['MR', 'CT', 'HM', 'BR', 'SK', 'EQ']:
                path = os.path.join(os.path.join(DATA_2D, dataset, data))
                if not os.path.exists(path):
                    os.makedirs(path)

        # Get File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)
        self.hmasks = os.listdir(HM)
        self.equals = os.listdir(EQ)

        # Check File Number
        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        # Get File Number
        self.len = 26

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

        self.slice_ordered(threshold = 0.075)

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
            nib.save(equal, os.path.join(EQ, 'EQ' + self.images[i][2:]))

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
        self.equals.clear()
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
                    self.equals.append('EQ' + str(num) + '.nii')
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
            equal = nib.load(os.path.join(EQ, self.equals[i])).get_fdata().astype('float32')
            hmask = nib.load(os.path.join(HM, self.hmasks[i])).get_fdata().astype('float32')
            
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
                eq = equal[:, :, k - 3 : k + 3 + 1]

                # Transpose (Z, X, Y) + Rotate
                eq = np.rot90(eq.transpose(2, 0, 1), k = 1, axes = (1, 2))

                # Save Data
                eq = nib.Nifti1Image(eq, np.eye(4))
                nib.save(eq, os.path.join(DATA_2D, dataset, 'EQ', self.equals[i][:-4] + '_' + str(k) + '.nii'))

            # Remove Redundant Area + Save Data
            equal = nib.Nifti1Image(equal[:, :, lower_overall : upper_overall], np.eye(4))
            nib.save(equal, os.path.join(EQ, self.equals[i]))
        print()

        # Check Training, Validation, Testing Set
        print('-------------------------------------------------------------------------------------------------------')
        print('Train')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.equals[:20]]))
        print()
        print('-------------------------------------------------------------------------------------------------------')
        print('Val')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.equals[20:24]]))
        print()
        print('-------------------------------------------------------------------------------------------------------')
        print('Test')
        print('-------------------------------------------------------------------------------------------------------')
        print(*sorted([file[2:4] for file in self.equals[24:]]))
        print()

        # Ascending Sort File Name List
        self.images.sort()
        self.equals.sort()
        self.hmasks.sort()

        return


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    test = Test()
    test.main()