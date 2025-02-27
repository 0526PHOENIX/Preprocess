"""
========================================================================================================================
Package
========================================================================================================================
"""
import os

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

MR = ""
CT = ""
HM = ""
BR = ""
SK = ""
EQ = ""

MR_PD = "C:/Users/user/Desktop/Data/Data_Pad/MR"
CT_PD = "C:/Users/user/Desktop/Data/Data_Pad/CT"
HM_PD = "C:/Users/user/Desktop/Data/Data_Pad/HM"
BR_PD = "C:/Users/user/Desktop/Data/Data_Pad/BR"
SK_PD = "C:/Users/user/Desktop/Data/Data_Pad/SK"

PATH_LIST = [MR_PD, CT_PD, HM_PD, BR_PD, SK_PD]


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
        print('=' * 110)
        print('Initialization')
        print('=' * 110)
        print()

        # Check File Path
        for path in PATH_LIST:
            if not os.path.exists(path):
                os.makedirs(path)

        # Get File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)
        self.hmasks = os.listdir(HM)
        self.brains = os.listdir(BR)
        self.skulls = os.listdir(SK)
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

        # self.histogram_equalization()

        self.pad_series()

        return

    """
    ====================================================================================================================
    Histogram Equalization
    ====================================================================================================================
    """
    def histogram_equalization(self, bins: int = 5000) -> None:
        
        print()
        print('=' * 110)
        print('Histogram Equalization')
        print('=' * 110)
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

            # [-1, 1]
            cdf = cdf / cdf[-1]
            cdf = cdf * 2 - 1
            
            # Map Intensity
            flat = np.interp(flat, edge[:-1], cdf)

            # Reconstruct
            equal = image.copy()
            equal[hmask] = flat
            
            # Save Data
            nib.save(nib.Nifti1Image(equal, np.eye(4)), os.path.join(EQ, 'EQ' + self.images[i][2:]))
        print()

        return
    
    """
    ====================================================================================================================
    
    ====================================================================================================================
    """
    def pad_series(self) -> None:

        print()
        print('=' * 110)
        print('Padding')
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
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32')

            # Padding
            z_axis = max(200 - image.shape[2], 0)
            image = np.pad(image, ((0, 0), (0, 0), (z_axis // 2, z_axis - z_axis // 2)), mode = 'constant', constant_values = -1)
            label = np.pad(label, ((0, 0), (0, 0), (z_axis // 2, z_axis - z_axis // 2)), mode = 'constant', constant_values = -1000)
            hmask = np.pad(hmask, ((0, 0), (0, 0), (z_axis // 2, z_axis - z_axis // 2)), mode = 'constant', constant_values = 0)
            brain = np.pad(brain, ((0, 0), (0, 0), (z_axis // 2, z_axis - z_axis // 2)), mode = 'constant', constant_values = 0)
            skull = np.pad(skull, ((0, 0), (0, 0), (z_axis // 2, z_axis - z_axis // 2)), mode = 'constant', constant_values = -1000)

            # Transpose (Z, X, Y) + Rotate
            image = np.rot90(image.transpose(2, 0, 1), k = 1, axes = (1, 2))
            label = np.rot90(label.transpose(2, 0, 1), k = 1, axes = (1, 2))
            hmask = np.rot90(hmask.transpose(2, 0, 1), k = 1, axes = (1, 2))
            brain = np.rot90(brain.transpose(2, 0, 1), k = 1, axes = (1, 2))
            skull = np.rot90(skull.transpose(2, 0, 1), k = 1, axes = (1, 2))

            # Save Data
            nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(MR_PD, self.images[i]))
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(CT_PD, self.labels[i]))
            nib.save(nib.Nifti1Image(hmask, np.eye(4)), os.path.join(HM_PD, self.hmasks[i]))
            nib.save(nib.Nifti1Image(brain, np.eye(4)), os.path.join(BR_PD, self.brains[i]))
            nib.save(nib.Nifti1Image(skull, np.eye(4)), os.path.join(SK_PD, self.skulls[i]))
        print()

        return


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    test = Test()
    test.main()