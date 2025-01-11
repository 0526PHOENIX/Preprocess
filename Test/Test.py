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

MR = "C:/Users/user/Desktop/Data/Data/MR"
CT = "C:/Users/user/Desktop/Data/Data/CT"
HM = "C:/Users/user/Desktop/Data/Data/HM"
BR = "C:/Users/user/Desktop/Data/Data/BR"
SK = "C:/Users/user/Desktop/Data/Data/SK"

EQ = "C:/Users/user/Desktop/Data/Data/EQ"
PD = "C:/Users/user/Desktop/Data/Data/PD"

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
        print('=' * 110)
        print('Initialization')
        print('=' * 110)
        print()

        # Data_2D File Path
        for dataset in ['Train', 'Val', 'Test']:
            for data in ['MR', 'CT', 'HM', 'BR', 'SK', 'EQ']:
                path = os.path.join(os.path.join(DATA_2D, dataset, data))
                if not os.path.exists(path):
                    os.makedirs(path)

        if not os.path.exists(PD):
            os.makedirs(PD)

        # Get File Name
        self.images = os.listdir(MR)
        self.labels = os.listdir(CT)
        self.hmasks = os.listdir(HM)
        self.equals = os.listdir(EQ)
        self.padded = os.listdir(PD)

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

        # self.pad_mr_series()

        self.concat_mr('Train')

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
            equal = nib.Nifti1Image(equal, np.eye(4))
            nib.save(equal, os.path.join(EQ, 'EQ' + self.images[i][2:]))
        print()

        return
    
    """
    ====================================================================================================================
    
    ====================================================================================================================
    """
    def pad_mr_series(self) -> None:

        print()
        print('=' * 110)
        print('Padding MR')
        print('=' * 110)
        print()

        # Progress Bar
        progress = tqdm(range(self.len), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            image = nib.load(os.path.join(MR, self.images[i])).get_fdata().astype('float32')

            # Padding
            z_axis = max(200 - image.shape[2], 0)
            image = np.pad(image, ((0, 0), (0, 0), (z_axis // 2, z_axis - z_axis // 2)), mode = 'constant', constant_values = -1)

            # Transpose (Z, X, Y) + Rotate
            image = np.rot90(image.transpose(2, 0, 1), k = 1, axes = (1, 2))

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(PD, self.images[i]))
        print()

        return
    
    """
    ====================================================================================================================
    
    ====================================================================================================================
    """
    def concat_mr(self, mode: str = 'Test') -> None:

        print()
        print('=' * 110)
        print('Concat MR')
        print('=' * 110)
        print()

        # Save Path
        save_path = os.path.join(DATA_2D, mode, 'PD')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # MR Slice Path
        mrs_path = os.path.join(DATA_2D, mode, 'MR')

        # MR Slice Filename
        mrs = []
        idx = []
        for mrs_name in sorted(os.listdir(mrs_path)):
            mrs.append(mrs_name)
            idx.append(int(mrs_name[2 : 4]))

        # Progress Bar
        progress = tqdm(range(len(mrs)), bar_format = '{l_bar}{bar:40}{r_bar}')
        for i in progress:

            # Load Data
            mr = nib.load(os.path.join(DATA_2D, mode, 'MR', mrs[i])).get_fdata().astype('float32')[3:4, :, :]

            # Load Data
            image = nib.load(os.path.join(PD, self.padded[idx[i] - 1])).get_fdata().astype('float32')

            # Concat
            image = np.concatenate((mr, image), axis = 0)

            # Check Path
            save_path

            # Save Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(save_path, mrs[i]))
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