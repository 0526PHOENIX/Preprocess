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
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':
    
    test = Test()
    test.main()