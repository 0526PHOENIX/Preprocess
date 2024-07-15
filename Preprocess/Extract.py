"""
========================================================================================================================
Package
========================================================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
import numpy as np
import nibabel as nib

from scipy import ndimage


"""
========================================================================================================================
Global Constant
========================================================================================================================
"""
MR_RAW = "C:/Users/user/Desktop/Preprocess/Data_Raw/MR"
CT_RAW = "C:/Users/user/Desktop/Preprocess/Data_Raw/CT"

MR = "C:/Users/user/Desktop/Preprocess/Data/MR"
CT = "C:/Users/user/Desktop/Preprocess/Data/CT"
HM = "C:/Users/user/Desktop/Preprocess/Data/HM"
BR = "C:/Users/user/Desktop/Preprocess/Data/BR"
VS = "C:/Users/user/Desktop/Preprocess/Data/VS"
SK = "C:/Users/user/Desktop/Preprocess/Data/SK"

DATA_2D = "C:/Users/user/Desktop/Preprocess/Data_2D"

PATH_LIST = [MR, CT, HM, BR, VS, SK, DATA_2D]


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

        return

    """
    ====================================================================================================================
    Main Process
    ====================================================================================================================
    """
    def main(self) -> None:
        
        self.extract()

        # self.statistic()

        return

    """
    ====================================================================================================================
    Extract Skull
    ====================================================================================================================
    """
    def extract(self) -> None:

        print()
        print('=======================================================================================================')
        print('Extract Skull')
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
            value = label[label != -1000].mean() + (label[label != -1000].std() * 0.5)

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
            smask = ndimage.binary_dilation(smask, np.ones((11, 11, 11)))
            smask = ndimage.binary_erosion(smask, np.ones((11, 11, 11)))

            # Apply Mask
            label = np.where(smask, label, -1000)

            # Save Data
            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(SK, 'SK' + self.labels[i][2:]))
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

        print('-------------------------------------------------------------------------------------------------------')
        print("{:15}\t{:15}\t{:15}\t{:15}\t{:15}\t{:15}".format('File Name', 'Voxel', 'Mean', 'STD', 'Minimum', 'Maximum'))
        print('-------------------------------------------------------------------------------------------------------')

        for i in range(self.len):

            # Load Data
            skull = nib.load(os.path.join(SK, self.skulls[i])).get_fdata().astype('float32').flatten()

            # Extract Non-Air Region
            skull = skull[skull > -1000.0]

            # Check Statistics
            space = "{:15}\t{: <15}\t{: <15.2f}\t{: <15.2f}\t{: <15.2f}\t{: <15.2f}"
            print()
            print(space.format(self.skulls[i], len(skull), skull.mean(), skull.std(), skull.min(), skull.max()))
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