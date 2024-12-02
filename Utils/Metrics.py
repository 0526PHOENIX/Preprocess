"""
========================================================================================================================
Package
========================================================================================================================
"""
import numpy as np
from numpy import ndarray

from scipy import ndimage

import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure


"""
========================================================================================================================
Metrics
========================================================================================================================
"""
class Metrics():

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, device: torch.device = None) -> None:
        
        # Device: CPU or GPU
        self.device = device

        return

    """
    ====================================================================================================================
    Get Head MAE, PSNR, SSIM
    ====================================================================================================================
    """
    def get_head(self, fake2_g: Tensor, real2_g: Tensor, hmask_g: Tensor) -> tuple[float, float, float]:
        
        # Remove Background
        fake2_g = torch.where(hmask_g, fake2_g, 0)
        real2_g = torch.where(hmask_g, real2_g, 0)

        # Head MAE
        mae = self.get_mae_region(fake2_g, real2_g, hmask_g)

        # Head PSNR
        psnr = self.get_psnr_region(fake2_g, real2_g, hmask_g)

        # Head SSIM
        ssim = self.get_ssim_region(fake2_g, real2_g, hmask_g, 'eva')

        return (mae, psnr, ssim)
    
    """
    ====================================================================================================================
    Get Bone MAE, PSNR, SSIM
    ====================================================================================================================
    """
    def get_bone(self, fake2_g: Tensor, skull_g: Tensor) -> tuple[float, float, float, float]:
        
        # Extract Skull
        fake2_g = torch.where(fake2_g > 150, fake2_g, -1000)

        # Skull Mask
        smask_g = torch.where((skull_g + fake2_g) > -2000, True, False)

        # Remove Background
        fake2_g = torch.where(smask_g, fake2_g, 0)
        skull_g = torch.where(smask_g, skull_g, 0)

        # Bone MAE
        mae = self.get_mae_region(fake2_g, skull_g, smask_g)

        # Bone PSNR
        psnr = self.get_psnr_region(fake2_g, skull_g, smask_g)

        # Bone SSIM
        ssim = self.get_ssim_region(fake2_g, skull_g, smask_g, 'eva')

        # Bone DICE
        dice = self.get_dice_global(fake2_g, skull_g)

        return (mae, psnr, ssim, dice)
    
    """
    ====================================================================================================================
    Evaluate Head MAE, PSNR, SSIM
    ====================================================================================================================
    """
    def eva_head(self, fake2_a: ndarray, real2_a: ndarray, hmask_a: ndarray) -> tuple[float, float, float]:

        # Transpose (Z, X, Y)
        fake2_a = fake2_a.transpose(2, 0, 1)
        real2_a = real2_a.transpose(2, 0, 1)
        hmask_a = hmask_a.transpose(2, 0, 1)

        # Numpy Array to Torch Tensor
        fake2_g = torch.from_numpy(fake2_a.copy()).to(self.device).unsqueeze(1)
        real2_g = torch.from_numpy(real2_a.copy()).to(self.device).unsqueeze(1)
        hmask_g = torch.from_numpy(hmask_a.copy()).to(self.device).unsqueeze(1)

        # Remove Background
        fake2_g = torch.where(hmask_g, fake2_g, 0)
        real2_g = torch.where(hmask_g, real2_g, 0)

        # Head MAE
        mae = self.get_mae_region(fake2_g, real2_g, hmask_g)

        # Head PSNR
        psnr = self.get_psnr_region(fake2_g, real2_g, hmask_g)

        # Head SSIM
        ssim = self.get_ssim_region(fake2_g, real2_g, hmask_g, 'infer')

        return (mae, psnr, ssim)
    
    """
    ====================================================================================================================
    Evaluate Bone MAE, PSNR, SSIM
    ====================================================================================================================
    """
    def eva_bone(self, fake2_a: ndarray, skull_a: ndarray, hmask_a: ndarray, brain_a: ndarray) -> tuple[float, float, float, float]:
        
        # Remove Background
        fake2_a = np.where(hmask_a, fake2_a, -1000)

        # Apply Erosion Brain Mask
        brain_a = ndimage.binary_erosion(brain_a, np.ones((13, 13, 13)))
        fake2_a = np.where(brain_a, -1000, fake2_a)

        # Find Threshold
        value = (fake2_a[fake2_a != -1000].mean() * 1) + (fake2_a[fake2_a != -1000].std() * 0)

        # Thresholding
        binary = (fake2_a > value)

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
        fake2_a = np.where(smask, fake2_a, -1000)

        # Transpose (Z, X, Y)
        fake2_a = fake2_a.transpose(2, 0, 1)
        skull_a = skull_a.transpose(2, 0, 1)

        # Numpy Array to Torch Tensor
        fake2_g = torch.from_numpy(fake2_a.copy()).to(self.device).unsqueeze(1)
        skull_g = torch.from_numpy(skull_a.copy()).to(self.device).unsqueeze(1)

        # Skull Mask
        smask_g = torch.where((skull_g + fake2_g) > -2000, True, False)

        # Remove Background
        fake2_g = torch.where(smask_g, fake2_g, 0)
        skull_g = torch.where(smask_g, skull_g, 0)

        # Bone MAE
        mae = self.get_mae_region(fake2_g, skull_g, smask_g)

        # Bone PSNR
        psnr = self.get_psnr_region(fake2_g, skull_g, smask_g)

        # Bone SSIM
        ssim = self.get_ssim_region(fake2_g, skull_g, smask_g, 'infer')

        # Bone DICE
        dice = self.get_dice_region(fake2_g, skull_g, smask_g)

        return (mae, psnr, ssim, dice)

    """
    ====================================================================================================================
    Get MAE
    ====================================================================================================================
    """
    def get_mae_global(self, fake2_g: Tensor, real2_g: Tensor) -> float:

        # MAE
        mae = torch.abs(fake2_g - real2_g).sum() / fake2_g.numel()

        return mae.item()

    """
    ====================================================================================================================
    Get MAE Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_mae_region(self, fake2_g: Tensor, real2_g: Tensor, mask_g: Tensor) -> float:

        # MAE Over Batch Dimension
        mae = torch.abs(fake2_g - real2_g).sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6)

        return mae.mean().item()

    """
    ====================================================================================================================
    Get PSNR
    ====================================================================================================================
    """
    def get_psnr_global(self, fake2_g: Tensor, real2_g: Tensor) -> float:
        
        # MSE
        mse = torch.square(fake2_g - real2_g).sum() / fake2_g.numel()

        # PSNR
        psnr = 10 * torch.log10(torch.square(real2_g.amax() - real2_g.amin()) / mse)

        return psnr.item()

    """
    ====================================================================================================================
    Get PSNR Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_psnr_region(self, fake2_g: Tensor, real2_g: Tensor, mask_g: Tensor) -> float:
        
        # MSE Over Batch Dimension
        mse = torch.square(fake2_g - real2_g).sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6)

        # Remove The Impact of Empty Slice
        mse = mse[mse != 0]

        # # PSNR Over Batch Dimension: Real Value
        # psnr = 10 * torch.log10(torch.square(real2_g.amax(dim = (1, 2, 3)) - real2_g.amin(dim = (1, 2, 3))) / (mse + 1e-6))

        # PSNR Over Batch Dimension: Theoretical Value
        psnr = 10 * torch.log10(4000 ** 2 / (mse + 1e-6))

        return psnr.mean().item()

    """
    ====================================================================================================================
    Get SSIM
    ====================================================================================================================
    """
    def get_ssim_global(self, fake2_g: Tensor, real2_g: Tensor) -> float:

        # SSIM
        ssim = StructuralSimilarityIndexMeasure(kernel_size = 5).to(self.device)(fake2_g, real2_g)

        return ssim.item()

    """
    ====================================================================================================================
    Get SSIM Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_ssim_region(self, fake2_g: Tensor, real2_g: Tensor, mask_g: Tensor, mode: str) -> float:

        # SSIM Map
        _, ssim = StructuralSimilarityIndexMeasure(kernel_size = 5, return_full_image = True).to(self.device)(fake2_g, real2_g)

        if mode == 'infer':
            
            # Remove Background
            ssim *= mask_g
            
            # SSIM
            ssim = ssim.sum() / (mask_g.sum() + 1e-6)

        else:
            
            # Remove Background
            ssim *= mask_g
            
            # SSIM Over Batch Dimension
            ssim = ssim.sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6)

        return ssim.mean().item()

    """
    ====================================================================================================================
    Get DICE
    ====================================================================================================================
    """
    def get_dice_global(self, fake2_g: Tensor, skull_g: Tensor) -> float:
        
        # Binary Mask
        fake2_g = torch.where(fake2_g > -1000, 1, 0)
        skull_g = torch.where(skull_g > -1000, 1, 0)

        # Dice
        dice = 2. * (fake2_g * skull_g).sum() / (fake2_g.sum() + skull_g.sum() + 1e-6)

        return dice.item()

    """
    ====================================================================================================================
    Get DICE Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_dice_region(self, fake2_g: Tensor, skull_g: Tensor, mask_g: Tensor) -> float:
        
        # Binary Mask
        fake2_g = torch.where(fake2_g > -1000, 1, 0)
        skull_g = torch.where(skull_g > -1000, 1, 0)

        # Dice Buffer
        dice = torch.zeros(fake2_g.shape[0]).to(self.device)

        for i in range(fake2_g.shape[0]):

            # Check Extreme Case
            if mask_g[i, 0].sum().item() == 0:

                # Dice
                dice[i] = 2. * (fake2_g * skull_g).sum() / (fake2_g.sum() + skull_g.sum() + 1e-6)

            else:

                # Non-Zero Points' Index
                index_g = torch.nonzero(mask_g[i, 0])
                
                # Cornor Index
                min_y, min_x = index_g.amin(dim = 0)
                max_y, max_x = index_g.amax(dim = 0)

                # Crop Image
                fake2_g_c = fake2_g[i : i + 1, :, min_y : max_y + 1, min_x : max_x + 1]
                skull_g_c = skull_g[i : i + 1, :, min_y : max_y + 1, min_x : max_x + 1]

                # Dice
                dice[i] = 2. * (fake2_g_c * skull_g_c).sum() / (fake2_g_c.sum() + skull_g_c.sum() + 1e-6)

        return dice.mean().item()
    

"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    pass