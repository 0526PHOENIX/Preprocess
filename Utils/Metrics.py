"""
========================================================================================================================
Package
========================================================================================================================
"""
from typing import Literal

import numpy as np
from numpy import ndarray
from scipy import ndimage

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import Conv2d, MaxPool2d
from torch.nn.functional import tanh
from torchmetrics.image import StructuralSimilarityIndexMeasure

from torchvision import models


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

        # Perceptual Loss
        self.get_per = PerceptualLoss(device = self.device)

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
        dice = self.get_dice(fake2_g, skull_g)

        return (mae, psnr, ssim, dice)
    
    """
    ====================================================================================================================
    Evaluate Head MAE, RMSE, PSNR, SSIM, LPIPS
    ====================================================================================================================
    """
    def eva_head(
                 self,
                 fake2_a: ndarray,
                 real2_a: ndarray,
                 hmask_a: ndarray,
                 mode: str | Literal['region', 'global']
                 ) -> tuple[float, float, float, float, float]:

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

        if mode == 'region':
            # Head MAE
            mae = self.get_mae_region(fake2_g, real2_g, hmask_g)
            # Head RMSE
            rmse = self.get_rmse_region(fake2_g, real2_g, hmask_g)
            # Head PSNR
            psnr = self.get_psnr_region(fake2_g, real2_g, hmask_g)
            # Head SSIM
            ssim = self.get_ssim_region(fake2_g, real2_g, hmask_g, 'infer')

        elif mode == 'global':
            # Head MAE
            mae = self.get_mae_global(fake2_g, real2_g)
            # Head RMSE
            rmse = self.get_rmse_global(fake2_g, real2_g)
            # Head PSNR
            psnr = self.get_psnr_global(fake2_g, real2_g)
            # Head SSIM
            ssim = self.get_ssim_global(fake2_g, real2_g)

        else:
            raise ValueError('Invalid Mode')

        # Head LPIPS
        lpips = self.get_lpips(fake2_g, real2_g)
        

        return (mae, rmse, psnr, ssim, lpips)
    
    """
    ====================================================================================================================
    Evaluate Bone MAE, RMSE, PSNR, SSIM, DICE
    ====================================================================================================================
    """
    def eva_bone(
                 self,
                 fake2_a: ndarray,
                 skull_a: ndarray,
                 hmask_a: ndarray,
                 brain_a: ndarray,
                 mode: str | Literal['region', 'global']
                 ) -> tuple[float, float, float, float, float]:
        
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

        if mode == 'region':
            # Bone MAE
            mae = self.get_mae_region(fake2_g, skull_g, smask_g)
            # Bone RMSE
            rmse = self.get_rmse_region(fake2_g, skull_g, smask_g)
            # Bone PSNR
            psnr = self.get_psnr_region(fake2_g, skull_g, smask_g)
            # Bone SSIM
            ssim = self.get_ssim_region(fake2_g, skull_g, smask_g, 'infer')

        elif mode == 'global':
            # Bone MAE
            mae = self.get_mae_global(fake2_g, skull_g)
            # Bone RMSE
            rmse = self.get_rmse_global(fake2_g, skull_g)
            # Bone PSNR
            psnr = self.get_psnr_global(fake2_g, skull_g)
            # Bone SSIM
            ssim = self.get_ssim_global(fake2_g, skull_g)

        else:
            raise ValueError('Invalid Mode')
        
        # Bone DICE
        dice = self.get_dice(fake2_g, skull_g)

        return (mae, rmse, psnr, ssim, dice)

    """
    ====================================================================================================================
    Get MAE 
    ====================================================================================================================
    """
    def get_mae_global(self, fake2_g: Tensor, real2_g: Tensor) -> float:

        # MAE
        mae = torch.abs(fake2_g - real2_g).sum() / fake2_g.numel()

        # Round
        mae = round(mae.item(), 3)

        return mae

    """
    ====================================================================================================================
    Get MAE Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_mae_region(self, fake2_g: Tensor, real2_g: Tensor, mask_g: Tensor) -> float:

        # MAE Over Batch Dimension
        mae = torch.abs(fake2_g - real2_g).sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6)

        # Round
        mae = round(mae.mean().item(), 3)

        return mae
    
    """
    ====================================================================================================================
    Get RMSE 
    ====================================================================================================================
    """
    def get_rmse_global(self, fake2_g: Tensor, real2_g: Tensor) -> float:

        # MAE
        rmse = torch.sqrt(torch.square(fake2_g - real2_g).sum() / fake2_g.numel())

        # Round
        rmse = round(rmse.item(), 3)

        return rmse

    """
    ====================================================================================================================
    Get RMSE Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_rmse_region(self, fake2_g: Tensor, real2_g: Tensor, mask_g: Tensor) -> float:

        # MAE Over Batch Dimension
        rmse = torch.sqrt(torch.square(fake2_g - real2_g).sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6))

        # Round
        rmse = round(rmse.mean().item(), 3)

        return rmse

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

        # Round
        psnr = round(psnr.item(), 3)

        return psnr

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

        # Round
        psnr = round(psnr.mean().item(), 3)

        return psnr

    """
    ====================================================================================================================
    Get SSIM
    ====================================================================================================================
    """
    def get_ssim_global(self, fake2_g: Tensor, real2_g: Tensor) -> float:

        # SSIM
        ssim = StructuralSimilarityIndexMeasure(kernel_size = 5).to(self.device)(fake2_g, real2_g)

        # Round
        ssim = round(ssim.item(), 3)

        return ssim

    """
    ====================================================================================================================
    Get SSIM Within Specific Region: Head or Bone
    ====================================================================================================================
    """
    def get_ssim_region(self, fake2_g: Tensor, real2_g: Tensor, mask_g: Tensor, mode: str | Literal['infer', 'eva']) -> float:

        # SSIM Map
        _, ssim = StructuralSimilarityIndexMeasure(kernel_size = 5, return_full_image = True).to(self.device)(fake2_g, real2_g)

        if mode == 'infer':
            # Remove Background
            ssim *= mask_g
            # SSIM
            ssim = ssim.sum() / (mask_g.sum() + 1e-6)

        elif mode == 'eva':
            # Remove Background
            ssim *= mask_g
            # SSIM Over Batch Dimension
            ssim = ssim.sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6)

        else: 
            raise ValueError('Invalid Mode')

        # Round
        ssim = round(ssim.mean().item(), 3)

        return ssim

    """
    ====================================================================================================================
    Get DICE
    ====================================================================================================================
    """
    def get_dice(self, fake2_g: Tensor, skull_g: Tensor) -> float:
        
        # Binary Mask
        fake2_g = torch.where(fake2_g > 150, 1, 0)
        skull_g = torch.where(skull_g > 150, 1, 0)

        # Dice
        dice = 2. * (fake2_g * skull_g).sum() / (fake2_g.sum() + skull_g.sum() + 1e-6)

        # Round
        dice = round(dice.mean().item(), 3)

        return dice
    
    """
    ====================================================================================================================
    Get LPIPS
    ====================================================================================================================
    """
    def get_lpips(self, fake2_g: Tensor, real2_g: Tensor) -> float:
        
        # Extent to RGB Form
        fake2_g = fake2_g.repeat(1, 3, 1, 1)
        real2_g = real2_g.repeat(1, 3, 1, 1)

        # LPIPS
        lpips = self.get_per(fake2_g, real2_g)

        # Round
        lpips = round(lpips.mean().item(), 3)   

        return lpips
    

"""
========================================================================================================================
Hook-Based Perceptual Loss
========================================================================================================================
"""
class PerceptualLoss(Module):

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, device: torch.device = None) -> None:

        super().__init__()

        # Device: CPU or GPU
        self.device = device

        # Pretrained VGG19
        model = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1).to(self.device).features
        
        # Ectract Specific Layer: [0, 5, 10, 19]
        self.layer = [0, 5, 10, 19]
        self.model = Sequential(*([temp for layer in self.layer for temp in (model[layer], MaxPool2d(2))][:-1]))

        # Freeze Model Parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Buffer for Perceptual Score
        self.score = []

        # Capture Perceptual Score from Target Layers
        self.hook_layer()

        return

    """
    ====================================================================================================================
    Capture Perceptual Score from Target Layers
    ====================================================================================================================
    """
    def hook_layer(self) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        Capture Perceptual Score from Forward Pass
        ----------------------------------------------------------------------------------------------------------------
        """
        def forward_hook(module: Module, feature_in: Tensor, feature_out: Tensor) -> None:
            
            # Append Perceptual Score to Specific Buffer
            self.score.append(tanh(feature_out.detach()))

            return

        """
        ----------------------------------------------------------------------------------------------------------------
        Register Hook
        ----------------------------------------------------------------------------------------------------------------
        """
        # Register Hook for Every Layer in Model
        for layer in self.model:
            if isinstance(layer, Conv2d):
                layer.register_forward_hook(forward_hook)

        return

    """
    ====================================================================================================================
    Forward Pass
    ====================================================================================================================
    """
    def forward(self, fake_g: Tensor, real_g: Tensor) -> Tensor:
        
        with torch.no_grad():
            
            # Forward Pass: Fake
            self.score.clear()
            self.model(fake_g)
            score_fake = [score for score in self.score]
        
            # Forward Pass: Real
            self.score.clear()
            self.model(real_g)
            score_real = [score for score in self.score]
            
            # Buffer for Perceptual Loss
            perceptual = torch.tensor(0.0, device = self.device)

            # Perceptual Loss
            for fake, real in zip(score_fake, score_real):

                # MAE
                perceptual += torch.abs(fake - real).mean()

            # Normalize to [0, 2]
            perceptual /= len(self.layer)

        return perceptual
    

"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    pass