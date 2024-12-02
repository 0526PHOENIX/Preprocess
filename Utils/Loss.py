"""
========================================================================================================================
Package
========================================================================================================================
"""
import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Module, Sequential
from torchmetrics.image import StructuralSimilarityIndexMeasure

from torchvision import models


"""
========================================================================================================================
Loss Function
========================================================================================================================
"""
class Loss():

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
    Evaluate Head PIX, GDL, SIM, PER Loss
    ====================================================================================================================
    """
    def eva_head(self, fake2_a: ndarray, real2_a: ndarray) -> tuple[float, float, float, float]:

        # Transpose (Z, X, Y)
        fake2_a = fake2_a.transpose(2, 0, 1)
        real2_a = real2_a.transpose(2, 0, 1)

        # Numpy Array to Torch Tensor
        fake2_g = torch.from_numpy(fake2_a.copy()).to(self.device).unsqueeze(1)
        real2_g = torch.from_numpy(real2_a.copy()).to(self.device).unsqueeze(1)

        #
        mae = self.get_pix_loss(fake2_g, real2_g).item()

        #
        gdl = self.get_gdl_loss(fake2_g, real2_g).item()

        #
        sim = self.get_sim_loss(fake2_g, real2_g).item()

        # 
        per = self.get_per_loss(fake2_g, real2_g).item()

        return (mae, gdl, sim, per)

    """
    ====================================================================================================================
    Get Pixelwise Loss: MAE Loss
    ====================================================================================================================
    """
    def get_pix_loss(self, fake_g: Tensor, real_g: Tensor, mask_g: Tensor = None) -> Tensor:

        # Global
        if mask_g is None:

            # MAE
            mae = torch.abs(fake_g - real_g).sum() / fake_g.numel()

            return mae.mean()

        # Regional
        else:

            # Absolute
            dif = torch.abs(fake_g - real_g) * mask_g

            # MAE
            mae = dif.sum(dim = (1, 2, 3)) / mask_g.sum(dim = (1, 2, 3))

            return mae.mean()

    """
    ====================================================================================================================
    Get Gradient Difference Loss
    ====================================================================================================================
    """
    def get_gdl_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # First Derivative of Predicts
        grad_fake_x = torch.abs(fake_g[:, :, 1:, :] - fake_g[:, :, :-1, :])
        grad_fake_y = torch.abs(fake_g[:, :, :, 1:] - fake_g[:, :, :, :-1])

        # First Derivative of Labels
        grad_real_x = torch.abs(real_g[:, :, 1:, :] - real_g[:, :, :-1, :])
        grad_real_y = torch.abs(real_g[:, :, :, 1:] - real_g[:, :, :, :-1])

        # MSE
        gdl_x = torch.square(grad_fake_x - grad_real_x).sum() / grad_fake_x.numel()
        gdl_y = torch.square(grad_fake_y - grad_real_y).sum() / grad_fake_y.numel()

        return gdl_x.mean() + gdl_y.mean()

    """
    ====================================================================================================================
    Get Similarity Loss
    ====================================================================================================================
    """
    def get_sim_loss(self, fake_g: Tensor, real_g: Tensor, mask_g: Tensor = None) -> Tensor:

        # SSIM and SSIM Map
        ssim, ssim_map = StructuralSimilarityIndexMeasure(kernel_size = 5, return_full_image = True).to(self.device)(fake_g, real_g)

        # Global
        if mask_g is None:

            return 1 - ssim.mean()
        
        # Regional
        else:

            # Remove Background
            ssim_map = ssim_map * mask_g
        
            # SSIM Over Batch Dimension
            ssim = ssim_map.sum(dim = (1, 2, 3)) / (mask_g.sum(dim = (1, 2, 3)) + 1e-6)

            return 1 - ssim.mean()

    """
    ====================================================================================================================
    Get Perceptual Loss
    ====================================================================================================================
    """
    def get_per_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # Extent to RGB Form
        fake_g = fake_g.repeat(1, 3, 1, 1)
        real_g = real_g.repeat(1, 3, 1, 1)

        perceptual = self.get_per(fake_g, real_g)

        return perceptual.mean()
    
    """
    ====================================================================================================================
    Get Adversarial Loss: MSE Loss
    ====================================================================================================================
    """
    def get_adv_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # MSE 
        mse = torch.square(real_g - fake_g).sum() / fake_g.numel()

        return mse.mean()

    """
    ====================================================================================================================
    Get Cycle Consistency Loss
    ====================================================================================================================
    """
    def get_cyc_loss(self, fake_g: Tensor, real_g: Tensor) -> Tensor:

        # MAE
        mae = torch.abs(fake_g - real_g).sum() / fake_g.numel()

        return mae.mean()


"""
========================================================================================================================
Hook-Based Perceptual Loss
========================================================================================================================
"""
class PerceptualLoss(nn.Module):

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
        self.model = Sequential(*([temp for layer in self.layer for temp in (model[layer], nn.MaxPool2d(2))][:-1]))

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
            self.score.append(feature_out.detach())

            return

        """
        ----------------------------------------------------------------------------------------------------------------
        Register Hook
        ----------------------------------------------------------------------------------------------------------------
        """
        # Register Hook for Every Layer in Model
        for layer in self.model:
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

        return perceptual


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    real = torch.rand((1, 1, 256, 256)).to(torch.device('cuda:2'))
    fake = torch.rand((1, 1, 256, 256)).to(torch.device('cuda:2'))

    real = real.repeat(1, 3, 1, 1)
    fake = fake.repeat(1, 3, 1, 1)

    loss = PerceptualLoss(device = torch.device('cuda:2'))

    perceptual = loss(fake, real)

    print(perceptual)