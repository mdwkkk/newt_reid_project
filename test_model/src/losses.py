# ═══════════════════════════════════════════════════════════════
# losses.py - Dice + Boundary Loss
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# DICE LOSS
# ═══════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

# ═══════════════════════════════════════════════════════════════
# BOUNDARY LOSS (SOBEL) - ИСПРАВЛЕНО ДЛЯ GPU
# ═══════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Фильтры Собеля
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # ═══════════════════════════════════════════════════════
        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Перенос на устройство input
        # ═══════════════════════════════════════════════════════
        
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # Границы предсказания
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        # Границы истины
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        # MSE между границами
        boundary_loss = F.mse_loss(pred_edge, target_edge)
        
        return boundary_loss

# ═══════════════════════════════════════════════════════════════
# COMBINED LOSS (Dice + Boundary)
# ═══════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, boundary_weight=0.05):
        """
        Комбинированный loss.
        
        Args:
            dice_weight: Вес Dice Loss (1.0)
            boundary_weight: Вес Boundary Loss (0.01-0.1)
                            Начните с 0.05, если плохо → уменьшите до 0.02
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.dice_weight * dice) + (self.boundary_weight * boundary)
        
        return total_loss