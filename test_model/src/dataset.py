# ═══════════════════════════════════════════════════════════════
# dataset.py - Загрузка данных для сегментации тритонов
# ═══════════════════════════════════════════════════════════════

import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

# ═══════════════════════════════════════════════════════════════
# Автоматическое определение путей
# ═══════════════════════════════════════════════════════════════

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
BASE_DIR = os.path.dirname(src_dir)


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class NewtDataset(Dataset):
    def __init__(self, images_dir=None, masks_dir=None, augment=False, image_size=512):
        """
        Датасет для сегментации брюха тритона.
        
        Args:
            images_dir: Путь к папке с изображениями
            masks_dir: Путь к папке с масками
            augment: Применять ли аугментацию
            image_size: Размер изображений (512x512)
        """
        # Пути по умолчанию
        if images_dir is None:
            images_dir = os.path.join(BASE_DIR, 'data/images')
        if masks_dir is None:
            masks_dir = os.path.join(BASE_DIR, 'data/masks')
        
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        
        # Список файлов
        self.files = [f for f in os.listdir(images_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
        
        # ═══════════════════════════════════════════════════════
        # АУГМЕНТАЦИЯ (ПРОСТАЯ И СТАБИЛЬНАЯ)
        # ═══════════════════════════════════════════════════════
        
        if augment:
            self.transform = A.Compose([
                # Геометрические трансформации
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                A.Affine(
                    translate_percent=0.2,
                    scale=(0.8, 1.2),
                    rotate=(-45, 45),
                    p=0.7
                ),
                
                # Цветовые трансформации
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        p=0.5
                    ),
                ], p=0.7),
                
                # Шум и размытие
                A.OneOf([
                    A.GaussNoise(std_range=(0.1, 0.3), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                ], p=0.5),
                
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Возвращает пару (изображение, маска) для индекса idx.
        """
        img_name = self.files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Поиск маски (несколько вариантов имени)
        mask_name = (img_name.replace('.jpg', '_mask.png')
                            .replace('.jpeg', '_mask.png')
                            .replace('.png', '_mask.png')
                            .replace('.JPG', '_mask.png')
                            .replace('.PNG', '_mask.png'))
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Если маска не найдена с _mask, пробуем то же имя
        if not os.path.exists(mask_path):
            mask_name = img_name
            mask_path = os.path.join(self.masks_dir, mask_name)
        
        # ═══════════════════════════════════════════════════════
        # Чтение изображений
        # ═══════════════════════════════════════════════════════
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # ═══════════════════════════════════════════════════════
        # Ресайз
        # ═══════════════════════════════════════════════════════
        
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), 
                           interpolation=cv2.INTER_NEAREST)
        
        # ═══════════════════════════════════════════════════════
        # Добавляем канал для маски (H, W) → (H, W, 1)
        # ═══════════════════════════════════════════════════════
        
        mask = np.expand_dims(mask, -1)
        
        # ═══════════════════════════════════════════════════════
        # Нормализация
        # ═══════════════════════════════════════════════════════
        
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # ═══════════════════════════════════════════════════════
        # Применение аугментаций
        # ═══════════════════════════════════════════════════════
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # ═══════════════════════════════════════════════════════
        # Конвертация в формат PyTorch (C, H, W)
        # ═══════════════════════════════════════════════════════
        
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return torch.from_numpy(image), torch.from_numpy(mask)