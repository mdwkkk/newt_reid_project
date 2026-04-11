# ═══════════════════════════════════════════════════════════════
# ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ ПУТЕЙ
# ═══════════════════════════════════════════════════════════════

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
BASE_DIR = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ CUDA
# ═══════════════════════════════════════════════════════════════

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ═══════════════════════════════════════════════════════════════
# ИМПОРТЫ ИЗ ПРОЕКТА
# ═══════════════════════════════════════════════════════════════

from dataset import NewtDataset
from model import create_model  # ← ИЗМЕНЕНО: было AttentionUNet
from losses import CombinedLoss

# ═══════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_dice = 0
    
    def __call__(self, val_loss, val_dice):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_dice = val_dice
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️ Early Stopping! Лучший Dice: {self.best_dice:.4f}")
        else:
            self.best_loss = val_loss
            self.best_dice = val_dice
            self.counter = 0

# ═══════════════════════════════════════════════════════════════
# ЗАГРУЗКА ЧЕКПОИНТА
# ═══════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path, model, optimizer, device):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Чекпоинт не найден: {checkpoint_path}")
        return 0, model, optimizer
    
    print(f"📂 Загрузка чекпоинта: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"✅ Загружено с эпохи {start_epoch-1}")
    print(f"📊 Лучший Dice: {checkpoint.get('val_dice', 0):.4f}")
    
    return start_epoch, model, optimizer

# ═══════════════════════════════════════════════════════════════
# ФУНКЦИЯ ОБУЧЕНИЯ
# ═══════════════════════════════════════════════════════════════

def train(
    images_dir=os.path.join(BASE_DIR, 'data/images'),
    masks_dir=os.path.join(BASE_DIR, 'data/masks'),
    epochs=80,
    batch_size=4,
    learning_rate=0.0005,
    device='cuda',
    save_path=os.path.join(BASE_DIR, 'models/best_model.pth'),
    val_split=0.20
):
    torch.cuda.empty_cache()
    
    print(f"🚀 Устройство: {device}")
    print(f"📊 Эпох: {epochs}")
    print(f"📊 Batch Size: {batch_size}")
    print(f"📊 Validation: {val_split*100:.0f}%")
    print("=" * 60)
    
    full_dataset = NewtDataset(images_dir, masks_dir, augment=True, image_size=512)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # ═══════════════════════════════════════════════════════════
    # ← ИЗМЕНЕНО: Новая модель из segmentation_models.pytorch
    # ═══════════════════════════════════════════════════════════
    
    model = create_model(
        arch='unet',
        encoder='efficientnet-b3',
        pretrained=True
    ).to(device)
    
    print(f"🏗️ Модель: UNet + EfficientNet-B3")
    print(f"📚 Предобученные веса: ImageNet")
    
    # ═══════════════════════════════════════════════════════════
    
    criterion = CombinedLoss(
    dice_weight=1.0,
    boundary_weight=0.05)


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    best_dice = 0
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=15)
    
    # Проверка на возобновление
    checkpoint_path = save_path
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"⚠️ Найдена предыдущая модель: {checkpoint_path}")
        response = input("🔄 Возобновить обучение? (y/n): ")
        if response.lower() == 'y':
            start_epoch, model, optimizer = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            print(f"🚀 Возобновляем с эпохи {start_epoch}")
            best_dice = 0
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("=" * 60)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_dice = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            with torch.no_grad():
                pred = torch.sigmoid(outputs)
                intersection = (pred * masks).sum()
                dice = (2. * intersection + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
                epoch_train_dice += dice.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_dice = epoch_train_dice / len(train_loader)
        
        model.eval()
        epoch_val_loss = 0
        epoch_val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                pred = torch.sigmoid(outputs)
                intersection = (pred * masks).sum()
                dice = (2. * intersection + 1e-6) / (pred.sum() + masks.sum() + 1e-6)
                epoch_val_dice += dice.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Dice={avg_train_dice:.4f} | Val Loss={avg_val_loss:.4f}, Dice={avg_val_dice:.4f}")
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
            }, save_path)
            print(f"💾 Лучшая модель сохранена! Dice={avg_val_dice:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early Stopping
        early_stopping(avg_val_loss, avg_val_dice)
        if early_stopping.early_stop:
            break
    
    # График
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, 'results/training_history.png'), dpi=150)
    
    print("=" * 60)
    print(f"✅ Обучение завершено! Лучший Dice: {best_dice:.4f}")
    print(f"📈 График: results/training_history.png")
    
    return model, history

if __name__ == '__main__':
    train()