# ═══════════════════════════════════════════════════════════════
# predict.py - Морфология на 512x512 (до ресайза!)
# ═══════════════════════════════════════════════════════════════

import cv2
import numpy as np
import torch
import os
from model import AttentionUNet

# ═══════════════════════════════════════════════════════════════
# ФИЛЬТРЫ
# ═══════════════════════════════════════════════════════════════

def keep_largest_contour(mask):
    """Оставляет только самый большой контур"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 100:  # Слишком маленький
        return np.zeros_like(mask)
    
    mask_filtered = np.zeros_like(mask)
    cv2.drawContours(mask_filtered, [largest], -1, 255, -1)
    
    return mask_filtered

def remove_head_region(mask, image_shape):
    """Мягко обрезает верхнюю часть"""
    h, w = image_shape[:2]
    mask_copy = mask.copy()
    
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask_copy
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area > 5000:
        x, y, wb, hb = cv2.boundingRect(largest)
        cut_height = int(hb * 0.15)
        cut_y = y + cut_height
        
        if hb > 100:
            mask_copy[y:cut_y, x:x+wb] = 0
    
    return mask_copy

# ═══════════════════════════════════════════════════════════════
# ПРЕДСКАЗАНИЕ
# ═══════════════════════════════════════════════════════════════

def predict(image_path, model_path='models/best_model.pth', device='cuda', threshold=0.55):
    print(f"📂 Загрузка модели...")
    model = AttentionUNet(n_channels=3, n_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Модель загружена")
    
    print(f"📷 Загрузка изображения: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    original = image.copy()
    print(f"📊 Размер: {image.shape}")
    
    # Ресайз для модели
    image_resized = cv2.resize(image, (512, 512))
    image_norm = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).unsqueeze(0).to(device)
    
    print(f"🔮 Предсказание...")
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    print(f"📊 Raw mask: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")
    
    # ═══════════════════════════════════════════════════════════
    # ПОРОГ + МОРФОЛОГИЯ НА 512x512 (ДО РЕСАЙЗА!)
    # ═══════════════════════════════════════════════════════════
    
    # Порог
    mask = (mask > threshold).astype(np.uint8) * 255
    pixels_before = cv2.countNonZero(mask)
    print(f"📊 После порога {threshold} (512x512): {pixels_before} пикселей")
    
    # Оставить только самый большой контур (на 512x512)
    mask = keep_largest_contour(mask)
    print(f"📊 После фильтра контуров (512x512): {cv2.countNonZero(mask)} пикселей")
    
    # Морфология НА 512x512 (маленькое ядро)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    print(f"📊 После морфологии (512x512): {cv2.countNonZero(mask)} пикселей")
    
    # ═══════════════════════════════════════════════════════════
    # РЕСАЙЗ НА ПОЛНЫЙ РАЗМЕР
    # ═══════════════════════════════════════════════════════════
    
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    pixels_after_resize = cv2.countNonZero(mask_resized)
    print(f"📊 После ресайза ({original.shape[1]}x{original.shape[0]}): {pixels_after_resize} пикселей")
    
    # ═══════════════════════════════════════════════════════════
    # УДАЛЕНИЕ ГОЛОВЫ
    # ═══════════════════════════════════════════════════════════
    
    mask_before_head = mask_resized.copy()
    mask_final = remove_head_region(mask_resized, original.shape)
    print(f"📊 После удаления головы: {cv2.countNonZero(mask_final)} пикселей")
    
    # Если удалили слишком много — вернуть оригинал
    if cv2.countNonZero(mask_final) < cv2.countNonZero(mask_before_head) * 0.5:
        print("⚠️ Фильтр головы удалил слишком много! Возвращаем оригинал...")
        mask_final = mask_before_head
    
    # Финальная проверка
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("❌ Тритон не найден!")
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"✅ Найдено! Площадь: {area} пикселей")
    
    if area < 500:
        print("⚠️ Площадь слишком маленькая!")
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    belly_roi = original[y:y+h, x:x+w]
    stretched = cv2.resize(belly_roi, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    overlay = original.copy()
    cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 3)
    
    return {
        'original': original,
        'mask': mask_final,
        'overlay': overlay,
        'belly_roi': belly_roi,
        'stretched': stretched,
        'bbox': (x, y, w, h),
        'dice_score': checkpoint.get('val_dice', 0)
    }

def save_results(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/original.png', results['original'])
    cv2.imwrite(f'{output_dir}/mask.png', results['mask'])
    cv2.imwrite(f'{output_dir}/overlay.png', results['overlay'])
    cv2.imwrite(f'{output_dir}/stretched.png', results['stretched'])
    print(f"✅ Результаты сохранены в {output_dir}/")

if __name__ == '__main__':
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'data/images/test.jpg'
    
    print("=" * 60)
    print("🦎 ПРЕДСКАЗАНИЕ")
    print("=" * 60)
    
    results = predict(image_path, threshold=0.55)
    
    if results:
        save_results(results)
        print(f"🎯 Dice Score: {results['dice_score']:.2%}")
        print("=" * 60)
    else:
        print("=" * 60)