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

def unwrap_belly(image, contour, mask):
    # Фильтры
    masked_pre = cv2.bitwise_and(image, image, mask=mask)
    image_smoothed = cv2.medianBlur(masked_pre, 5)

    kernel_erode = np.ones((5, 5), np.uint8) 
    mask_eroded = cv2.erode(mask, kernel_erode, iterations=1)
    
    contours_eroded, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_eroded: return image 
    cnt = max(contours_eroded, key=cv2.contourArea).squeeze()
    if len(cnt.shape) < 2: return image
        
    step = max(1, len(cnt) // 200) 
    sub_cnt = cnt[::step]
    dists = np.linalg.norm(sub_cnt[:, None] - sub_cnt[None, :], axis=-1)
    sub_idx1, sub_idx2 = np.unravel_index(np.argmax(dists), dists.shape)
    idx1, idx2 = sub_idx1 * step, sub_idx2 * step
    if idx1 > idx2: idx1, idx2 = idx2, idx1
        
    edge1 = cnt[idx1:idx2]
    edge2 = np.concatenate((cnt[idx2:], cnt[:idx1]), axis=0)[::-1]

    def curve_length(c):
        if len(c) < 2: return 1
        return np.sum(np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1)))

    def resample_curve(c, N):
        l = np.sqrt(np.sum(np.diff(c, axis=0)**2, axis=1))
        l = np.insert(l, 0, 0)
        cum = np.cumsum(l)
        new_l = np.linspace(0, cum[-1], N)
        new_x = np.interp(new_l, cum, c[:, 0])
        new_y = np.interp(new_l, cum, c[:, 1])
        return np.column_stack((new_x, new_y))

    natural_length = int(max(curve_length(edge1), curve_length(edge2)))
    if natural_length < 10: return image

    edge1_resampled = resample_curve(edge1, natural_length)
    edge2_resampled = resample_curve(edge2, natural_length)

    # ПРОСТО БЕРЕМ ШИРИНУ, БЕЗ АНАТОМИЧЕСКИХ ПЕРЕВОРОТОВ
    widths = np.linalg.norm(edge1_resampled - edge2_resampled, axis=1)
    natural_width = int(np.mean(widths))
    if natural_width < 10: return image

    map_x = np.zeros((natural_length, natural_width), dtype=np.float32)
    map_y = np.zeros((natural_length, natural_width), dtype=np.float32)

    for i in range(natural_length):
        map_x[i, :] = np.linspace(edge1_resampled[i][0], edge2_resampled[i][0], natural_width)
        map_y[i, :] = np.linspace(edge1_resampled[i][1], edge2_resampled[i][1], natural_width)

    unwrapped = cv2.remap(image_smoothed, map_x, map_y, interpolation=cv2.INTER_CUBIC)
    unwrapped_final = cv2.bilateralFilter(unwrapped, 9, 75, 75)
    
    return cv2.resize(unwrapped_final, (512, 512), interpolation=cv2.INTER_LANCZOS4)

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
    
    # Вызываем нашу новую функцию выпрямления
    unwrapped_img = unwrap_belly(original, largest_contour, mask_final)
    
    overlay = original.copy()
    cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 3)
    
    return {
        'original': original,
        'mask': mask_final,
        'overlay': overlay,
        'unwrapped': unwrapped_img, # <-- Добавили новый ключ с идеальной разверткой
        'bbox': (x, y, w, h),
        'dice_score': checkpoint.get('val_dice', 0)
    }

def save_results(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/original.png', results['original'])
    cv2.imwrite(f'{output_dir}/mask.png', results['mask'])
    cv2.imwrite(f'{output_dir}/overlay.png', results['overlay'])
    cv2.imwrite(f'{output_dir}/unwrapped.png', results['unwrapped'])
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