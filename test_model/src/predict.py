# ═══════════════════════════════════════════════════════════════
# predict.py - Сегментация + Ориентация (180° + 90°)
# ═══════════════════════════════════════════════════════════════

import cv2
import numpy as np
import torch
import os
import sys
from model import create_model

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

THRESHOLD = 0.55
MIN_AREA = 1000
MORPH_KERNEL = 3
CANONICAL_SIZE = (256, 256)
HEAD_CUT_RATIO = 0.15

# ═══════════════════════════════════════════════════════════════
# ПУТИ
# ═══════════════════════════════════════════════════════════════

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
BASE_DIR = os.path.dirname(src_dir)

images_dir = os.path.join(BASE_DIR, 'data/images')
models_dir = os.path.join(BASE_DIR, 'models')
results_dir = os.path.join(BASE_DIR, 'results')

# ═══════════════════════════════════════════════════════════════
# ФИЛЬТРЫ
# ═══════════════════════════════════════════════════════════════

def keep_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 100:
        return np.zeros_like(mask), None
    mask_filtered = np.zeros_like(mask)
    cv2.drawContours(mask_filtered, [largest], -1, 255, -1)
    return mask_filtered, largest

# ═══════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ ОРИЕНТАЦИИ ПО НОГАМ (180°)
# ═══════════════════════════════════════════════════════════════

def detect_legs(mask, contour):
    """Ищет ноги как выступы на контуре."""
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    
    legs = []
    for i in range(len(contour)):
        px, py = contour[i][0]
        distance_from_center = abs(px - center_x)
        
        if distance_from_center > w * 0.45:
            rel_y = (py - y) / h
            legs.append({
                'x': px, 'y': py, 'rel_y': rel_y,
                'side': 'left' if px < center_x else 'right'
            })
    
    return legs, (x, y, w, h)

def orient_by_legs(image, mask, contour):
    """Определяет ориентацию по ногам (180° поворот)."""
    h, w = image.shape[:2]
    legs, bbox = detect_legs(mask, contour)
    x, y, bw, bh = bbox
    
    print(f"🦵 Найдено ног: {len(legs)}")
    
    if len(legs) < 2:
        print("⚠️ Недостаточно ног → запасной метод (по хвосту)")
        return orient_by_tail(image, mask, contour)
    
    front_legs = [leg for leg in legs if leg['rel_y'] < 0.4]
    back_legs = [leg for leg in legs if leg['rel_y'] > 0.6]
    
    print(f"   Передние ноги (верх): {len(front_legs)}")
    print(f"   Задние ноги (низ): {len(back_legs)}")
    
    if len(front_legs) > len(back_legs):
        print("\n🔄 Передние ноги сверху → голова снизу → поворот на 180°...")
        image = cv2.rotate(image, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        rotation_angle = 180
        contour = contour.copy()
        contour[:, :, 0] = w - contour[:, :, 0]
        contour[:, :, 1] = h - contour[:, :, 1]
    elif len(back_legs) > len(front_legs):
        print("\n✅ Задние ноги снизу → голова сверху → без поворота")
        rotation_angle = 0
    else:
        print("\n⚠️ Ног поровну → запасной метод (по хвосту)...")
        return orient_by_tail(image, mask, contour)
    
    return image, mask, contour, rotation_angle

def orient_by_tail(image, mask, contour):
    """Запасной метод: определение по хвосту."""
    h, w = image.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    
    third_h = bh // 3
    top_area = cv2.countNonZero(mask[y:y+third_h, x:x+bw])
    bottom_area = cv2.countNonZero(mask[y+2*third_h:y+bh, x:x+bw])
    
    print(f"📊 Запасной метод: Верх={top_area}, Низ={bottom_area}")
    
    needs_rotation = top_area < bottom_area * 0.7
    
    if needs_rotation:
        print("🔄 Хвост сверху → поворот на 180°...")
        image = cv2.rotate(image, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        rotation_angle = 180
        contour = contour.copy()
        contour[:, :, 0] = w - contour[:, :, 0]
        contour[:, :, 1] = h - contour[:, :, 1]
    else:
        print("✅ Хвост снизу → без поворота")
        rotation_angle = 0
    
    return image, mask, contour, rotation_angle

# ═══════════════════════════════════════════════════════════════
# ПОВОРОТ НА 90° (ГОРИЗОНТАЛЬНЫЙ → ВЕРТИКАЛЬНЫЙ)
# ═══════════════════════════════════════════════════════════════

def make_vertical(belly_rgba):
    """
    Проверяет ориентацию вырезанного брюха.
    Если горизонтальное (ширина > высоты) → повернуть на 90°.
    """
    h, w = belly_rgba.shape[:2]
    
    print(f"\n📐 Проверка ориентации брюха...")
    print(f"   Размер: {w}x{h}")
    
    # Если ширина больше высоты → горизонтальное → повернуть на 90°
    if w > h:
        print(f"🔄 Брюхо горизонтальное ({w}x{h}) → поворот на 90°...")
        belly_rgba = cv2.rotate(belly_rgba, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(f"✅ Новый размер: {belly_rgba.shape[1]}x{belly_rgba.shape[0]}")
        rotation_90 = 90
    else:
        print(f"✅ Брюхо вертикальное ({w}x{h}) → без поворота")
        rotation_90 = 0
    
    return belly_rgba, rotation_90

# ═══════════════════════════════════════════════════════════════
# УДАЛЕНИЕ ГОЛОВЫ (СВЕРХУ)
# ═══════════════════════════════════════════════════════════════

def remove_head_region(mask, image_shape):
    h, w = image_shape[:2]
    mask_copy = mask.copy()
    
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_copy
    
    largest = max(contours, key=cv2.contourArea)
    x, y, wb, hb = cv2.boundingRect(largest)
    
    cut_height = int(hb * HEAD_CUT_RATIO)
    cut_y = y + cut_height
    
    if hb > 100:
        mask_copy[y:cut_y, x:x+wb] = 0
        print(f"✂️ Голова удалена (сверху {cut_height} пикселей)")
    
    return mask_copy

# ═══════════════════════════════════════════════════════════════
# ВЫРЕЗКА ПО КОНТУРУ
# ═══════════════════════════════════════════════════════════════

def extract_belly_exact(image, mask, contour):
    x, y, bw, bh = cv2.boundingRect(contour)
    belly_roi = image[y:y+bh, x:x+bw].copy()
    mask_roi = mask[y:y+bh, x:x+bw].copy()
    contour_shifted = contour - np.array([[x, y]])
    belly_rgba = cv2.cvtColor(belly_roi, cv2.COLOR_BGR2BGRA)
    alpha = np.zeros_like(mask_roi)
    cv2.drawContours(alpha, [contour_shifted], -1, 255, -1)
    belly_rgba[:, :, 3] = alpha
    return belly_rgba, (x, y, bw, bh), contour_shifted

# ═══════════════════════════════════════════════════════════════
# КАНОНИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

def canonicalize_belly(belly_rgba, target_size=(512, 512)):
    h, w = belly_rgba.shape[:2]
    tw, th = target_size
    alpha = belly_rgba[:, :, 3]
    
    center_x = []
    width_at_y = []
    for y in range(h):
        row = alpha[y, :]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            center_x.append(np.mean(indices))
            width_at_y.append(indices[-1] - indices[0] + 1)
        else:
            center_x.append(w / 2)
            width_at_y.append(0)
    center_x = np.array(center_x)
    width_at_y = np.array(width_at_y)
    
    canonical = np.zeros((th, tw, 4), dtype=np.uint8)
    for ty in range(th):
        sy = int((ty / th) * h)
        sy = min(sy, h - 1)
        if width_at_y[sy] > 0:
            cx = center_x[sy]
            sw = width_at_y[sy]
            if sw > 0:
                scale = tw / sw
                for tx in range(tw):
                    sx = int(cx - (tw / 2 - tx) / scale)
                    sx = max(0, min(sx, w - 1))
                    if alpha[sy, sx] > 0:
                        canonical[ty, tx] = belly_rgba[sy, sx]
                    else:
                        canonical[ty, tx, 3] = 0
    return canonical

# ═══════════════════════════════════════════════════════════════
# ПРЕДСКАЗАНИЕ
# ═══════════════════════════════════════════════════════════════

def predict(image_path, model_path='models/best_model.pth', device='cuda'):
    print(f"📂 Загрузка модели...")
    model = create_model(arch='unet', encoder='efficientnet-b3', pretrained=False)
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
    h_orig, w_orig = image.shape[:2]
    
    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    image_norm = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).unsqueeze(0).to(device)
    
    print(f"🔮 Предсказание...")
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    mask = (mask > THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    mask_final, contour = keep_largest_contour(mask_resized)
    if contour is None:
        print("❌ Тритон не найден!")
        return None
    
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        print(f"⚠️ Площадь слишком маленькая: {area}")
        return None
    
    print(f"✅ Найдено! Площадь: {area} пикселей")
    
    # ═══════════════════════════════════════════════════════════
    # ШАГ 1: Поворот 180° (голова сверху)
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("ШАГ 1: Ориентация головы (180°)")
    print("=" * 60)
    original, mask_final, contour, rotation_180 = orient_by_legs(original, mask_final, contour)
    
    # ═══════════════════════════════════════════════════════════
    # ШАГ 2: Удаление головы
    # ═══════════════════════════════════════════════════════════
    
    mask_before = mask_final.copy()
    mask_final = remove_head_region(mask_final, original.shape)
    
    if cv2.countNonZero(mask_final) < cv2.countNonZero(mask_before) * 0.5:
        print("⚠️ Удалено слишком много! Возвращаем оригинал...")
        mask_final = mask_before
    
    # ═══════════════════════════════════════════════════════════
    # ШАГ 3: Вырезка брюха
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("ШАГ 2: Вырезка брюха")
    print("=" * 60)
    belly_exact, bbox, contour_shifted = extract_belly_exact(original, mask_final, contour)
    print(f"✅ Вырезано: {belly_exact.shape}")
    
    # ═══════════════════════════════════════════════════════════
    # ШАГ 4: Поворот 90° (вертикальная ориентация)
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("ШАГ 3: Вертикальная ориентация (90°)")
    print("=" * 60)
    belly_exact, rotation_90 = make_vertical(belly_exact)
    
    # ═══════════════════════════════════════════════════════════
    # ШАГ 5: Канонизация
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("ШАГ 4: Канонизация")
    print("=" * 60)
    canonical = canonicalize_belly(belly_exact, CANONICAL_SIZE)
    
    overlay = original.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
    
    total_rotation = rotation_180 + rotation_90
    
    return {
        'original': original,
        'mask': mask_final,
        'overlay': overlay,
        'belly_exact': belly_exact,
        'canonical': canonical,
        'bbox': bbox,
        'contour': contour,
        'dice_score': checkpoint.get('val_dice', 0),
        'area': area,
        'rotation_180': rotation_180,
        'rotation_90': rotation_90,
        'total_rotation': total_rotation
    }

def save_results(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/original.png', results['original'])
    cv2.imwrite(f'{output_dir}/mask.png', results['mask'])
    cv2.imwrite(f'{output_dir}/overlay.png', results['overlay'])
    if results['belly_exact'] is not None:
        cv2.imwrite(f'{output_dir}/belly_exact.png', results['belly_exact'])
    if results['canonical'] is not None:
        cv2.imwrite(f'{output_dir}/canonical.png', results['canonical'])
    print(f"\n✅ Результаты сохранены в {output_dir}/")

# ═══════════════════════════════════════════════════════════════
# ВЫБОР ФОТО
# ═══════════════════════════════════════════════════════════════

def select_photo():
    print("╔" + "═" * 60 + "╗")
    print("║" + " " * 15 + "🦎 ВЫБОР ФОТО" + " " * 28 + "║")
    print("╚" + "═" * 60 + "╝")
    files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
    if not files:
        print(f"❌ Нет фото в папке: {images_dir}")
        return None
    print(f"📊 Найдено фото: {len(files)}\n")
    for i, filename in enumerate(files, 1):
        print(f"  {i:3}. {filename}")
    print()
    while True:
        try:
            choice = input(f"Выберите фото (1-{len(files)}) или путь: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return os.path.join(images_dir, files[idx])
            elif os.path.exists(choice):
                return choice
            elif os.path.exists(os.path.join(images_dir, choice)):
                return os.path.join(images_dir, choice)
            print("❌ Не найдено.")
        except KeyboardInterrupt:
            return None
        except Exception as e:
            print(f"❌ Ошибка: {e}")

# ═══════════════════════════════════════════════════════════════
# ГЛАВНАЯ
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🦎 СЕГМЕНТАЦИЯ БРЮХА ТРИТОНА")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Устройство: {device}\n")
    
    image_path = select_photo()
    if image_path is None:
        return
    
    model_path = os.path.join(models_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    results = predict(image_path, model_path=model_path, device=device)
    if results:
        output_dir = os.path.join(results_dir, 'predict')
        save_results(results, output_dir)
        print(f"\n📁 Файлы:")
        print(f"   - original.png   (ориентировано)")
        print(f"   - mask.png")
        print(f"   - overlay.png")
        print(f"   - belly_exact.png (вырезано)")
        print(f"   - canonical.png  ← ГОЛОВА СВЕРХУ, ВЕРТИКАЛЬНО!")
        
        if results.get('rotation_180', 0) == 180:
            print(f"\n🔄 Поворот 180°: Было выполнено")
        if results.get('rotation_90', 0) == 90:
            print(f"🔄 Поворот 90°: Было выполнено")
        
        print("=" * 60)
        if sys.platform == 'win32':
            os.startfile(output_dir)
    else:
        print("❌ Не удалось найти тритона")

if __name__ == '__main__':
    main()
