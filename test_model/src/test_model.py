# ═══════════════════════════════════════════════════════════════
# test_model.py - Тестирование модели сегментации тритонов
# ═══════════════════════════════════════════════════════════════

import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# Пути
# ═══════════════════════════════════════════════════════════════

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
BASE_DIR = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)

from dataset import NewtDataset
from model import AttentionUNet

# ═══════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(BASE_DIR, 'models/best_model.pth')
TEST_DIR = os.path.join(BASE_DIR, 'data/images')
MASKS_DIR = os.path.join(BASE_DIR, 'data/masks')
RESULTS_DIR = os.path.join(BASE_DIR, 'results/test')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
MAX_IMAGES = 20  # Сколько фото тестировать (0 = все)

# ═══════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛИ
# ═══════════════════════════════════════════════════════════════

print("╔" + "═" * 60 + "╗")
print("║" + " " * 15 + " ТЕСТИРОВАНИЕ МОДЕЛИ" + " " * 20 + "║")
print("╚" + "═" * 60 + "╝")
print()

print("📂 Загрузка модели...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    print("   Сначала обучите модель: python src/train.py")
    sys.exit(1)

model = AttentionUNet(n_channels=3, n_classes=1)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"✅ Модель загружена!")
print(f"📊 Лучший Dice при обучении: {checkpoint.get('val_dice', 0):.4f}")
print(f"🚀 Устройство: {DEVICE}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# ПОДГОТОВКА ТЕСТОВЫХ ДАННЫХ
# ═══════════════════════════════════════════════════════════════

test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]

if MAX_IMAGES > 0:
    test_files = test_files[:MAX_IMAGES]

print(f"📷 Найдено фото: {len(test_files)}")
print(f"📁 Результаты: {RESULTS_DIR}")
print("=" * 60)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# ТЕСТИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════

results = []
failed = []

for filename in tqdm(test_files, desc="Тестирование"):
    img_path = os.path.join(TEST_DIR, filename)
    
    # Поиск маски
    mask_name = (filename.replace('.jpg', '_mask.png')
                        .replace('.jpeg', '_mask.png')
                        .replace('.png', '_mask.png')
                        .replace('.JPG', '_mask.png')
                        .replace('.PNG', '_mask.png'))
    mask_path = os.path.join(MASKS_DIR, mask_name)
    
    # Загрузка фото
    image = cv2.imread(img_path)
    if image is None:
        failed.append((filename, "Не удалось прочитать"))
        continue
    
    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Загрузка маски (если есть)
    has_mask = os.path.exists(mask_path)
    mask_gt = None
    if has_mask:
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_gt = cv2.resize(mask_gt, (512, 512))
        mask_gt = (mask_gt > 127).astype(np.float32)
    
    # Предсказание
    image_resized = cv2.resize(image_rgb, (512, 512))
    image_norm = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > THRESHOLD).astype(np.float32)
    
    # Расчёт Dice (если есть маска)
    dice_score = None
    if has_mask:
        intersection = (pred_mask * mask_gt).sum()
        dice_score = (2. * intersection + 1e-6) / (pred_mask.sum() + mask_gt.sum() + 1e-6)
        results.append(dice_score)
    
    # Постобработка маски
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask_uint8, (w, h))
    
    # Морфология (очистка шума)
    kernel = np.ones((5, 5), np.uint8)
    pred_mask_clean = cv2.morphologyEx(pred_mask_resized, cv2.MORPH_CLOSE, kernel)
    pred_mask_clean = cv2.morphologyEx(pred_mask_clean, cv2.MORPH_OPEN, kernel)
    
    # Поиск контура
    contours, _ = cv2.findContours(pred_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Overlay
    overlay = original.copy()
    bbox = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Минимальная площадь
            cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)
            bbox = cv2.boundingRect(largest_contour)
    
    # Сохранение результатов
    base_name = os.path.splitext(filename)[0]
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(image_rgb)
    plt.title('📷 Оригинал')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('🎭 Предсказание')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(pred_mask_clean, cmap='gray')
    plt.title('🧹 Очищенная')
    plt.axis('off')
    
    if has_mask:
        plt.subplot(1, 5, 4)
        plt.imshow(mask_gt, cmap='gray')
        plt.title('✅ Истинная маска')
        plt.axis('off')
    else:
        plt.subplot(1, 5, 4)
        plt.text(0.5, 0.5, 'Нет маски\nдля сравнения', 
                ha='center', va='center', fontsize=12, color='gray')
        plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    title = f'🎯 Overlay'
    if dice_score is not None:
        title += f'\n(Dice: {dice_score:.3f})'
    if bbox:
        title += f'\n{bbox[2]}x{bbox[3]} px'
    plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{base_name}_result.png'), dpi=150)
    plt.close()
    
    # Сохранение отдельных файлов
    cv2.imwrite(os.path.join(RESULTS_DIR, f'{base_name}_mask.png'), pred_mask_clean)
    cv2.imwrite(os.path.join(RESULTS_DIR, f'{base_name}_overlay.png'), overlay)

# ═══════════════════════════════════════════════════════════════
# СТАТИСТИКА
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("📊 СТАТИСТИКА ТЕСТИРОВАНИЯ")
print("=" * 60)

if results:
    results_np = np.array(results)
    
    print(f"📈 Протестировано фото: {len(results)}")
    print(f"❌ Ошибок: {len(failed)}")
    print()
    print(f"📊 Средний Dice: {results_np.mean():.4f}")
    print(f"📊 Медианный Dice: {np.median(results_np):.4f}")
    print(f"📊 Мин Dice: {results_np.min():.4f}")
    print(f"📊 Макс Dice: {results_np.max():.4f}")
    print(f"📊 Стандартное отклонение: {results_np.std():.4f}")
    print()
    
    # Распределение по диапазонам
    excellent = np.sum(results_np >= 0.80)
    good = np.sum((results_np >= 0.70) & (results_np < 0.80))
    normal = np.sum((results_np >= 0.60) & (results_np < 0.70))
    poor = np.sum(results_np < 0.60)
    
    print("📈 Распределение качества:")
    print(f"   🎉 Отлично (≥0.80): {excellent} ({excellent/len(results)*100:.1f}%)")
    print(f"   ✅ Хорошо (0.70-0.80): {good} ({good/len(results)*100:.1f}%)")
    print(f"   ⚠️ Нормально (0.60-0.70): {normal} ({normal/len(results)*100:.1f}%)")
    print(f"   ❌ Плохо (<0.60): {poor} ({poor/len(results)*100:.1f}%)")
    print()
    
    # Оценка качества
    avg_dice = results_np.mean()
    if avg_dice >= 0.80:
        print("🎉 ОТЛИЧНО! Модель готова к продакшену!")
    elif avg_dice >= 0.70:
        print("✅ ХОРОШО! Можно использовать с доработками.")
    elif avg_dice >= 0.60:
        print("⚠️ НОРМАЛЬНО! Нужно дообучить или добавить данных.")
    else:
        print("❌ ПЛОХО! Проверьте разметку или обучите заново.")
else:
    print("⚠️ Нет масок для сравнения! Проверка только визуальная.")

if failed:
    print()
    print("⚠️ Ошибки:")
    for filename, error in failed:
        print(f"   ❌ {filename}: {error}")

print("=" * 60)
print(f"📁 Результаты сохранены в: {RESULTS_DIR}/")
print()
print("🚀 Для запуска интерфейса: python src/app.py")
print("=" * 60)