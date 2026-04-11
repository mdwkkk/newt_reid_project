import os
import cv2

images_dir = 'data/images'
masks_dir = 'data/masks'

print("🔍 Проверка масок...")
print("=" * 60)

ok = 0
errors = 0

for img_file in os.listdir(images_dir):
    if not img_file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG')):
        continue
    
    mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.png', '_mask.png').replace('.JPG', '_mask.png').replace('.PNG', '_mask.png')
    mask_path = os.path.join(masks_dir, mask_file)
    
    if not os.path.exists(mask_path):
        print(f"❌ НЕТ МАСКИ: {img_file}")
        errors += 1
        continue
    
    img = cv2.imread(os.path.join(images_dir, img_file))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        print(f"❌ ОШИБКА ЧТЕНИЯ: {img_file}")
        errors += 1
        continue
    
    if img.shape[:2] != mask.shape[:2]:
        print(f"⚠️ РАЗМЕРЫ НЕ СОВПАДАЮТ: {img_file}")
        errors += 1
        continue
    
    if mask.max() == 0:
        print(f"⚠️ ПУСТАЯ МАСКА: {mask_file}")
        errors += 1
        continue
    
    ok += 1
    print(f"✅ OK: {img_file}")

print("=" * 60)
print(f"📊 Итого: {ok} OK, {errors} ошибок")

if errors == 0:
    print("🎉 Все маски в порядке! Можно запускать обучение.")
else:
    print("⚠️ Исправьте ошибки перед обучением!")