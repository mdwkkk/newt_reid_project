import os
import pandas as pd
from pathlib import Path
from PIL import Image
from src.inference_utils import crop_belly

def sync_dataset_domains(source_dir, output_dir, output_csv):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    source_path = Path(source_dir)
    
    processed_unet = 0
    copied_original = 0
    
    print(f"🚀 Синхронизация доменов: пропускаем {source_dir} через U-Net...")
    
    # Идем по папкам с ID (1, 32, 56, 76 и т.д.)
    for label_dir in source_path.iterdir():
        if not label_dir.is_dir():
            continue
            
        label = label_dir.name
        new_label_dir = Path(output_dir) / label
        new_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Берем каждую картинку
        for img_path in label_dir.glob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            new_img_path = new_label_dir / img_path.name
            
            # 1. Пытаемся прогнать через U-Net
            cropped_img = crop_belly(str(img_path))
            
            if cropped_img is not None:
                # U-Net успешно применил свои трансформации
                cropped_img.save(new_img_path)
                processed_unet += 1
            else:
                # 2. Fallback: U-Net не понял картинку (т.к. она уже обрезана). 
                # Берем оригинал, но сохраняем через PIL для стандартизации формата
                img = Image.open(img_path).convert('RGB')
                img.save(new_img_path)
                copied_original += 1
                
            # Записываем в будущий CSV
            data.append({
                'image_path': str(new_img_path),
                'label': label
            })
            
    # Сохраняем новую таблицу
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"\n🎉 Готово! Создан датасет из {len(df)} изображений.")
        print(f"   - Обработано и изменено через U-Net: {processed_unet}")
        print(f"   - Оставлено как есть (Fallback): {copied_original}")
        print(f"📊 Новый файл разметки сохранен в {output_csv}")
    else:
        print("\n⚠️ Ошибка: Папка исходных данных пуста или структура не совпадает.")

if __name__ == '__main__':
    # Указываем папку с твоими готовыми обучающими кропами
    SOURCE_DIRECTORY = 'data/train_crops' 
    
    # Указываем, куда сохранить новую базу
    OUTPUT_DIRECTORY = 'data/train_crops_unet'
    OUTPUT_CSV = 'data/train_unet.csv'
    
    sync_dataset_domains(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, OUTPUT_CSV)