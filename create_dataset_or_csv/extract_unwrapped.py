import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_unwrapped_images(source_dir, output_dir, output_csv):
    """
    Фильтрует датасет, оставляя только нормализованные (unwrapped) изображения.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Создаем базовую папку, если ее нет
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = []
    total_found = 0
    missing_ids = []
    
    print(f"🔍 Сканирование директории: {source_dir}...")
    
    # Идем по всем папкам с ID
    for label_dir in tqdm(list(source_path.iterdir()), desc="Обработка папок"):
        if not label_dir.is_dir():
            continue
            
        label = label_dir.name
        new_label_dir = output_path / label
        
        # Ищем все файлы с припиской __unwrapped
        # Используем glob для поиска любых расширений (jpg, png, jpeg)
        unwrapped_files = list(label_dir.glob('*__unwrapped.*'))
        
        if not unwrapped_files:
            missing_ids.append(label)
            continue
            
        # Создаем папку ID в новой директории только если нашли файлы
        new_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in unwrapped_files:
            new_img_path = new_label_dir / img_path.name
            
            # Копируем файл
            shutil.copy2(img_path, new_img_path)
            total_found += 1
            
            # Записываем в будущий CSV
            data.append({
                'image_path': str(new_img_path),
                'label': label
            })

    # Сохраняем новую таблицу
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print("\n" + "="*40)
        print("🎉 Экстракция успешно завершена!")
        print("="*40)
        print(f"📁 Найдено и скопировано изображений: {total_found}")
        print(f"📊 Новый CSV файл сохранен в: {output_csv}")
        
        if missing_ids:
            print(f"\n⚠️ Внимание: Для следующих ID не найдено unwrapped изображений ({len(missing_ids)} шт.):")
            print(", ".join(missing_ids))
    else:
        print("\n❌ Ошибка: В исходной директории не найдено ни одного unwrapped изображения.")

if __name__ == '__main__':
    # Укажи здесь папку, где сейчас лежат все вперемешку (сырые, кропы, unwrapped)
    # Например, если они лежат в старой галерее или общей папке с кропами:
    SOURCE_DIRECTORY = 'data/train_crops' # <-- ПОМЕНЯЙ НА СВОЙ ПУТЬ
    
    # Куда сохраняем чистую эталонную базу
    OUTPUT_DIRECTORY = 'data/train_unwrapped'
    OUTPUT_CSV = 'data/train_unwrapped.csv'
    
    extract_unwrapped_images(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, OUTPUT_CSV)