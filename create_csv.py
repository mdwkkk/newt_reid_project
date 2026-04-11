import os
import pandas as pd
from pathlib import Path

def generate_train_csv(data_dir, output_csv):
    """
    Проходит по директории датасета и создает CSV файл.
    Ожидается структура: data_dir / label_name / image_files
    """
    data = []
    data_path = Path(data_dir)
    
    # Проходим по всем подпапкам (каждая папка - это отдельный тритон)
    for label_dir in data_path.iterdir():
        if label_dir.is_dir():
            label = label_dir.name # Имя папки становится ID тритона
            
            # Ищем все изображения внутри папки
            for img_path in label_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    data.append({
                        'image_path': str(img_path),
                        'label': label
                    })
                    
    # Создаем DataFrame и сохраняем
    df = pd.DataFrame(data)
    
    if df.empty:
        print(f"⚠️ Внимание: Изображения в {data_dir} не найдены!")
    else:
        df.to_csv(output_csv, index=False)
        print(f"✅ Успешно! CSV сохранен в {output_csv}")
        print(f"📊 Всего изображений: {len(df)}")
        print(f"🦎 Уникальных тритонов: {df['label'].nunique()}")

if __name__ == '__main__':
    # Укажи точный путь до папки с ID тритонов
    DATA_DIRECTORY = r'C:\Proekt_Practicum\newt_reid_project\data\train_crops'
    OUTPUT_FILE = r'C:\Proekt_Practicum\newt_reid_project\data\train.csv'
    
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    generate_train_csv(DATA_DIRECTORY, OUTPUT_FILE)