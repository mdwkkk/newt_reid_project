import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import time

from src.model import NewtReIDModel
from src.database import VectorDatabase

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Запуск на: {device}")

    # 1. Загрузка модели
    print("📦 Загрузка модели...")
    model = NewtReIDModel(pretrained=False).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    # Трансформации для инференса (без аугментаций!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Инициализация базы данных
    db = VectorDatabase(device)
    df = pd.read_csv('data/train.csv')
    
    # Для теста возьмем 200 случайных картинок из трейна, чтобы быстро собрать базу
    # (В продакшене тут будут вообще все известные эталонные фото)
    gallery_df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"🗄️ Создание базы данных из {len(gallery_df)} изображений...")
    start_time = time.time()
    
    with torch.no_grad():
        for idx, row in gallery_df.iterrows():
            img = Image.open(row['image_path']).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(tensor)
            db.add(embedding, [row['label']])
            
    print(f"✅ База собрана за {time.time() - start_time:.2f} сек.")

    # 3. Выбираем тестовое изображение (Query)
    # Возьмем случайное изображение из нашей базы, чтобы точно найти совпадение
    query_row = gallery_df.iloc[random.randint(0, len(gallery_df) - 1)]
    query_img_path = query_row['image_path']
    true_label = query_row['label']
    
    print(f"\n🔍 Тестируем поиск. Ищем: {true_label} (Файл: {query_img_path})")
    
    query_img = Image.open(query_img_path).convert('RGB')
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_embedding = model(query_tensor)

    # 4. Поиск в базе
    THRESHOLD = 0.75 # Порог из ТЗ
    matches = db.search(query_embedding, top_k=5) # Выведем топ-5 для удобства
    
    print("\n🏆 Результаты поиска (Топ-5):")
    best_match = matches[0]
    
    for i, match in enumerate(matches, 1):
        # Косинусное сходство: 1.0 = идеальное совпадение, 0.0 = вообще не похожи
        print(f"  {i}. ID: {match['label']} | Сходство: {match['score']:.4f}")
        
    print("-" * 30)
    if best_match['score'] < THRESHOLD:
        print("🦎 Вердикт: НОВЫЙ ТРИТОН (Совпадений ниже порога)")
    else:
        print(f"✅ Вердикт: Это тритон {best_match['label']}!")
        if best_match['label'] == true_label:
            print("🎯 Модель угадала абсолютно верно!")

if __name__ == '__main__':
    main()