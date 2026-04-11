import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import os

from src.model import NewtReIDModel

def build_offline_db(csv_path, model_weights, output_pt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Инициализация индексации на {device}...")

    # 1. Загрузка модели
    model = NewtReIDModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()

    # 2. Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Чтение данных
    df = pd.read_csv(csv_path)
    
    embeddings_list = []
    labels_list = []
    
    print(f"📦 Обработка {len(df)} изображений...")
    
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 100 == 0 and i > 0:
                print(f"  ... обработано {i}/{len(df)}")
                
            img = Image.open(row['image_path']).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(tensor)
            
            # Сохраняем вектор на CPU, чтобы не забить видеопамять (если есть GPU)
            embeddings_list.append(embedding.cpu()) 
            labels_list.append(row['label'])

    # 4. Сохранение на диск
    data = {
        'embeddings': embeddings_list,
        'labels': labels_list
    }
    torch.save(data, output_pt_path)
    print(f"✅ База данных успешно скомпилирована и сохранена в {output_pt_path}!")

if __name__ == "__main__":
    CSV_PATH = 'data/train_unet.csv' # Наша выровненная база
    MODEL_WEIGHTS = 'models/best_model.pth' # Наши актуальные веса
    OUTPUT_PT = 'data/vector_database.pt' # Файл, который будет загружаться за 0.1 сек
    
    build_offline_db(CSV_PATH, MODEL_WEIGHTS, OUTPUT_PT)