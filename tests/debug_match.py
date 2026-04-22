import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

from src.model import NewtReIDModel
from src.inference_utils import crop_belly

def load_image(path):
    """Умная загрузка: если в имени есть unwrapped, грузим как есть, иначе кропаем"""
    if 'unwrapped' in path.lower():
        return Image.open(path).convert('RGB')
    else:
        return crop_belly(path)

def main(query_path, gt_path_67, imposter_path_73):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Загрузка модели
    print("📂 Загрузка модели...")
    model = NewtReIDModel(pretrained=False).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Обработка изображений
    print("📷 Подготовка изображений...")
    img_q = load_image(query_path)
    img_67 = load_image(gt_path_67)
    img_73 = load_image(imposter_path_73)
    
    if None in [img_q, img_67, img_73]:
        print("❌ Ошибка: Одно из изображений не удалось сегментировать.")
        return

    # 3. Извлечение векторов
    print("🔮 Вычисление сходства...")
    with torch.no_grad():
        # Считаем для Query (Нормальное и 180, чтобы сымитировать наш TTA)
        t_q_norm = transform(img_q).unsqueeze(0).to(device)
        t_q_180 = transform(img_q.rotate(180)).unsqueeze(0).to(device)
        
        t_67 = transform(img_67).unsqueeze(0).to(device)
        t_73 = transform(img_73).unsqueeze(0).to(device)
        
        emb_q_norm = F.normalize(model(t_q_norm), p=2, dim=1)
        emb_q_180 = F.normalize(model(t_q_180), p=2, dim=1)
        emb_67 = F.normalize(model(t_67), p=2, dim=1)
        emb_73 = F.normalize(model(t_73), p=2, dim=1)
        
    # 4. Математическая дуэль
    # Сравниваем с 67
    score_67_norm = torch.mm(emb_q_norm, emb_67.t()).item()
    score_67_180 = torch.mm(emb_q_180, emb_67.t()).item()
    best_67 = max(score_67_norm, score_67_180)
    
    # Сравниваем с 73
    score_73_norm = torch.mm(emb_q_norm, emb_73.t()).item()
    score_73_180 = torch.mm(emb_q_180, emb_73.t()).item()
    best_73 = max(score_73_norm, score_73_180)
    
    # 5. Визуализация (Отрисовка улик)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_q)
    axes[0].set_title(f"Query (Test)\nСпорное фото")
    axes[0].axis('off')
    
    axes[1].imshow(img_67)
    axes[1].set_title(f"ID 67 (Истина)\nScore: {best_67:.4f}")
    axes[1].axis('off')
    
    axes[2].imshow(img_73)
    axes[2].set_title(f"ID 73 (Самозванец)\nScore: {best_73:.4f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnostic_duel.png')
    print("\n" + "="*40)
    print(f"📊 РЕЗУЛЬТАТЫ ДУЭЛИ:")
    print(f"Сходство с 67: {best_67:.4f}")
    print(f"Сходство с 73: {best_73:.4f}")
    print("✅ Картинка-улика сохранена в 'diagnostic_duel.png'")
    print("="*40)

if __name__ == "__main__":
    # ЗАМЕНИ ПУТИ НА РЕАЛЬНЫЕ!
    # Выбери одно хорошее, эталонное фото 67-го и одно 73-го из своей галереи
    Q_PATH = 'data/gallery/67/IMG_9747.JPG'
    GT_PATH = 'data/train_unwrapped/67/IMG_9747__unwrapped.jpg' 
    IMP_PATH = 'data/train_unwrapped/73/IMG_9787__unwrapped.jpg'
    
    main(Q_PATH, GT_PATH, IMP_PATH)