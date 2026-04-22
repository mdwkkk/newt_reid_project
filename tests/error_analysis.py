import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os

from src.model import NewtReIDModel

def load_embeddings(csv_path, model_weights_path, device):
    """Извлекает векторы из тестовой базы (аналогично evaluate.py)"""
    model = NewtReIDModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(csv_path)
    embeddings, labels, paths = [], [], []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение признаков"):
            img = Image.open(row['image_path']).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            embeddings.append(model(tensor).cpu().numpy()[0])
            labels.append(str(row['label']))
            paths.append(row['image_path'])

    return np.array(embeddings), np.array(labels), paths

def find_and_plot_errors(embeddings, labels, paths, output_file="error_analysis.png"):
    # Нормализация и матрица сходства
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    hard_positives = [] # Свои, но низкий скор
    hard_negatives = [] # Чужие, но высокий скор
    
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            is_same = (labels[i] == labels[j])
            
            if is_same:
                hard_positives.append((sim, paths[i], paths[j], labels[i], labels[j]))
            else:
                hard_negatives.append((sim, paths[i], paths[j], labels[i], labels[j]))
                
    # Сортируем: берем самые низкие скоры для "Своих" и самые высокие для "Чужих"
    hard_positives.sort(key=lambda x: x[0]) 
    hard_negatives.sort(key=lambda x: x[0], reverse=True)
    
    # --- Отрисовка ---
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Анализ ошибок Re-ID (Топ-4 худших случаев)", fontsize=16)
    
    def draw_pair(ax, pair_data, title, is_error_red):
        sim, p1, p2, l1, l2 = pair_data
        img1, img2 = Image.open(p1), Image.open(p2)
        
        # Склеиваем две картинки рядом
        dst = Image.new('RGB', (img1.width + img2.width + 20, max(img1.height, img2.height)), (255, 255, 255))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width + 20, 0))
        
        ax.imshow(dst)
        ax.axis('off')
        color = 'red' if is_error_red else 'orange'
        ax.set_title(f"{title}\nСкор: {sim:.4f} | ID: {l1} vs {l2}", color=color, fontsize=12, fontweight='bold')

    # Рисуем топ-4 Hard Positives (Ложный отказ)
    for idx in range(min(4, len(hard_positives))):
        draw_pair(axes[idx, 0], hard_positives[idx], "🔴 HARD POSITIVE (Не узнал своего)", is_error_red=True)
        
    # Рисуем топ-4 Hard Negatives (Ложный допуск)
    for idx in range(min(4, len(hard_negatives))):
        draw_pair(axes[idx, 1], hard_negatives[idx], "🟠 HARD NEGATIVE (Перепутал чужих)", is_error_red=False)
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"\n✅ Анализ завершен! Открой файл {output_file}")

if __name__ == "__main__":
    TEST_CSV = 'data/test_unet.csv'
    MODEL_WEIGHTS = 'models/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    emb, lbl, pth = load_embeddings(TEST_CSV, MODEL_WEIGHTS, device)
    find_and_plot_errors(emb, lbl, pth)