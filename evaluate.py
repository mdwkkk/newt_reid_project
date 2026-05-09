import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import mlflow # Шаг 1: Импорт MLflow

# Импортируем архитектуру нашей модели
from src.model import NewtReIDModel

def extract_features(csv_path, model_weights_path, device):
    """Прогоняет тестовые картинки через модель и собирает векторы"""
    print(f"📦 Загрузка модели и извлечение признаков из {csv_path}...")
    
    model = NewtReIDModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
            # Было: transforms.Resize((224, 224)),
            # СТАЛО:
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    df = pd.read_csv(csv_path)
    embeddings = []
    labels = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка фото"):
            img = Image.open(row['image_path']).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            emb = model(tensor).cpu().numpy()[0]
            
            embeddings.append(emb)
            labels.append(str(row['label']))

    return np.array(embeddings), np.array(labels)

def compute_metrics_and_distributions(embeddings, labels):
    """Считает Rank-K, mAP и собирает распределения скоров"""
    print("📊 Вычисление метрик ранжирования и матриц расстояний...")
    
    n = len(labels)
    # Нормализуем векторы для косинусного сходства
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    aps = []
    cmc_ranks = {1: 0, 5: 0, 20: 0}
    valid_queries = 0
    
    pos_scores = []
    neg_scores = []

    for i in tqdm(range(n), desc="Оценка метрик"):
        query_label = labels[i]
        sim_scores = similarity_matrix[i]
        mask_self = np.arange(n) != i
        
        sim_scores_others = sim_scores[mask_self]
        labels_others = labels[mask_self]

        is_same_label = (labels_others == query_label)
        pos_scores.extend(sim_scores_others[is_same_label])
        neg_scores.extend(sim_scores_others[~is_same_label])

        if not np.any(is_same_label):
            continue
            
        valid_queries += 1
        sort_indices = np.argsort(sim_scores_others)[::-1]
        sorted_matches = is_same_label[sort_indices]

        if sorted_matches[0]: cmc_ranks[1] += 1
        if np.any(sorted_matches[:5]): cmc_ranks[5] += 1
        if np.any(sorted_matches[:20]): cmc_ranks[20] += 1

        relevant_hits = 0
        precision_sum = 0.0
        for rank, is_match in enumerate(sorted_matches, 1):
            if is_match:
                relevant_hits += 1
                precision_sum += relevant_hits / rank
        
        aps.append(precision_sum / relevant_hits)

    # Собираем словарь метрик для MLflow
    metrics = {
        "test_mAP": np.mean(aps) if aps else 0,
        "test_Rank-1": cmc_ranks[1] / valid_queries if valid_queries else 0,
        "test_Rank-5": cmc_ranks[5] / valid_queries if valid_queries else 0,
        "test_Rank-20": cmc_ranks[20] / valid_queries if valid_queries else 0
    }

    print("\n" + "="*40)
    print("🏆 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*40)

    return pos_scores, neg_scores, metrics

def plot_distributions(pos_scores, neg_scores, output_path="score_distribution.png"):
    """Строит графики для визуального выбора идеального порога"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pos_scores, color="green", fill=True, label="Свои")
    sns.kdeplot(neg_scores, color="red", fill=True, label="Чужие")
    plt.axvline(x=0.49, color='blue', linestyle='--', label="Порог (0.49)")
    plt.title("Распределение косинусного сходства")
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close() # Закрываем фигуру, чтобы не забивать память

if __name__ == "__main__":
    TEST_CSV = 'data/test_unet.csv' 
    MODEL_WEIGHTS = 'models/best_model.pth'
    
    # Шаг 2: Настройка эксперимента MLflow
    mlflow.set_experiment("Newt_ReID_Optimization")
    
    # Запускаем оценку в контексте MLflow
    with mlflow.start_run(run_name="Final_Evaluation"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(TEST_CSV):
            print(f"❌ Ошибка: Файл {TEST_CSV} не найден!")
            exit(1)

        # 1. Извлекаем фичи
        embeddings, labels = extract_features(TEST_CSV, MODEL_WEIGHTS, device)
        
        # 2. Считаем метрики (теперь возвращает и словарь метрик)
        pos_scores, neg_scores, metrics = compute_metrics_and_distributions(embeddings, labels)
        
        # 3. Логируем метрики в MLflow
        mlflow.log_metrics(metrics)
        
        # 4. Рисуем и логируем график как артефакт
        plot_path = "score_distribution.png"
        plot_distributions(pos_scores, neg_scores, plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        print(f"✅ Результаты и график успешно залогированы в MLflow")