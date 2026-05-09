import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
<<<<<<< HEAD
import mlflow # Шаг 1: Импорт MLflow
=======
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406

# Импортируем архитектуру нашей модели
from src.model import NewtReIDModel

def extract_features(csv_path, model_weights_path, device):
    """Прогоняет тестовые картинки через модель и собирает векторы"""
    print(f"📦 Загрузка модели и извлечение признаков из {csv_path}...")
    
    model = NewtReIDModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
<<<<<<< HEAD
            # Было: transforms.Resize((224, 224)),
            # СТАЛО:
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
=======
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406

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
<<<<<<< HEAD
    # Нормализуем векторы для косинусного сходства
=======
    # Считаем матрицу косинусного сходства (NxN)
    # Нормализуем векторы
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
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
<<<<<<< HEAD
=======
        
        # Получаем сходство со всеми фото, кроме самого себя (i)
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
        sim_scores = similarity_matrix[i]
        mask_self = np.arange(n) != i
        
        sim_scores_others = sim_scores[mask_self]
        labels_others = labels[mask_self]

<<<<<<< HEAD
=======
        # Разделяем скоры для графиков
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
        is_same_label = (labels_others == query_label)
        pos_scores.extend(sim_scores_others[is_same_label])
        neg_scores.extend(sim_scores_others[~is_same_label])

<<<<<<< HEAD
=======
        # Если в галерее больше нет фото этого тритона — пропускаем из расчета mAP/Rank
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
        if not np.any(is_same_label):
            continue
            
        valid_queries += 1
<<<<<<< HEAD
        sort_indices = np.argsort(sim_scores_others)[::-1]
        sorted_matches = is_same_label[sort_indices]

=======

        # Сортируем от большего сходства к меньшему
        sort_indices = np.argsort(sim_scores_others)[::-1]
        sorted_matches = is_same_label[sort_indices]

        # Rank-K
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
        if sorted_matches[0]: cmc_ranks[1] += 1
        if np.any(sorted_matches[:5]): cmc_ranks[5] += 1
        if np.any(sorted_matches[:20]): cmc_ranks[20] += 1

<<<<<<< HEAD
=======
        # Average Precision (AP)
        # Считаем классическую метрику AP для одного запроса
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
        relevant_hits = 0
        precision_sum = 0.0
        for rank, is_match in enumerate(sorted_matches, 1):
            if is_match:
                relevant_hits += 1
                precision_sum += relevant_hits / rank
        
        aps.append(precision_sum / relevant_hits)

<<<<<<< HEAD
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
=======
    mAP = np.mean(aps) if aps else 0
    rank1 = cmc_ranks[1] / valid_queries if valid_queries else 0
    rank5 = cmc_ranks[5] / valid_queries if valid_queries else 0
    rank20 = cmc_ranks[20] / valid_queries if valid_queries else 0

    print("\n" + "="*40)
    print("🏆 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ (Open-Set Re-ID)")
    print("="*40)
    print(f"mAP (Mean Avg Precision): {mAP:.4f}")
    print(f"Rank-1 Accuracy:          {rank1:.4f} ({(rank1*100):.1f}%)")
    print(f"Rank-5 Accuracy:          {rank5:.4f} ({(rank5*100):.1f}%)")
    print(f"Rank-20 Accuracy:         {rank20:.4f} ({(rank20*100):.1f}%)")
    print("="*40)

    return pos_scores, neg_scores

def plot_distributions(pos_scores, neg_scores, output_path="score_distribution.png"):
    """Строит графики для визуального выбора идеального порога"""
    print(f"📈 Отрисовка распределений, сохранение в {output_path}...")
    
    plt.figure(figsize=(10, 6))
    
    # Строим график плотности с помощью Seaborn
    sns.kdeplot(pos_scores, color="green", fill=True, label="Свои (Один и тот же тритон)")
    sns.kdeplot(neg_scores, color="red", fill=True, label="Чужие (Разные тритоны)")
    
    # Добавляем наш текущий порог
    plt.axvline(x=0.49, color='blue', linestyle='--', linewidth=2, label="Текущий Порог (0.49)")
    
    plt.title("Распределение косинусного сходства (Positive vs Negative Pairs)", fontsize=14)
    plt.xlabel("Косинусное сходство (Cosine Similarity)", fontsize=12)
    plt.ylabel("Плотность (Density)", fontsize=12)
    plt.xlim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print("✅ График успешно сохранен!")

if __name__ == "__main__":
    # ВАЖНО: Укажи путь к отложенной ТЕСТОВОЙ выборке (а не к train.csv)
    # Если ее нет, можешь запустить на train.csv, чтобы проверить на переобучение.
    TEST_CSV = 'data/test_unet.csv' 
    MODEL_WEIGHTS = 'models/best_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(TEST_CSV):
        print(f"❌ Ошибка: Файл {TEST_CSV} не найден! Создай тестовую выборку.")
        exit(1)

    # 1. Извлекаем фичи
    embeddings, labels = extract_features(TEST_CSV, MODEL_WEIGHTS, device)
    
    # 2. Считаем метрики
    pos_scores, neg_scores = compute_metrics_and_distributions(embeddings, labels)
    
    # 3. Рисуем график
    plot_distributions(pos_scores, neg_scores)
>>>>>>> 42df82769ba38241f1aa129fb40bc2a7e53a5406
