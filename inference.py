import os
import json
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

from src.model import NewtReIDModel
from src.database import VectorDatabase
from src.inference_utils import crop_belly

class NewtMatchEngine:
    def __init__(self, model_weights_path, db_pt_path, threshold=0.75):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # 1. Загрузка модели
        self.model = NewtReIDModel(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()
        
        # 2. Настройка трансформаций (без аугментаций)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 3. Инициализация и наполнение векторной базы
        self.db = VectorDatabase(self.device)
        self._load_database(db_pt_path)

    def _load_database(self, pt_path):
        """Мгновенная загрузка заранее вычисленных векторов"""
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Файл БД {pt_path} не найден. Сначала запустите build_vector_db.py")
            
        data = torch.load(pt_path, map_location=self.device)
        
        # Загружаем векторы из архива напрямую в нашу VectorDatabase
        for emb, label in zip(data['embeddings'], data['labels']):
            self.db.add(emb.to(self.device), [label])

    def process_query(self, query_id, image_path, is_raw=False):
        """
        Основной метод пайплайна.
        is_raw=True -> Сначала отправляем в YOLO для обрезки.
        is_raw=False -> Картинка уже обрезана, сразу в ViT.
        """
        # --- ЭТАП 1: Подготовка изображения ---
        if is_raw:
            processed_img = crop_belly(image_path)
            if processed_img is None:
                return self._generate_error(query_id, "Ошибка сегментации изображения")
        else:
            processed_img = Image.open(image_path).convert('RGB')

        # --- ЭТАП 2: Извлечение признаков (Embedding) ---
        input_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model(input_tensor)

        # --- ЭТАП 3: Поиск и логика принятия решения ---
        # Ищем топ-20 кандидатов, как указано в ТЗ
        matches = self.db.search(query_embedding, top_k=20) 
        
        if not matches:
            return self._generate_error(query_id, "База данных пуста")

        best_match = matches[0]
        # Если сходство меньше порога (0.75), считаем особь новой
        is_new = best_match['score'] < self.threshold
        
        # --- ЭТАП 4: Форматирование JSON-ответа ---
        response = {
            "query_id": query_id,
            "top_k": [m['label'] for m in matches],
            "scores": [round(m['score'], 4) for m in matches],
            "best_match": None if is_new else best_match['label'],
            "confidence": round(best_match['score'], 4),
            "is_new": is_new
        }
        
        return json.dumps(response, indent=4, ensure_ascii=False)

    def _generate_error(self, query_id, error_msg):
        """Формирование JSON при ошибке"""
        return json.dumps({
            "query_id": query_id,
            "error": error_msg,
            "is_new": None
        }, indent=4, ensure_ascii=False)


# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================
if __name__ == "__main__":
    # Инициализируем движок (потребуется пара секунд на загрузку базы)
    print("Инициализация системы Re-ID...")
    engine = NewtMatchEngine(
        model_weights_path='models/best_model.pth',
        db_pt_path='data/vector_database.pt', # <-- ТЕПЕРЬ ПУТЬ К БИНАРНИКУ
        threshold=0.75
    )
    
    print("\n--- Тест 1: Готовое фото (is_raw=False) ---")
    # Подставь сюда путь к любой картинке из твоего датасета
    test_path_crop = "data/train_crops/76/IMG_9810__unwrapped.jpg" 
    result_crop = engine.process_query(query_id="req_001", image_path=test_path_crop, is_raw=False)
    print(result_crop)

    print("\n--- Тест 2: Сырое фото с фоном (is_raw=True) ---")
    # Подставь сюда путь к сырой картинке тритона в чашке Петри
    test_path_raw = "data/gallery/1/IMG_9301.jpg"
    # Создадим фиктивный файл для теста, если его нет
    if not os.path.exists(test_path_raw):
        os.makedirs(os.path.dirname(test_path_raw), exist_ok=True)
        Image.new('RGB', (800, 600)).save(test_path_raw)
        
    result_raw = engine.process_query(query_id="req_002", image_path=test_path_raw, is_raw=True)
    print(result_raw)