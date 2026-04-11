# запуск: python inference.py "путь фотографии" (например, python inference.py data/gallery/1/IMG_9301.jpg)
import os
import json
import argparse
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
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ (CLI)
# ==========================================
if __name__ == "__main__":
    # Настраиваем парсер аргументов
    parser = argparse.ArgumentParser(description="🦎 Система распознавания тритонов (Re-ID)")
    parser.add_argument("image_path", type=str, help="Путь к сырой фотографии тритона")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Ошибка: Файл '{args.image_path}' не найден!")
        exit(1)

    print("🚀 Инициализация системы Re-ID...")
    # Загружаем движок (с оптимизированной загрузкой .pt файла)
    engine = NewtMatchEngine(
        model_weights_path='models/best_model.pth',
        db_pt_path='data/vector_database.pt', 
        threshold=0.75
    )
    
    print(f"🔍 Анализ фотографии: {args.image_path}")
    
    # Всегда используем is_raw=True для реальных пользовательских данных
    result_json = engine.process_query(
        query_id=os.path.basename(args.image_path), 
        image_path=args.image_path, 
        is_raw=True
    )
    
    print("\n📊 Результат распознавания:")
    print(result_json)