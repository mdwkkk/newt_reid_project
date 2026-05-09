import os
import json
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

from src.model import NewtReIDModel
from src.database import VectorDatabase
from src.inference_utils import crop_belly

class NewtMatchEngine:
    def __init__(self, model_weights_path, db_pt_path, threshold=0.49):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # 1. Загрузка модели
        self.model = NewtReIDModel(pretrained=False, img_size=256).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()
        
        # 2. Настройка трансформаций (без аугментаций)
        self.transform = transforms.Compose([
            # Было: transforms.Resize((224, 224)),
            # СТАЛО:
            transforms.Resize((256, 256)), 
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
        for emb, label in zip(data['embeddings'], data['labels']):
            # === БРОНЕБОЙНАЯ ЗАЩИТА РАЗМЕРНОСТИ ===
            # .view(-1) сплющивает любые скобки, делая чистый 1D вектор (512,)
            # .unsqueeze(0) делает из него строгую 2D матрицу (1, 512)
            emb_2d = emb.view(-1).unsqueeze(0).to(self.device)
            
            self.db.add(emb_2d, [label])
    
    def process_query(self, query_id, image_path, is_raw=True):
        """Главный метод обработки: кроп -> извлечение (Без TTA) -> поиск"""
        try:
            if is_raw:
                pil_image = crop_belly(image_path)
                if pil_image is None:
                    return self._generate_error(query_id, "Не удалось сегментировать тритона")
            else:
                pil_image = Image.open(image_path).convert('RGB')

            # === ПРЯМОЙ ИНФЕРЕНС (Без TTA) ===
            # predict.py уже аппаратно выровнял тритона головой вверх!
            
            # Сохраняем DEBUG INPUT, чтобы глазами видеть идеальную каноничную развертку
            pil_image.save(f"debug_input_{query_id}.jpg")

            # Переводим в тензор (размер батча = 1)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Прогоняем всего 1 картинку (Инференс стал в 2 раза быстрее!)
                embedding = self.model(tensor)
                embedding = F.normalize(embedding, p=2, dim=1) # (1, 512)

            # Поиск в векторной базе (Берем Топ-20)
            results = self.db.search(embedding, top_k=20) 
            best_overall_score = float(results[0]['score'])
            
            # Проверка порога (например, 0.49)
            is_new = bool(best_overall_score < self.threshold)

            # Формируем красивый список из 20 конкурентов
            top_matches_clean = [
                {"label": res['label'], "score": round(float(res['score']), 4)} 
                for res in results
            ]

            response = {
                "query_id": query_id,
                "top_20_candidates": top_matches_clean,
                "best_match": str(results[0]['label']) if not is_new else None,
                "confidence": round(best_overall_score, 4),
                "is_new": is_new,
            }
            return json.dumps(response, indent=4, ensure_ascii=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._generate_error(query_id, str(e))

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
    parser = argparse.ArgumentParser(description="🦎 Система распознавания тритонов (Re-ID)")
    parser.add_argument("image_path", type=str, help="Путь к сырой фотографии тритона")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Ошибка: Файл '{args.image_path}' не найден!")
        exit(1)

    print("🚀 Инициализация системы Re-ID...")
    engine = NewtMatchEngine(
        model_weights_path='models/best_model.pth',
        db_pt_path='data/vector_database.pt', 
        threshold=0.49
    )
    
    print(f"🔍 Анализ фотографии: {args.image_path}")
    
    result_json = engine.process_query(
        query_id=os.path.basename(args.image_path), 
        image_path=args.image_path, 
        is_raw=True
    )
    
    print("\n📊 Результат распознавания:")
    print(result_json)