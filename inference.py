import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.database import VectorDatabase
from src.inference_utils import crop_belly
from src.model import NewtReIDModel


class NewtMatchEngine:
    def __init__(self, model_weights_path, db_pt_path=None, threshold=0.49):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.db_pt_path = db_pt_path

        self.model = NewtReIDModel(pretrained=False, img_size=256).to(self.device)
        self.model.load_state_dict(
            torch.load(model_weights_path, map_location=self.device)
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.db = VectorDatabase(self.device)
        if db_pt_path and os.path.exists(db_pt_path):
            self._load_database(db_pt_path)

    def _load_database(self, pt_path):
        loaded = VectorDatabase.load_from_pt(pt_path, self.device)
        self.db = loaded

    @property
    def is_empty(self) -> bool:
        return self.db.is_empty

    def persist_db(self, path: str | None = None) -> None:
        target = path or self.db_pt_path
        if not target:
            raise ValueError("db path not set")
        self.db.save(target)
        self.db_pt_path = target

    def add_embedding(self, label: str, embedding) -> None:
        self.db.add(embedding, [str(label)])

    def remove_label(self, label: str) -> int:
        return self.db.remove_label(str(label))

    def extract_embedding_from_pil(self, pil_image: Image.Image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def extract_embedding_from_path(self, image_path, is_raw=True):
        if is_raw:
            pil_image = crop_belly(image_path)
            if pil_image is None:
                return None, "Не удалось сегментировать тритона"
        else:
            pil_image = Image.open(image_path).convert("RGB")
        emb = self.extract_embedding_from_pil(pil_image)
        return emb, pil_image

    def process_query(self, query_id, image_path, is_raw=True):
        try:
            if self.db.is_empty:
                return json.dumps(
                    {
                        "query_id": query_id,
                        "error_code": "project_empty",
                        "error": "В проекте нет особей. Добавьте особь, чтобы начать поиск.",
                        "is_new": None,
                    },
                    indent=4,
                    ensure_ascii=False,
                )

            if is_raw:
                pil_image = crop_belly(image_path)
                if pil_image is None:
                    return self._generate_error(
                        query_id, "Не удалось сегментировать тритона"
                    )
            else:
                pil_image = Image.open(image_path).convert("RGB")

            pil_image.save(f"debug_input_{query_id}.jpg")

            embedding = self.extract_embedding_from_pil(pil_image)

            results = self.db.search(embedding, top_k=20)
            if not results:
                return self._generate_error(query_id, "Поиск не вернул результатов")

            best_overall_score = float(results[0]["score"])
            is_new = bool(best_overall_score < self.threshold)

            top_matches_clean = [
                {"label": res["label"], "score": round(float(res["score"]), 4)}
                for res in results
            ]

            response = {
                "query_id": query_id,
                "top_20_candidates": top_matches_clean,
                "best_match": str(results[0]["label"]) if not is_new else None,
                "confidence": round(best_overall_score, 4),
                "is_new": is_new,
            }
            return json.dumps(response, indent=4, ensure_ascii=False)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._generate_error(query_id, str(e))

    def _generate_error(self, query_id, error_msg):
        return json.dumps(
            {
                "query_id": query_id,
                "error": error_msg,
                "is_new": None,
            },
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Система распознавания тритонов (Re-ID)"
    )
    parser.add_argument("image_path", type=str, help="Путь к фотографии тритона")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Ошибка: файл '{args.image_path}' не найден!")
        exit(1)

    print("Инициализация системы Re-ID...")
    engine = NewtMatchEngine(
        model_weights_path="models/best_model.pth",
        db_pt_path="data/vector_database.pt",
        threshold=0.49,
    )

    print(f"Анализ фотографии: {args.image_path}")

    result_json = engine.process_query(
        query_id=os.path.basename(args.image_path),
        image_path=args.image_path,
        is_raw=True,
    )

    print("\nРезультат распознавания:")
    print(result_json)
