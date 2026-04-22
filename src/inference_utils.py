import sys
import os
import cv2
import torch
from PIL import Image

# КРИТИЧЕСКАЯ ПРАВКА: Указываем путь прямо до папки 'src' внутри 'test_model'
repo_src_path = os.path.abspath(os.path.join('test_model', 'src'))
if repo_src_path not in sys.path:
    sys.path.insert(0, repo_src_path)

def crop_belly(raw_image_path):
        """
        Адаптер для внешней модели детекции и вырезания брюшка (AttentionUNet).
        """
        try:
            # Теперь Python без проблем найдет predict.py внутри test_model/src/
            from predict import predict
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Путь к весам модели сегментации
            seg_model_path = os.path.join('test_model', 'models', 'best_model.pth')
            
            if not os.path.exists(seg_model_path):
                print(f"⚠️ Ошибка: Не найдены веса сегментации по пути {seg_model_path}")
                return None

            # Обрабатываем сырое фото
            results = predict(raw_image_path, model_path=seg_model_path, device=device)
            
            # === ПЕРЕКЛЮЧАЕМСЯ НА НОВЫЙ АЛГОРИТМ РАЗВЕРТКИ ===
            if results is None or results.get('unwrapped') is None:
                print(f"⚠️ Тритон не найден на фото: {raw_image_path}")
                return None
                
            unwrapped_bgr = results['unwrapped']
            
            # Конвертируем BGR (OpenCV) в RGB (наш ViT) и отдаем
            unwrapped_rgb = cv2.cvtColor(unwrapped_bgr, cv2.COLOR_BGR2RGB)
            cropped_pil_image = Image.fromarray(unwrapped_rgb)
            
            return cropped_pil_image
            
        except Exception as e:
            print(f"⚠️ Ошибка сегментации: {e}")
            import traceback
            traceback.print_exc()
            return None