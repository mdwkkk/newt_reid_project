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
        
        # Путь к весам модели сегментации (Проверь, что они лежат именно тут!)
        seg_model_path = os.path.join('test_model', 'models', 'best_model.pth')
        
        if not os.path.exists(seg_model_path):
            print(f"⚠️ Ошибка: Не найдены веса сегментации по пути {seg_model_path}")
            return None

        # Обрабатываем сырое фото
        results = predict(raw_image_path, model_path=seg_model_path, device=device)
        
        if results is None or results.get('stretched') is None:
            print(f"⚠️ Тритон не найден на фото: {raw_image_path}")
            return None
            
        stretched_bgr = results['stretched']
        
        # Конвертируем BGR (OpenCV) в RGB (наш ViT) и отдаем
        stretched_rgb = cv2.cvtColor(stretched_bgr, cv2.COLOR_BGR2RGB)
        cropped_pil_image = Image.fromarray(stretched_rgb)
        
        return cropped_pil_image
        
    except Exception as e:
        import traceback
        print(f"⚠️ Ошибка сегментации:\n{traceback.format_exc()}")
        return None