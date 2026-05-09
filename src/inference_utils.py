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
    Адаптер для внешней модели детекции и вырезания брюшка.
    Обновлен под новую версию predict.py (EfficientNet + Canonical).
    """
    try:
        from predict import predict
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seg_model_path = os.path.join('test_model', 'models', 'best_model.pth')
        
        if not os.path.exists(seg_model_path):
            print(f"⚠️ Ошибка: Не найдены веса сегментации по пути {seg_model_path}")
            return None

        # Обрабатываем сырое фото
        results = predict(raw_image_path, model_path=seg_model_path, device=device)
        
        # === ОБНОВЛЕНИЕ: Ищем ключ 'canonical' вместо 'unwrapped' ===
        if results is None or results.get('canonical') is None:
            print(f"⚠️ Тритон не найден или не канонизирован: {raw_image_path}")
            return None
            
        canonical_bgra = results['canonical']
        
        # === ОБНОВЛЕНИЕ: Конвертируем 4 канала (BGRA) в 3 канала (RGB) ===
        # Прозрачный фон станет черным, что отлично подходит для ViT
        canonical_rgb = cv2.cvtColor(canonical_bgra, cv2.COLOR_BGRA2RGB)
        
        cropped_pil_image = Image.fromarray(canonical_rgb)
        return cropped_pil_image
        
    except Exception as e:
        print(f"⚠️ Ошибка сегментации: {e}")
        import traceback
        traceback.print_exc()
        return None

