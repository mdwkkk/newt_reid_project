import gradio as gr
import cv2
import numpy as np
import torch
import os
from PIL import Image
import tempfile
from model import AttentionUNet
from test_model.src.predict import predict, save_results

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'models/best_model.pth'

def load_model():
    model = AttentionUNet(n_channels=3, n_classes=1)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        dice = checkpoint.get('val_dice', 0)
        return model, dice
    return None, 0

model, model_dice = load_model()
if model:
    model.to(device)
    model.eval()

def process_image(image, threshold=0.5):
    if image is None:
        return None, None, None, "❌ Загрузите изображение"
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, 'temp_input.jpg')
    
    if isinstance(image, np.ndarray):
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        image.save(temp_path)
    
    try:
        results = predict(temp_path, model_path=model_path, device=device, threshold=threshold)
    except Exception as e:
        return None, None, None, f"❌ Ошибка: {str(e)}"
    
    if results is None:
        return None, None, None, "❌ Тритон не найден на изображении"
    
    save_results(results, 'results')
    
    overlay_pil = Image.fromarray(cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB))
    
    if results['stretched'] is not None:
        stretched_pil = Image.fromarray(cv2.cvtColor(results['stretched'], cv2.COLOR_BGR2RGB))
    else:
        stretched_pil = None
    
    x, y, w, h = results['bbox']
    message = f"✅ Найдено! | Размер: {w}x{h} px | Dice: {model_dice:.2%}"
    
    return overlay_pil, stretched_pil, results['mask'], message

with gr.Blocks(title="🦎 Сегментация тритонов", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦎 Автоматическая сегментация брюха тритона")
    gr.Markdown("Загрузите фото тритона в чашке Петри для автоматического выделения брюха")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="📷 Загрузить фото", type="numpy")
            threshold_slider = gr.Slider(minimum=0.3, maximum=0.8, value=0.5, step=0.05, label="🎚️ Порог")
            submit_btn = gr.Button("🚀 Обработать", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_overlay = gr.Image(label="🎯 Результат")
            output_stretched = gr.Image(label="📐 Растянутое брюхо")
    
    output_mask = gr.Image(label="🎭 Маска", type="numpy")
    status_text = gr.Textbox(label="📊 Статус", interactive=False)
    
    submit_btn.click(fn=process_image, inputs=[input_img, threshold_slider], outputs=[output_overlay, output_stretched, output_mask, status_text])

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)