import os
import json
import numpy as np
from PIL import Image, ImageDraw

def convert_labelme_json_to_png(json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    width = int(data['imageWidth'])
    height = int(data['imageHeight'])
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = [(p[0], p[1]) for p in shape['points']]
            draw.polygon(points, fill=255)
    
    base_name = os.path.basename(json_path).replace('.json', '_mask.png')
    output_path = os.path.join(output_dir, base_name)
    mask.save(output_path)
    
    return output_path

masks_dir = 'data/masks'

print("🔄 Конвертация всех JSON файлов...")
print("=" * 60)

count = 0
errors = 0

for filename in os.listdir(masks_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(masks_dir, filename)
        try:
            output_path = convert_labelme_json_to_png(json_path, masks_dir)
            print(f"✅ {filename} → {os.path.basename(output_path)}")
            count += 1
        except Exception as e:
            print(f"❌ Ошибка с {filename}: {e}")
            errors += 1

print("=" * 60)
print(f"✅ Конвертировано: {count} файлов")
print(f"❌ Ошибок: {errors}")

if errors == 0:
    print("\n🎉 Все файлы успешно конвертированы!")