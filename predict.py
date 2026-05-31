import cv2
import numpy as np
import torch
import os
import sys
import math

# НАСТРОЙКИ
THRESHOLD = 0.55
MIN_AREA = 1000
MORPH_KERNEL = 3
CANONICAL_SIZE = (256, 256)
HEAD_CUT_RATIO = 0.15

# АВТО-ПУТИ
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
BASE_DIR = os.path.dirname(src_dir)

images_dir = os.path.join(BASE_DIR, 'data/images')
models_dir = os.path.join(BASE_DIR, 'models')
results_dir = os.path.join(BASE_DIR, 'results')

def keep_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 100:
        return np.zeros_like(mask), None
    mask_filtered = np.zeros_like(mask)
    cv2.drawContours(mask_filtered, [largest], -1, 255, -1)
    return mask_filtered, largest

def find_head_tail_points(contour, mask):
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 20: return None, None
    
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    main_axis = eigenvectors[0]
    perp_axis = np.array([-main_axis[1], main_axis[0]])
    
    projections = np.dot(points - mean, main_axis)
    min_proj, max_proj = np.min(projections), np.max(projections)
    body_length = max_proj - min_proj
    
    if body_length < 50: return None, None
    
    tip_zone_size = body_length * 0.15
    points_tip1 = points[projections < (min_proj + tip_zone_size)]
    points_tip2 = points[projections > (max_proj - tip_zone_size)]
    
    if len(points_tip1) < 10 or len(points_tip2) < 10: return None, None
    
    thickness1 = np.std(np.dot(points_tip1 - mean, perp_axis))
    thickness2 = np.std(np.dot(points_tip2 - mean, perp_axis))
    
    if thickness1 > thickness2:
        head_idx, tail_idx = np.argmin(projections), np.argmax(projections)
    else:
        head_idx, tail_idx = np.argmax(projections), np.argmin(projections)
        
    return tuple(points[head_idx].astype(int)), tuple(points[tail_idx].astype(int))

def align_to_vertical(image, mask, contour, head_point, tail_point):
    h, w = image.shape[:2]
    dx = tail_point[0] - head_point[0]
    dy = tail_point[1] - head_point[1]
    
    current_angle = math.degrees(math.atan2(dy, dx))
    rotate_angle = current_angle - 90.0
    
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int(np.ceil((h * sin) + (w * cos)))
    new_h = int(np.ceil((h * cos) + (w * sin)))
    
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    
    rotated_img = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    contour_float = contour.astype(np.float32)
    rotated_contour = cv2.transform(contour_float, M).astype(np.int32)
    
    new_head = cv2.transform(np.array([[head_point]], dtype=np.float32), M)[0][0].astype(int)
    new_tail = cv2.transform(np.array([[tail_point]], dtype=np.float32), M)[0][0].astype(int)
    
    return rotated_img, rotated_mask, rotated_contour, tuple(new_head), tuple(new_tail)

def remove_head_region(mask, image_shape):
    h, w = image_shape[:2]
    mask_copy = mask.copy()
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask_copy
    largest = max(contours, key=cv2.contourArea)
    x, y, wb, hb = cv2.boundingRect(largest)
    cut_height = int(hb * HEAD_CUT_RATIO)
    if hb > 100:
        mask_copy[y:y+cut_height, x:x+wb] = 0
    return mask_copy

def extract_belly_exact(image, mask, contour):
    x, y, bw, bh = cv2.boundingRect(contour)
    belly_roi = image[y:y+bh, x:x+bw].copy()
    mask_roi = mask[y:y+bh, x:x+bw].copy()
    contour_shifted = contour - np.array([[x, y]])
    belly_rgba = cv2.cvtColor(belly_roi, cv2.COLOR_BGR2BGRA)
    alpha = np.zeros_like(mask_roi)
    cv2.drawContours(alpha, [contour_shifted], -1, 255, -1)
    belly_rgba[:, :, 3] = alpha
    return belly_rgba, (x, y, bw, bh), contour_shifted

def ensure_vertical_orientation(belly_rgba):
    return belly_rgba

def canonicalize_belly(belly_rgba, target_size=(256, 256)):
    h, w = belly_rgba.shape[:2]
    tw, th = target_size
    alpha = belly_rgba[:, :, 3]
    center_x, width_at_y = [], []
    for y in range(h):
        row = alpha[y, :]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            center_x.append(np.mean(indices))
            width_at_y.append(indices[-1] - indices[0] + 1)
        else:
            center_x.append(w / 2)
            width_at_y.append(0)
    center_x, width_at_y = np.array(center_x), np.array(width_at_y)
    
    canonical = np.zeros((th, tw, 4), dtype=np.uint8)
    for ty in range(th):
        sy = min(int((ty / th) * h), h - 1)
        if width_at_y[sy] > 0:
            cx, sw = center_x[sy], width_at_y[sy]
            if sw > 0:
                scale = tw / sw
                for tx in range(tw):
                    sx = max(0, min(int(cx - (tw / 2 - tx) / scale), w - 1))
                    if alpha[sy, sx] > 0:
                        canonical[ty, tx] = belly_rgba[sy, sx]
                    else:
                        canonical[ty, tx, 3] = 0
    return canonical

from model import create_model

def predict(image_path, model_path='models/best_model.pth', device='cuda'):
    model = create_model(arch='unet', encoder='efficientnet-b3', pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    image = cv2.imread(image_path)
    if image is None: return None
    original = image.copy()
    h_orig, w_orig = image.shape[:2]
    
    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(np.transpose(image_resized.astype(np.float32)/255.0, (2,0,1))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()
        
    mask = (mask > THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_resized = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    mask_final, contour = keep_largest_contour(mask_resized)
    return {'original': original, 'mask': mask_final, 'contour': contour}
