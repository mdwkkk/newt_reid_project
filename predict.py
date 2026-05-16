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
