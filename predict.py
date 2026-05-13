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
