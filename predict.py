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
