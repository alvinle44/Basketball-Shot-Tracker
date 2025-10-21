import cv2
import math
import json
import numpy as np
import torch
from ultralytics import YOLO

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def smooth_point(prev, new, alpha=0.7):
    if prev is None:
        return new
    px, py = prev
    nx, ny = new
    return (alpha * px + (1 - alpha) * nx, alpha * py + (1 - alpha) * ny)

def detect_up(ball_traj, rim_box):
    if not ball_traj or not rim_box:
        return False
    (rx1, ry1, rx2, ry2) = rim_box
    x1, y1 = rx1 - (rx2 - rx1) * 2.5, ry1 - (ry2 - ry1) * 2.5
    x2, y2 = rx2 + (rx2 - rx1) * 2.5, ry1 - (ry2 - ry1) * 0.5
    bx, by = ball_traj[-1]
    return (x1 < bx < x2 and y1 < by < y2)


def detect_down(ball_traj, rim_box):
    if not ball_traj or not rim_box:
        return False
    (rx1, ry1, rx2, ry2) = rim_box
    bx, by = ball_traj[-1]
    return by > ry2 + 0.6 * (ry2 - ry1)

def score_prediction(ball_traj, rim_box):
    if len(ball_traj) < 3 or rim_box is None:
        return False
    (rx1, ry1, rx2, ry2) = rim_box
    rim_y = ry1 + 0.3 * (ry2 - ry1)
    x, y = [], []
    for i in reversed(range(len(ball_traj))):
        bx, by = ball_traj[i]
        if by < rim_y:
            x.append(bx)
            y.append(by)
            if i + 1 < len(ball_traj):
                bx2, by2 = ball_traj[i + 1]
                x.append(bx2)
                y.append(by2)
            break
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        pred_x = (rim_y - b) / m
        rim_x1 = (rx1 + rx2) / 2 - 0.35 * (rx2 - rx1)
        rim_x2 = (rx1 + rx2) / 2 + 0.35 * (rx2 - rx1)
        return rim_x1 < pred_x < rim_x2
