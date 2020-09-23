"""
Written by Matteo Dunnhofer - 2020

Utility functions
"""
import numpy as np


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def cxcywh2xyxy(bbox):
    x1 = bbox[0] - (bbox[2] / 2)
    y1 = bbox[1] - (bbox[3] / 2)

    return [x1, y1, x1 + bbox[2], y1 + bbox[3]]

def xywh2xyxy(bbox):
    x1 = bbox[0]
    y1 = bbox[1]

    return [x1, y1, x1 + bbox[2], y1 + bbox[3]]

def xyxy2cxcywh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + (w / 2)
    y = bbox[1] + (h / 2)

    return [x, y, w, h]

def xyxy2xywh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0]
    y = bbox[1]

    return [x, y, w, h]


def clip_bb(bbox, image):
    x = clip(bbox[0], 0, image.size[0])
    y = clip(bbox[1], 0, image.size[1])
    w = clip(bbox[2], 0, image.size[0])
    h = clip(bbox[3], 0, image.size[1])

    bbox = np.array([x, y, w, h])

    return bbox

def get_crop_bb(bb, img_w, img_h, k):
    # compute the coordinates of a padded crop

    w = clip(bb[2], 0, img_w)
    h = clip(bb[3], 0, img_h)

    cx = clip(bb[0] + (w / 2), 0, img_w)
    cy = clip(bb[1] + (h / 2), 0, img_h)

    new_w = k * w
    new_h = k * h

    new_x1 = cx - (new_w / 2)
    new_y1 = cy - (new_h / 2)
    new_x2 = cx + (new_w / 2)
    new_y2 = cy + (new_h / 2)

    crop = [new_x1, new_y1, new_w, new_h]

    return crop

def denorm_action(action, old_bb):
    # from shift to bounding box coordinates
    norm_coeff = 1.0
    new_x = old_bb[0] + (norm_coeff * action[0] * old_bb[2])
    new_y = old_bb[1] + (norm_coeff * action[1] * old_bb[3])
    new_w = old_bb[2] + (norm_coeff * action[2] * old_bb[2])
    new_h = old_bb[3] + (norm_coeff * action[3] * old_bb[3])

    return [new_x, new_y, new_w, new_h]