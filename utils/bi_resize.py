"""
This file contains image resize function.
It resize an image to a specific size, then resize it back after detection.
"""
import cv2
import sys
import numpy as np

if sys.version_info[0] < 3:
    raise RuntimeError("Requires Python 3")


def resize_img(src, dst_wh):
    """
    Resize opencv img src to dst, ratio kept
    Grayscale version
    :param src:
    :param dst_wh:
    :return: resized_image, record
    """
    # If not HWC image, quit
    if len(src.shape) != 2:
        raise ValueError("scr is not gray scale")

    sh = src.shape[0]
    sw = src.shape[1]
    dh = dst_wh[1]
    dw = dst_wh[0]
    # ratio: W/H
    ratio_src = sw / sh
    ratio_dst = dw / dh

    if ratio_src >= ratio_dst:
        # resize by W
        resize_ratio = dw / sw
        nw = dw
        nh = int(sh * resize_ratio)
    else:
        # resize by H
        resize_ratio = dh / sh
        nw = int(sw * resize_ratio)
        nh = dh

    resized_img = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_CUBIC)
    black = np.zeros([dh, dw], dtype=np.uint8)
    left = (dw-nw)//2
    top = (dh-nh)//2
    black[top: top+nh, left: left+nw] = resized_img[...]
    result = black

    resize_record = (left, top, resize_ratio)
    return result, resize_record


def resize_pt(xy, resize_record):
    """Compute resized point on image, point must be integer"""
    left, top, ratio = resize_record
    nx = xy[0] * ratio + left
    ny = xy[1] * ratio + top
    return nx, ny


def reverse(xy, resize_record):
    """inverse resize apply to points (X,Y)"""
    left, top, ratio = resize_record

    nx = xy[0] - left
    nx = nx / ratio

    ny = xy[1] - top
    ny = ny / ratio
    return nx, ny