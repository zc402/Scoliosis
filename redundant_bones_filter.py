import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import box_crop
import cv2

def _get_filtered_pairs(pair_lr_value, evidences, low, high):
    # delete redundant pairs
    # evidences: according to what to delete pairs. e.g. length array, x value array etc.
    remain_indices = []
    for i, x in enumerate(evidences):
        if x > low and x < high:
            remain_indices.append(i)
    remain_indices = np.array(remain_indices)
    pair_lr_value = pair_lr_value[:, remain_indices,:]
    return pair_lr_value

def filter(pair_lr_value):
    delete_pairs = True
    assert len(pair_lr_value.shape) == 3
    # TODO: Slope change

    result_dict = {}
    # ------------------------------------------------
    # Crop by x value of mid points
    # Midpoints
    hmids = (pair_lr_value[0] + pair_lr_value[1]) / 2  # Horizontal mid points, [p][xy]
    hmids_x = hmids[:, 0]  # [p] Midpoint x

    limit_factor = 2
    m = np.mean(hmids_x)
    dev = np.std(hmids_x)
    x_low = m - limit_factor * dev
    x_high = m + limit_factor * dev
    # print("mean: ", m)
    # print("deviation: ", dev)
    # print("+-{} dev: ".format(limit_factor), x_low*dev, x_high)
    # plt.hist(hmids_x, bins=20)
    result_dict["x_low"] = x_low
    result_dict["x_high"] = x_high
    # delete from pairs
    if delete_pairs:
        pair_lr_value = _get_filtered_pairs(pair_lr_value, hmids_x, x_low, x_high)
    # print(pair_lr_value.shape[1])

    # -----------------------------------------------------
    # Crop by length of bones


    limit_factor = 3
    bones = pair_lr_value[1] - pair_lr_value[0]  # [p][xy]
    lens = np.linalg.norm(bones, axis=-1)

    m = np.mean(lens)
    dev = np.std(lens)
    len_low = m - limit_factor * dev
    len_high = m + limit_factor * dev
    # print("mean: ", m)
    # print("deviation: ", dev)
    # print("+-{} dev: ".format(limit_factor), m-limit_factor*dev, m+limit_factor*dev)
    result_dict["len_low"] = len_low
    result_dict["len_high"] = len_high
    # plt.hist(lens, bins=30)

    # delete from pairs
    if delete_pairs:
        pair_lr_value = _get_filtered_pairs(pair_lr_value, lens, len_low, len_high)
    # print(pair_lr_value.shape[1])

    # -----------------------------------------------------
    # Delete first/ last bone or not?
    # Crop by NEAREST Y INTERVAL (must proceed after other standards)
    # Suppose pair_lr_value is sorted by y
    assert pair_lr_value.shape[1] > 4, "not enough bones to sample and to trim first/last one"

    num_del = -1
    while num_del != 0:  # do until no more crops

        hmids = (pair_lr_value[0] + pair_lr_value[1]) / 2  # Horizontal mid points, [p][xy]
        hmids_y = hmids[:, 1]  # [p] Midpoint y
        intervals = hmids_y[2:-1] - hmids_y[1:-2]  # 1-0, 2-1, 3-2... 19-18

        limit_factor = 3
        m = np.mean(intervals)
        dev = np.std(intervals)

        int_high = m + limit_factor * dev
        # print("+-{} dev: ".format(limit_factor), int_high)

        result_dict["int_high"] = int_high
        # plt.hist(intervals, bins=30)

        first_bone_int = hmids_y[1] - hmids_y[0]
        last_bone_int = hmids_y[-1] - hmids_y[-2]
        # print("first/last", first_bone_int, last_bone_int)
        # delete from pairs
        prev_length = pair_lr_value.shape[1]
        if delete_pairs:
            if first_bone_int > int_high:
                pair_lr_value = pair_lr_value[:, 1:, :]
            if last_bone_int > int_high:
                pair_lr_value = pair_lr_value[:, :-1, :]

        current_length = pair_lr_value.shape[1]
        num_del = prev_length - current_length
        if num_del < 0:
            raise ValueError()

    # -----------------------------------------
    # If still more than 17, reduce redundant pairs from TOP
    if pair_lr_value.shape[1] > 16:
        pair_lr_value = pair_lr_value[:, -16:, :]
    result_dict["pair_lr_value"] = pair_lr_value
    return result_dict

def simple_filter(pair_lr_value):
    # delete y < 190
    hmids = (pair_lr_value[0] + pair_lr_value[1]) / 2  # Horizontal mid points, [p][xy]
    hmids_y = hmids[:, 1]  # [p] Midpoint y
    pair_lr_value = _get_filtered_pairs(pair_lr_value, hmids_y, 190, 1120)
    # Keep index 0 ~ 17
    if pair_lr_value.shape[1] > 18:
        pass
        # pair_lr_value = pair_lr_value[:, :18, :]
    return pair_lr_value

class BoxNetFilter():
    def __init__(self):
        self.box = box_crop.Box()

    def filter(self, pair_lr_value, image):
        assert len(image.shape)==2
        h, w = image.shape
        h, w = float(h), float(w)
        hmids = (pair_lr_value[0] + pair_lr_value[1]) / 2  # Horizontal mid points, [p][xy]
        hmids_x = hmids[:, 0]  # [p] Midpoint x
        box = self.box
        target_w = 256.
        zoom_rate = target_w / w
        target_h = h * zoom_rate
        zoom_img_gray = cv2.resize(image, dsize=(int(target_w), int(target_h)), interpolation=cv2.INTER_CUBIC)

        x_min, x_max, y_min, y_max = box.predict_box(zoom_img_gray)
        x_left = int(w * x_min)
        x_right = int(w * x_max)
        pair_lr_value = _get_filtered_pairs(pair_lr_value, hmids_x, x_left, x_right)
        return pair_lr_value

