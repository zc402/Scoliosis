"""This script tells us that there are too many wrong annotations in dataset (around 100)
To keep enough training samples, we'll have to write compatible codes rather than deleting them"""
import numpy as np
import os.path as path
import glob
import os
import random
import cv2
import folders as f
import csv

def select_wrong_annotation(img_folder, label_folder):
    """Find annotations with large indices at top of image"""
    img_paths = glob.glob(path.join(img_folder, "*.jpg"))
    def check_annotation(img_path):
        # Return True for correct annotations, False for wrong annotations.
        label_name = path.basename(img_path) + ".npy"
        label_path = path.join(label_folder, label_name)
        label = np.load(label_path)  # P, xy
        left = label[0::2]
        right = label[1::2]
        for i in range(left.shape[0] - 1):
            if left[i+1, 1] < left[i, 1]:
                return False
            if right[i+1, 1] < right[i, 1]:
                return False
        return True
    wrong_paths = [p if not check_annotation(p) else None for p in img_paths]
    wrong_paths = list(filter(lambda x: x is not None, wrong_paths))
    print(wrong_paths)

if __name__ == "__main__":
    select_wrong_annotation(f.train_img, f.resize_train_label)
