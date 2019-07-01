"""
Load images and corresponding npy labels from file.
"""
import numpy as np
import os.path as path
import glob
import os
import random
import cv2
import folders as f


def load_imgs_labels(batch_size, label_folder, img_folder, rand):
    """
    Internal generator for loading train or test data
    :param batch_size:
    :param label_folder:
    :param img_folder:
    :return: imgs, labels
    """
    label_list = glob.glob(path.join(label_folder, "*"))
    total_size = len(label_list)
    loop_range = total_size - (total_size % batch_size)
    while True:
        if rand:
            random.shuffle(label_list)
        for i in range(0, loop_range, batch_size):
            batch_label_path = label_list[i:i+batch_size]
            batch_label = [np.load(j) for j in batch_label_path]
            # label contains .npy, use splitext to delete it.
            batch_img_name = [path.splitext(path.basename(j))[0] for j in batch_label_path]
            batch_img_path = [path.join(img_folder, name) for name in batch_img_name]
            batch_img = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in batch_img_path]
            yield batch_img, batch_label

def train_loader(batch_size):
    """
    Training data generator
    :param batch_size:
    :return: batch_img, batch_label
    """
    img_folder = f.resize_train_img
    label_folder = f.resize_train_label
    loader = load_imgs_labels(batch_size, label_folder, img_folder, rand=True)
    for img_la in loader:
        yield img_la


def test_loader(batch_size):
    """
    Test data generator
    :param batch_size:
    :return: batch_img, batch_label
    """
    img_folder = f.resize_test_img
    label_folder = f.resize_test_label
    loader = load_imgs_labels(batch_size, label_folder, img_folder, rand=False)
    for img_la in loader:
        yield img_la
