"""
If we flip left and right, the left top point will become right top point.
So this can't be done at spine_augmentation.py.
Run this script to fliplr and adjust points, and save to training folder.
"""
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
import numpy as np
import os
import os.path as path
import glob
import scipy.io
import cv2
import folders as f

def flip_lr(img_folder, mat_folder, save_img_folder, save_label_folder):
    # List of path
    img_list = glob.glob(path.join(img_folder, "*"))
    # List of name
    img_name_list = [path.basename(p) for p in img_list]

    for img_name in img_name_list:
        img_basename, ext = path.splitext(img_name)
        img_path = path.join(img_folder, img_name)
        mat_path = path.join(mat_folder, img_name)
        mat = scipy.io.loadmat(mat_path)["p2"]  # [[x1,y1],[x2,y2]...]

        cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_flip, keypoints_flip = iaa.Fliplr(1.0)(images=cv_img, keypoints=mat[np.newaxis])
        keypoints_flip = keypoints_flip[0]  # The output has a batch size of 1
        # Recover left and right. 0->1, 1->0, 2->3, 3->2
        pts_recover = np.zeros_like(keypoints_flip)
        pts_recover[::2, :] = keypoints_flip[1::2, :]
        pts_recover[1::2, :] = keypoints_flip[::2, :]
        # Save mat back
        mat_recover = {"p2":pts_recover}
        flip_name = "%s_flip.jpg" % img_basename
        flip_img_path = path.join(save_img_folder, flip_name)
        flip_mat_path = path.join(save_label_folder, flip_name)
        scipy.io.savemat(flip_mat_path, mat_recover)
        cv2.imwrite(flip_img_path, img_flip)
        print(flip_img_path)


if __name__ == '__main__':

    [os.makedirs(p, exist_ok=True) for p in [f.train_img_flip, f.train_mat_flip, f.val_img_flip, f.val_mat_flip]]
    flip_lr(f.train_img, f.train_mat, f.train_img_flip, f.train_mat_flip)
    flip_lr(f.val_img, f.val_mat, f.val_img_flip, f.val_mat_flip)
