"""
Run this script to resize images and points and save to folder.
Pad images to same dimension, do not change width:height
also change point positions
Width:Height of spine images are 1:3, approximately
A typical pair would be 500:1500
"""

import numpy as np
import glob
import os.path as path
import scipy.io
import matplotlib.pyplot as plt
import os
import utils.bi_resize as br
import cv2
import folders as f
import argparse
import shutil
import fliplr_and_points

def resize_save(dst_wh, img_folder, mat_folder, save_img_folder, save_label_folder, plot=False):
    """
    Select images and labels to resize
    :param img_folder:
    :param mat_folder:
    :param save_img_folder:
    :param save_label_folder:
    :return:
    """
    # List of paths
    img_list = glob.glob(path.join(img_folder, "*.jpg"))
    # List of name
    img_name_list = [path.basename(p) for p in img_list]

    for img_name in img_name_list:
        img_path = path.join(img_folder, img_name)
        cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_image, rec = br.resize_img(cv_img, dst_wh)

        save_img_path = path.join(save_img_folder, img_name)
        cv2.imwrite(save_img_path, resized_image)
        print(save_img_path)

        if mat_folder is not None:
            mat_path = path.join(mat_folder, img_name)
            mat = scipy.io.loadmat(mat_path)["p2"]  # [[x1,y1],[x2,y2]...]

            resized_pts = [br.resize_pt(xy, rec) for xy in mat]
            resized_pts = np.array(resized_pts, dtype=np.int)

            if plot:
                plot_image(resized_image, resized_pts)

            save_label_path = path.join(save_label_folder, img_name)
            np.save(save_label_path, resized_pts)



def plot_image(img, mat):
    # Plot
    x_list, y_list = list(zip(*mat))  #[[x,x,...][y,y,...]]
    # plt.style.use('grayscale')
    plt.imshow(img, cmap='gray')
    plt.scatter(x_list, y_list, color='yellow', s=10)
    for i in range(len(x_list)):
        plt.annotate(i, (x_list[i], y_list[i]), color='yellow', size=5)
    plt.axis("off")
    if not path.isdir("plotted_fig"):
        os.mkdir("plotted_fig")
    plt.show()
    plt.clf()

# def cut(img_folder):

def less_head(img_folder):
    # Crop top and bottom area
    file_list = glob.glob(path.join(img_folder, "*.jpg"))
    for file in file_list:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        h = img.shape[0]
        head_area = int(0.15 * h)
        leg_area = int(0.15 * h)
        img = img[head_area:, :]
        img = img[:-leg_area, :]
        cv2.imwrite(file, img)

def crop(img_folder):
    # Crop an image an save to it's original place
    file_list = glob.glob(path.join(img_folder, "*.jpg"))
    for file in file_list:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img[88: 88+752, 32: 32+256]
        cv2.imwrite(file, img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean:
        print("Remove all generated folders")
        list(map(lambda x: shutil.rmtree(x, ignore_errors=True),
                 [f.resize_train_img, f.resize_train_label,
                  f.resize_test_img, f.resize_test_label,
                  f.resize_submit_test_img, f.resize_trainval_img,
                  f.resize_trainval_label, f.train_img_flip, f.train_mat_flip,
                  f.val_img_flip, f.val_mat_flip]))

    [os.makedirs(f, exist_ok=True) for f in [f.resize_train_img, f.resize_train_label,
                                             f.resize_test_img, f.resize_test_label,
                                             f.resize_submit_test_img, f.resize_trainval_img,
                                             f.resize_trainval_label]]

    # Set (256, 752) to be able to divide by 16
    # Resize, crop submit test images
    # resize_save((384, 1120), f.submit_test_img, None, f.resize_submit_test_img, None)  # Was (320, 928)
    resize_save((384, 1120), f.submit_test_trim_images, None, f.resize_submit_test_img, None)  # Was (320, 928)

    print("flip lr")
    fliplr_and_points.main()

    # less_head(f.resize_submit_test_img)  # For angle_net
    # crop(f.resize_submit_test_img)  # For angle_net
    # Train-val folder for final training
    resize_save((256, 752), f.train_img, f.train_mat, f.resize_trainval_img, f.resize_trainval_label)
    resize_save((256, 752), f.val_img, f.val_mat, f.resize_trainval_img, f.resize_trainval_label)
    resize_save((256, 752), f.train_img_flip, f.train_mat_flip, f.resize_trainval_img, f.resize_trainval_label)
    resize_save((256, 752), f.val_img_flip, f.val_mat_flip, f.resize_trainval_img, f.resize_trainval_label)


    # Original training images
    resize_save((256, 752), f.train_img, f.train_mat,
                f.resize_train_img, f.resize_train_label)
    # Flipped training images
    resize_save((256, 752), f.train_img_flip, f.train_mat_flip,
                f.resize_train_img, f.resize_train_label)
    # Test images
    resize_save((256, 752), f.val_img, f.val_mat,
                f.resize_test_img, f.resize_test_label)
