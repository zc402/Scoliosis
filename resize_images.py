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
    img_list = glob.glob(path.join(img_folder, "*"))
    # List of name
    img_name_list = [path.basename(p) for p in img_list]

    for img_name in img_name_list:
        img_path = path.join(img_folder, img_name)
        mat_path = path.join(mat_folder, img_name)
        mat = scipy.io.loadmat(mat_path)["p2"]  # [[x1,y1],[x2,y2]...]
        cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        resized_image, rec = br.resize_img(cv_img, dst_wh)
        resized_pts = [br.resize_pt(xy, rec) for xy in mat]
        resized_pts = np.array(resized_pts, dtype=np.int)

        if plot:
            plot_image(resized_image, resized_pts)

        save_img_path = path.join(save_img_folder, img_name)
        save_label_path = path.join(save_label_folder, img_name)

        cv2.imwrite(save_img_path, resized_image)
        np.save(save_label_path, resized_pts)
        print(img_name)


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

if __name__ == "__main__":
    # Folder
    train_img_folder = path.join("data_spine", "image", "training")
    train_mat_folder = path.join("data_spine", "labels", "training")
    test_img_folder = path.join("data_spine", "image", "test")
    test_mat_folder = path.join("data_spine", "labels", "test")

    save_train_img_folder = path.join("resized_data", "image", "training")
    save_train_label_folder = path.join("resized_data", "labels", "training")
    save_test_img_folder = path.join("resized_data", "image", "test")
    save_test_label_folder = path.join("resized_data", "labels", "test")

    # Set (256, 752) to be able to divide by 16
    resize_save((256, 752), train_img_folder, train_mat_folder,
                save_train_img_folder, save_train_label_folder)
    resize_save((256, 752), test_img_folder, test_mat_folder,
                save_test_img_folder, save_test_label_folder)
