"""
Run this script to plot keypoints on training images, and save to folder.
"""
import numpy as np
import glob
import os.path as path
import scipy.io
import matplotlib.pyplot as plt
import os


# Folder
train_img_folder = path.join("data_spine", "image", "training")
train_mat_folder = path.join("data_spine", "labels", "training")
# List of paths
train_img_list = glob.glob(path.join(train_img_folder, "*"))
# List of name
train_img_name_list = [path.basename(p) for p in train_img_list]

for train_img_name in train_img_name_list:
    train_img_path = path.join(train_img_folder, train_img_name)
    train_mat_path = path.join(train_mat_folder, train_img_name)
    mat = scipy.io.loadmat(train_mat_path)["p2"]  # [[x1,y1],[x2,y2]...]
    print(mat)
    x_list, y_list = list(zip(*mat))  #[[x,x,...][y,y,...]]
    # Plot
    plt.style.use('grayscale')
    plt_img = plt.imread(train_img_path)
    plt.imshow(plt_img)
    plt.scatter(x_list, y_list, color='yellow', s=10)
    for i in range(len(x_list)):
        plt.annotate(i, (x_list[i], y_list[i]), color='yellow', size=5)
    plt.axis("off")
    if not path.isdir("plotted_fig"):
        os.mkdir("plotted_fig")
    plt.savefig(path.join("plotted_fig", train_img_name), dpi=300)
    plt.clf()
    # plt.show()

