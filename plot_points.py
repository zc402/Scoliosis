"""
Run this script to plot keypoints on training images, and save to folder.
"""
import numpy as np
import glob
import os.path as path
import scipy.io
import matplotlib.pyplot as plt
import os
import folders as f


if __name__ == '__main__':

    [os.makedirs(p, exist_ok=True) for p in [f.plot]]

    def plot(img_folder, mat_folder):
        # List of paths
        train_img_list = glob.glob(path.join(img_folder, "*"))
        # List of name
        train_img_name_list = [path.basename(p) for p in train_img_list]

        for train_img_name in train_img_name_list:
            train_img_path = path.join(img_folder, train_img_name)
            train_mat_path = path.join(mat_folder, train_img_name)
            mat = scipy.io.loadmat(train_mat_path)["p2"]  # [[x1,y1],[x2,y2]...]
            # print(mat)
            x_list, y_list = list(zip(*mat))  # [[x,x,...][y,y,...]]
            # Plot
            plt.style.use('grayscale')
            plt_img = plt.imread(train_img_path)
            plt.imshow(plt_img)
            plt.scatter(x_list, y_list, color='yellow', s=10)
            for i in range(len(x_list)):
                plt.annotate(i, (x_list[i], y_list[i]), color='red', size=5)
            plt.axis("off")

            plt.savefig(path.join(f.plot, train_img_name), dpi=300)
            plt.clf()
            print(path.join(f.plot, train_img_name))
            # plt.show()

    # plot(f.train_img_flip, f.train_mat_flip)
    # plot(f.train_img, f.train_mat)
    plot(f.val_img, f.val_mat)
