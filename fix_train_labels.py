import numpy as np
import os
import os.path as path
import glob
import scipy.io
import cv2
import folders as f

# Convert manually annotated training labels to mat format
npy_list = glob.glob(path.join(f.manual_fix_train, "*.npy"))
for npy_path in npy_list:
    img_path = path.join(f.train_img, path.basename(npy_path)[:-4] + ".jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hw = img.shape[:2]
    npy = np.load(npy_path)
    npy_ori = npy * [hw[1], hw[0]]
    mat_recover = {"p2": npy_ori}
    # aaa.jpg.mat
    scipy.io.savemat(path.join(f.manual_fix_train, path.basename(npy_path)[:-4] + ".jpg.mat"), mat_recover)