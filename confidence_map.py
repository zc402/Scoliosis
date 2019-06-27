"""
Create 2D Gaussian map from keypoints
Run this script to show the created maps
"""
import numpy as np
import torch
import torch.nn as nn
"""
def gaussian_2d(img, pt):
    sigma = 5
    assert len(img.shape) == 2  # grayscale image: H,W
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_d, y_d = x-pt[0], y-pt[1]  # Distance of each point on map to keypoint pt
    d_2 = (x_d*x_d + y_d*y_d)  # Square of Straight distance
    g = np.exp(-d_2 / (2.0*sigma**2))
    return g


def gaussian_2d_pts(img, pts):
    maps = [gaussian_2d(img, pt) for pt in pts]  # Multiple maps with 1 gaussian on each of them
    cmap = np.amax(maps, axis=0)  # One map with multiple gaussian circles
    print("1map")
    return cmap
"""

class ConfidenceMap():
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def gaussian_2d_torch(self, hw, pt):
        """
        Create an image with 1 gaussian circle
        :param hw:
        :param pt:
        :return:
        """
        # Use cuda for big pictures (256:752), use CPU for smaller ones (64: 188)
        # device = torch.device("cuda:0")  # small img: 9.05
        device = torch.device("cpu")  # small img: 3.35
        assert len(hw) == 2  # grayscale image: H,W
        h, w = hw
        i, j = torch.meshgrid([torch.arange(h, dtype=torch.float, device=device), torch.arange(w, dtype=torch.float, device=device)])
        i_d, j_d = i-pt[1], j-pt[0]
        d_square = (i_d*i_d + j_d*j_d)
        g = torch.exp((-d_square / (2.0*self.sigma**2)))
        return g


    def gaussian_2d_pts_torch(self, hw, pts):
        """
        Create an image with multiple gaussian circles
        :param img:
        :param pts:
        :return:
        """
        maps = [self.gaussian_2d_torch(hw, pt) for pt in pts]  # Multiple maps with 1 gaussian on each of them
        maps = torch.stack(maps)
        cmap = torch.max(maps, dim=0)

        return cmap[0].cpu().numpy()

    def batch_gaussian(self, hw, pts):
        """
        Create a batch of gaussian images
        :param hw:
        :param pts:
        :return: List of gaussian images, range: [0,1]
        """
        b = [self.gaussian_2d_pts_torch(hw, p) for p in pts]
        return b

    def split_labels_by_corner(self, batch_labels):
        """
        Index of: Top left, Top right, Bottom left, Bottom right
        :param batch_labels: [batch][pts][xy]
        :return: [4(tl tr bl br)][batch][17(joint)][xy]
        """
        batch_labels = np.asarray(batch_labels)
        ind_1 = np.array(list(range(0, 68, 4)))  # 0, 4, 8...
        # [4(tl tr bl br)][N][17(joint)][xy]
        four_corner = [np.take(batch_labels, ind, axis=1).tolist() for ind in (ind_1, ind_1+1, ind_1+2, ind_1+3)]
        return four_corner

    def batch_gaussian_split_corner(self, imgs, pts, zoom):
        """
        Generate gaussian for batch images
        Split four corner to different maps
        :param imgs:
        :param pts:
        :param zoom: size of input/output
        :return: NCHW format gaussian map, C for corner
        """

        hw = np.asarray(np.asarray(imgs).shape[1:3])
        if np.all(hw % zoom) == 0:
            hw = hw // zoom
        else:
            raise RuntimeError("Image size can not be divided by %d" % zoom)
        pts = np.array(pts) / zoom
        pts_corner = self.split_labels_by_corner(pts)  # CNJO, C for corner, J for joint, O for coordinate xy
        CNHW = [self.batch_gaussian(hw, pts) for pts in pts_corner]
        NCHW = np.asarray(CNHW).transpose([1, 0, 2, 3])
        return NCHW


def main():
    import load_utils
    import cv2
    import time
    train_data_loader = load_utils.train_loader(10)
    train_imgs, train_labels = next(train_data_loader)
    ts = time.time()
    NCHW_gaussian = ConfidenceMap().batch_gaussian_split_corner(train_imgs, train_labels, 4)
    te = time.time()
    print("Duration for gaussians: %f" % (te-ts))  # Time duration for generating gaussians
    for n in range(NCHW_gaussian.shape[0]):
        for c in range(NCHW_gaussian.shape[1]):
            cv2.imshow("Gaussian", train_imgs[n])
            g = NCHW_gaussian[n, c]
            g = cv2.resize(g, dsize=None, fx=4, fy=4)
            cv2.imshow("Image", np.amax([train_imgs[n].astype(np.float32)/255, g], axis=0))
            cv2.waitKey()

if __name__ == "__main__":
    main()