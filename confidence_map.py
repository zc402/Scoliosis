"""
Create 2D Gaussian map from keypoints
Run this script to show the created maps
"""
import numpy as np
import torch
import cv2
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
    def __init__(self, sigma=4.0, thickness=6):
        self.sigma = sigma
        self.thickness = thickness

    def _gaussian_2d_torch(self, hw, pt):
        """
        Create an image with 1 gaussian circle
        :param hw:
        :param pt:
        :return:
        """
        # Use cuda for big pictures (256:752), use CPU for smaller ones (64: 188)
        # device = torch.device("cuda:0")  # small img: 9.05
        device = torch.device("cuda")  # small img: 3.35
        assert len(hw) == 2  # grayscale image: H,W
        h, w = hw
        i, j = torch.meshgrid([torch.arange(h, dtype=torch.float, device=device), torch.arange(w, dtype=torch.float, device=device)])
        i_d, j_d = i-pt[1], j-pt[0]
        d_square = (i_d*i_d + j_d*j_d)
        g = torch.exp((-d_square / (2.0*self.sigma**2)))
        return g


    def _gaussian_2d_pts_torch(self, hw, pts):
        """
        Create an image with multiple gaussian circles
        :param img:
        :param pts: A list of points [pts][xy]
        :return:
        """

        maps = [self._gaussian_2d_torch(hw, pt) for pt in pts]  # Multiple maps with 1 gaussian on each of them
        maps = torch.stack(maps)
        cmap = torch.max(maps, dim=0)

        return cmap[0].cpu().numpy()

    def _batch_gaussian(self, hw, pts):
        """
        Create a batch of gaussian images
        :param hw:
        :param pts:
        :return: List of gaussian images, range: [0,1]
        """
        b = [self._gaussian_2d_pts_torch(hw, p) for p in pts]
        return b

    def _split_labels_by_corner(self, batch_labels):
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
        pts_corner = self._split_labels_by_corner(pts)  # CNJO, C for corner, J for joint, O for coordinate xy
        CNHW = [self._batch_gaussian(hw, pts) for pts in pts_corner]
        NCHW = np.asarray(CNHW).transpose([1, 0, 2, 3])
        return NCHW

    def _find_LCenter_RCenter(self, batch_labels):
        """
        Find two centers: center of left top and left bottom, center of right top and right bottom
        :param batch_labels:
        :return: l_center r_center [lr][N][17(joint)][xy]
        """
        # [4(tl tr bl br)][N][17(joint)][xy]
        four_corner = np.asarray(self._split_labels_by_corner(batch_labels))
        l_center = (four_corner[0, ...] + four_corner[2, ...]) / 2.  # N J xy
        r_center = (four_corner[1, ...] + four_corner[3, ...]) / 2.
        return l_center, r_center

    def batch_gaussian_LRCenter(self, imgs, pts, zoom):
        """
        Generate gaussian images of L R Center
        :return:
        """
        hw = np.asarray(np.asarray(imgs).shape[1:3])
        if np.all(hw % zoom) == 0:
            hw = hw // zoom
        else:
            raise RuntimeError("Image size can not be divided by %d" % zoom)
        pts = np.array(pts) / zoom

        pts_centers = self._find_LCenter_RCenter(pts)  # CNJO, C for lr centers, J for joint, O for coordinate xy
        CNHW = [self._batch_gaussian(hw, pts) for pts in pts_centers]
        NCHW = np.asarray(CNHW).transpose([1, 0, 2, 3])
        return NCHW

    def _lines_on_img(self, hw, l_cs, r_cs):
        paf_img = np.zeros(hw, dtype=np.uint8)

        lr_cs = zip(l_cs, r_cs)
        [cv2.line(paf_img, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), 255, self.thickness) for p1, p2 in lr_cs]
        # Convert to [0,1]
        paf_img = paf_img.astype(np.float32)
        paf_img = paf_img / 255.
        return paf_img

    def batch_lines(self, heat_hw, l_pts, r_pts):
        l_pts, r_pts = np.array(l_pts), np.array(r_pts)
        heat_hw = np.array(heat_hw)
        assert l_pts.shape[0] == r_pts.shape[0]
        assert len(heat_hw) == 2
        paf_imgs = []  # NHW
        for i in range(l_pts.shape[0]):
            paf_img = self._lines_on_img(heat_hw, l_pts[i], r_pts[i])
            paf_imgs.append(paf_img)
        paf_imgs = np.asarray(paf_imgs)[:, np.newaxis]  # NCHW
        return paf_imgs

    def batch_lines_LRTop(self, heat_hw, batch_labels):
        """
        Draw part affinity field for Left Right Top centers
        :param batch_labels:
        :return: pafs shape: NCHW
        """
        batch_labels = np.array(batch_labels)
        lps = batch_labels[:, 0::4, :]
        rps = batch_labels[:, 1::4, :]
        pafs = self.batch_lines(heat_hw, lps, rps)
        return pafs

    def batch_lines_LRBottom(self, heat_hw, batch_labels):
        batch_labels = np.array(batch_labels)
        lps = batch_labels[:, 2::4, :]
        rps = batch_labels[:, 3::4, :]
        pafs = self.batch_lines(heat_hw, lps, rps)
        return pafs

    def batch_lines_LRCenter(self, heat_hw, pts, zoom):
        """
        Draw Part Affinity Fields (no direction, 1 dim) between each 2 center points.
        :param imgs:
        :param pts:
        :param zoom:
        :return:
        """
        hw = np.array(heat_hw)
        if np.all(hw % zoom) == 0:
            hw = hw // zoom
        else:
            raise RuntimeError("Image size can not be divided by %d" % zoom)
        pts = np.array(pts) / zoom
        l_bcs, r_bcs = self._find_LCenter_RCenter(pts)  # [N][17][xy]
        return self.batch_lines(hw, l_bcs, r_bcs)

    def batch_gaussian_first_lrpt(self, imgs, batch_labels):

        imgs = np.asarray(imgs)
        assert len(imgs.shape) == 3, "(N, h, w)"
        batch_labels = np.array(batch_labels)

        four_corner = np.asarray(self._split_labels_by_corner(batch_labels))
        first_lrpt = four_corner[0:2, :, 0:1, :]  # [tl tr][batch][Joint][xy]
        # first_lrpt = np.transpose(first_lrpt, [1, 0, 2])  # [batch][tl tr][xy]

        lrNHW = np.array([self._batch_gaussian(imgs.shape[1:3], pts) for pts in first_lrpt])
        NHW = np.max(lrNHW, axis=0)

        NCHW = NHW[:, np.newaxis, :, :]  # NCHW.
        return NCHW

    def batch_gaussian_last_lrpt(self, imgs, batch_labels):
        # [4(tl tr bl br)][batch][17(joint)][xy]
        imgs = np.asarray(imgs)
        assert len(imgs.shape) == 3, "(N, h, w)"
        batch_labels = np.array(batch_labels)

        def find_max_Y_index(pts):
            assert len(pts.shape) == 3
            pts_Y = pts[:, :, 1]
            sorted_Y_ind = np.argsort(pts_Y)
            max_Y_ind = sorted_Y_ind[:, -1]
            return max_Y_ind

        left_pts = batch_labels[:, 0::2, :]
        right_pts = batch_labels[:, 1::2, :]

        left_indices = find_max_Y_index(left_pts)
        right_indices = find_max_Y_index(right_pts)  # indices on batch

        l_max = np.array([left_pts[b, ind, :] for b, ind in enumerate(left_indices)])
        r_max = np.array([right_pts[b, ind, :] for b, ind in enumerate(right_indices)])

        lrpt = np.stack([l_max, r_max], axis=0)[:, :, np.newaxis, :]  # [tl tr][batch][Joint][xy]
        lrNHW = np.array([self._batch_gaussian(imgs.shape[1:3], pts) for pts in lrpt])
        NHW = np.max(lrNHW, axis=0)
        NCHW = NHW[:, np.newaxis, :, :]
        return NCHW

    def batch_spine_mask(self, heat_hw, batch_labels):
        batch_labels = np.asarray(batch_labels)
        assert len(heat_hw) == 2
        assert len(batch_labels.shape) == 3, "(N, P, xy)"
        def draw_polygon(hw, labels):
            assert len(hw) == 2, "(h, w)"
            assert len(labels.shape) == 2, "(P, xy)"
            mask = np.zeros(hw, dtype=np.uint8)
            for p in range(0, labels.shape[0], 4):
                p1234 = labels[p:p+4, :].astype(np.int32)
                p1243 = np.stack([p1234[0, :], p1234[1, :], p1234[3, :], p1234[2, :]],
                                 axis=0)
                cv2.fillPoly(mask, [p1243], 255)
            return mask
        batch_mask = [draw_polygon(heat_hw, labels) for labels in batch_labels[:]]
        batch_mask = np.asarray(batch_mask, np.float32)
        batch_mask = batch_mask / 255.
        batch_mask = batch_mask[:, np.newaxis, :, :]
        return batch_mask

    def batch_spine_mask_top3(self, heat_hw, batch_labels):
        pass



def main():
    import load_utils
    import cv2
    import time
    train_data_loader = load_utils.train_loader(10)
    train_imgs, train_labels = next(train_data_loader)
    ts = time.time()
    cm = ConfidenceMap()
    heat_scale = 1
    heat_hw = np.asarray(train_imgs).shape[1:3]
    NCHW_corner_gau = cm.batch_gaussian_split_corner(train_imgs, train_labels, heat_scale)
    NCHW_center_gau = cm.batch_gaussian_LRCenter(train_imgs, train_labels, heat_scale)
    NCHW_c_lines = cm.batch_lines_LRCenter(heat_hw, train_labels, heat_scale)
    NCHW_t_lines = cm.batch_lines_LRTop(heat_hw, train_labels)
    NCHW_b_lines = cm.batch_lines_LRBottom(heat_hw, train_labels)
    NCHW_spine_mask = cm.batch_spine_mask(heat_hw, train_labels)
    NCHW_first_lrpt = cm.batch_gaussian_first_lrpt(train_imgs, train_labels)
    NCHW_last_lrpt = cm.batch_gaussian_last_lrpt(train_imgs, train_labels)
    NCHW_gaussian = np.concatenate((NCHW_first_lrpt, NCHW_last_lrpt, NCHW_spine_mask), axis=1)#NCHW_corner_gau, NCHW_center_gau, NCHW_lines, NCHW_first_lrpt), axis=1)
    te = time.time()
    print("Duration for gaussians: %f" % (te-ts))  # Time duration for generating gaussians
    for n in range(NCHW_gaussian.shape[0]):
        for c in range(NCHW_gaussian.shape[1]):
            assert NCHW_gaussian.max() < 1.5, "expect normalized values"
            cv2.imshow("Image", train_imgs[n])
            g = NCHW_gaussian[n, c]
            g = cv2.resize(g, dsize=None, fx=heat_scale, fy=heat_scale)
            cv2.imshow("Image Heat", np.amax([train_imgs[n].astype(np.float32)/255, g], axis=0))
            cv2.imshow("Heat Only", g)
            cv2.waitKey()

if __name__ == "__main__":
    main()
