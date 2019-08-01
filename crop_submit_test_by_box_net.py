"""
Crop submit images
1. zoom original image, fix height to 512, let width adjust itself to keep ratio
2. predict box y_min, y_max percentage
"""
import numpy as np
import glob
import torch
import os.path as path
import folders as f
import cv2

class Box():
    def __init__(self):
        import torchvision
        import torch.nn as nn
        import torch
        net = torchvision.models.densenet121()
        num_conv_features = net.features[-1].num_features
        classifier = nn.Sequential(nn.Linear(num_conv_features, 4), nn.Sigmoid())
        net.classifier = classifier
        net.eval().cuda()

        save_path = f.checkpoint_box_path
        if path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            print("Model loaded")
        else:
            raise FileNotFoundError()

        self.net = net

    def predict_box(self, img_gray):
        assert len(img_gray.shape) == 2
        assert np.max(img_gray) > 1.1, "expect uint8 image [0, 255]"

        img = [[img_gray]]  # NCHW
        img = np.asarray(img, np.float32)
        img_01 = img / 255.0
        img_01 = img_01 * np.ones([1, 3, 1, 1], np.float32)
        test_imgs_tensor = torch.from_numpy(img_01).cuda()
        with torch.no_grad():
            pred_box = self.net(test_imgs_tensor)  # NCHW
        pred_box = pred_box.detach().cpu().numpy()
        return pred_box[0]


class TrimMachine():
    def __init__(self):
        self.box_predictor = Box()

    def trim_height(self, img_gray):
        assert len(img_gray.shape) == 2, "h, w"
        hw = img_gray.shape
        h, w = float(hw[0]), float(hw[1])
        # Zoom h to 1120 (set according to training image size)
        target_h = 752.
        zoom_rate = target_h / h
        target_w = w * zoom_rate
        zoom_img_gray = cv2.resize(img_gray, dsize=(int(target_w), int(target_h)), interpolation=cv2.INTER_CUBIC)
        box = self.box_predictor.predict_box(zoom_img_gray)
        _, _, y_min, y_max = box
        y_top = int(h * y_min)
        y_bottom = int(h * y_max)
        assert y_top < y_bottom
        crop_img = img_gray[y_top: y_bottom, :]
        return crop_img

    def trim_width(self, img_gray):
        assert len(img_gray.shape) == 2, "h, w"
        hw = img_gray.shape
        h, w = float(hw[0]), float(hw[1])
        target_w = 256.
        zoom_rate = target_w / w
        target_h = h * zoom_rate

        zoom_img_gray = cv2.resize(img_gray, dsize=(int(target_w), int(target_h)), interpolation=cv2.INTER_CUBIC)
        box = self.box_predictor.predict_box(zoom_img_gray)
        x_min, x_max, _, _ = box
        x_left = int(w * x_min)
        x_right = int(w * x_max)
        assert x_left < x_right
        crop_img = img_gray[:, x_left: x_right]
        return crop_img

def main():
    trim_machine = TrimMachine()
    test_imgs = glob.glob(path.join(f.submit_test_img, '*'))  # Wildcard of test images
    for img_path in test_imgs:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # HW

        crop_img = trim_machine.trim_width(img_gray)

        crop_img_show = cv2.resize(crop_img, dsize=None, fx=0.1, fy=0.1)
        img_gray_show = cv2.resize(img_gray, dsize=None, fx=0.1, fy=0.1)
        cv2.imshow("Ori", img_gray_show)
        cv2.imshow("Crop", crop_img_show)
        cv2.waitKey(0)
        img_gray = crop_img


main()