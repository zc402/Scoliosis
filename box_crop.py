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
import os

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

        save_path = f.checkpoint_box_trainval_path
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
        # raise NotImplementedError("cobb angle parse will use image height, can't change it now.")
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
        if y_max < 0.7:
            y_max = 0.7
        y_top = int(h * y_min)
        y_bottom = int(h * y_max + 0.05 * h)
        assert y_top < y_bottom
        # use zero to fill height
        img_gray[:y_top, :] = 0
        img_gray[y_bottom:, :] = 0
        return img_gray

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
        # Center after crop
        x_center = (w * x_min + w * x_max) / 2
        expected_w = h / 3
        x_left = max(0, x_center-(expected_w/2))
        x_right = min(w, x_center+(expected_w/2))

        crop_img = img_gray[:, int(x_left): int(x_right)]
        # x_left = int(w * x_min - 0.02 * h)  # Use h, because w becomes wider if original image is wider.
        # x_right = int(w * x_max + 0.02 * h)
        # assert x_left < x_right
        # crop_img = img_gray[:, x_left: x_right]
        return crop_img

    def trim_width_height(self, img_gray):

        assert len(img_gray.shape) == 2, "h, w"
        hw = img_gray.shape
        h, w = float(hw[0]), float(hw[1])
        target_w = 256.
        zoom_rate = target_w / w
        target_h = h * zoom_rate

        zoom_img_gray = cv2.resize(img_gray, dsize=(int(target_w), int(target_h)), interpolation=cv2.INTER_CUBIC)
        box = self.box_predictor.predict_box(zoom_img_gray)
        x_min, x_max, y_min, y_max = box
        x_left = int(w * x_min - 0.03 * w)
        x_right = int(w * x_max + 0.03 * w)
        assert x_left < x_right
        img_gray = img_gray[:, x_left: x_right]

        y_top = int(h * y_min)
        y_bottom = int(h * y_max + 0.05 * h)
        assert y_top < y_bottom
        # use zero to fill height
        img_gray[:y_top, :] = 0
        img_gray[y_bottom:, :] = 0
        return img_gray


def main():
    plot = False
    os.makedirs(f.submit_test_trim_images, exist_ok=True)
    trim_machine = TrimMachine()
    test_imgs = glob.glob(path.join(f.submit_test_img, '*.jpg'))  # Wildcard of test images
    for img_path in test_imgs:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # HW
        basename = path.basename(img_path)
        # Crop width, then crop height might be better, because
        # width crop is easier, and a trimmed width gives more budget to height in a fixed resize ratio (1: 3)
        crop_img = trim_machine.trim_width(img_gray)
        # crop_img = trim_machine.trim_height(img_gray)
        # crop_img = trim_machine.trim_width_height(img_gray)

        if plot:
            crop_img_show = cv2.resize(crop_img, dsize=(256, 752))
            img_gray_show = cv2.resize(img_gray, dsize=(256, 752))
            cv2.imshow("Ori", img_gray_show)
            cv2.imshow("Crop", crop_img_show)
            print(path.basename(img_path))
            cv2.waitKey(0)
        else:
            cv2.imwrite(path.join(f.submit_test_trim_images, basename), crop_img)
        print(basename)

if __name__=="__main__":
    main()