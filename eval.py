import numpy as np
import spine_model
import torch
import os
import os.path as path
import glob
import folders as f
import cv2
import torch.nn.functional as F

def plot_validation_set():
    # Validation Set
    plot_test_images(f.resize_test_img, f.validation_plot_out)

def plot_submit_test_set():
    # Test set to be submitted
    plot_test_images(f.resize_submit_test_img, f.submit_test_plot_out)

def plot_test_images(img_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    test_imgs = glob.glob(path.join(img_folder, '*'))  # Wildcard of test images
    device = torch.device("cuda")  # CUDA
    net = spine_model.SpineModelPAF()  # Spine Network Model
    net.eval()
    net.cuda()
    save_path = path.join(f.checkpoint, "checkpoint.pth")
    if path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("Model loaded")
    else:
        print("No checkpoint.pth at %s", save_path)

    for img_path in test_imgs:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # HW
        img = [[img_gray]]  # NCHW
        img = np.asarray(img, np.float32)
        img_01 = img / 255.0
        test_imgs_tensor = torch.from_numpy(img_01).to(device)
        out_pcm, out_paf, _, _ = net(test_imgs_tensor)  # NCHW

        # Plot and save image
        heats = torch.cat([out_pcm, out_paf], dim=1)
        heats = F.interpolate(heats, size=(test_imgs_tensor.size(2), test_imgs_tensor.size(3)), mode="bilinear")
        np_heats = heats.detach().cpu().numpy()  # NCHW [0,1]
        np_heats = np.clip(np_heats, 0., 1.)[0]  # Remove dim 'N'
        np_heats = np.transpose(np_heats, (1, 2, 0))  # HWC [0,1]

        # Plot on image
        # RGB: Blue, Yellow, Cyan, Magenta, Red, Lime, Green
        colors = np.array([(0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,0,0), (0,255,0), (0,128,0)], np.float32)
        bgr_colors = colors[:, ::-1]  # [Channel][Color]
        np_heats = np_heats[..., np.newaxis]  # HW[Channel][Color] [0,1]
        color_heats = np_heats * bgr_colors  # HW[Channel][Color]
        img_bgr = np.asarray(img_gray, np.float32)[..., np.newaxis][..., np.newaxis]  # [H][W][Ch][Co] [0,255]

        img_heats = (img_bgr / 2.) + (color_heats / 2.)
        ch_HWCo = np.split(img_heats, img_heats.shape[2], axis=2)  # CH [H W 1 CO]
        ch_HWCo = [np.squeeze(HW1Co, axis=2) for HW1Co in ch_HWCo]  # CH [H W CO]

        lt_rt_img = np.amax(ch_HWCo[0:2], axis=0)
        lb_rb_img = np.amax(ch_HWCo[2:4], axis=0)
        lc_rc_img = np.amax(ch_HWCo[4:6], axis=0)
        paf_img = ch_HWCo[6]
        img_bgr = img_bgr[:,:,0,:] * np.ones([3])  # Expand color channels 1->3
        grid_image = np.concatenate([img_bgr, lt_rt_img, lb_rb_img, lc_rc_img, paf_img], axis=1)  # Concat to Width dim, H W C
        grid_image = grid_image.astype(np.uint8)
        img_name = path.basename(img_path)
        cv2.imwrite(path.join(out_folder, img_name), grid_image)
        print(img_name)
        # cv2.imshow("image", grid_image)
        # cv2.waitKey()

if __name__ == '__main__':
    plot_submit_test_set()