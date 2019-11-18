import numpy as np
import ladder_shufflenet
import part_affinity_field_net
import torch
import os
import os.path as path
import glob
import folders as f
import cv2
import torch.nn.functional as F
import argparse

def centeroid(heat, gaussian_thresh = 0.5):
    # Parse center point of connected components
    # Return [p][xy]
    ret, heat = cv2.threshold(heat, gaussian_thresh, 1., cv2.THRESH_BINARY)
    heat = np.array(heat * 255., np.uint8)
    # num: point number + 1 background
    num, labels = cv2.connectedComponents(heat)
    coords = []
    for label in range(1, num):
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coords.append([cX, cY])
    return coords

def predict_heatmaps(img_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    test_imgs = glob.glob(path.join(img_folder, '*'))  # Wildcard of test images
    device = torch.device("cuda")  # CUDA
    net = ladder_shufflenet.LadderModelAdd()
    net.eval()
    net.cuda()

    if args.trainval:
        print("Load [train, val] checkpoint")
        save_path = f.checkpoint_heat_trainval_path
    else:
        print("Load [train] checkpoint")
        save_path = f.checkpoint_heat_path

    if path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("Heat Model loaded")
    else:
        raise FileNotFoundError("No checkpoint.pth at %s", save_path)
    print("images to be predict:" + str(test_imgs))
    for img_path in test_imgs:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # HW
        img = [[img_gray]]  # NCHW
        img = np.asarray(img, np.float32)
        img_01 = img / 255.0
        test_imgs_tensor = torch.from_numpy(img_01).to(device)
        with torch.no_grad():
            out_dict = net(test_imgs_tensor)  # NCHW
            out_pcm, out_paf = out_dict["pcm"], out_dict["paf"]

        # Plot and save image (exclude neck)
        heats = torch.cat([test_imgs_tensor, out_pcm[:, 0:6], out_paf], dim=1)
        # heats = F.interpolate(heats, size=(test_imgs_tensor.size(2), test_imgs_tensor.size(3)), mode="bilinear")
        np_heats = heats.detach().cpu().numpy()  # NCHW (0,1)
        np_heats = np.clip(np_heats, 0., 1.)[0]  # Remove dim 'N'
        np_heats = np.transpose(np_heats, (1, 2, 0))  # HWC (0,1)

        # Plot on image
        # 6 corner 1 paf
        # RGB: White(original image), Blue, Yellow, Cyan, Magenta, Red, Lime, Green
        colors = np.array([(255, 255, 255), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,0,0), (0,255,0), (0,128,0)], np.float32)

        bgr_colors = colors[:, ::-1]  # [Channel][Color]
        np_heats_c = np_heats[..., np.newaxis]  # HW[Channel][Color] [0,1]
        # Heat mask
        color_heats = np_heats_c * bgr_colors  # HW[Channel][Color]
        # Image as background
        img_bgr = np.asarray(img_gray, np.float32)[..., np.newaxis][..., np.newaxis]  # [H][W][Ch][Co] (0,255)

        img_heats = (img_bgr / 2.) + (color_heats / 2.)
        ch_HWCo = np.split(img_heats, img_heats.shape[2], axis=2)  # CH [H W 1 CO]
        ch_HWCo = [np.squeeze(HW1Co, axis=2) for HW1Co in ch_HWCo]  # CH [H W CO]

        ori_img = ch_HWCo[0]
        lt_rt_img = np.amax(ch_HWCo[1:3], axis=0)
        lb_rb_img = np.amax(ch_HWCo[3:5], axis=0)
        lc_rc_img = np.amax(ch_HWCo[5:7], axis=0)
        paf_img = ch_HWCo[7]
        # img_bgr = img_bgr[:,:,0,:] * np.ones([3])  # Expand color channels 1->3
        grid_image = np.concatenate([ori_img, lt_rt_img, lb_rb_img, lc_rc_img, paf_img], axis=1)  # Concat to Width dim, H W Color
        grid_image = grid_image.astype(np.uint8)
        img_name = path.basename(img_path)
        cv2.imwrite(path.join(out_folder, img_name), grid_image)
        print(img_name)
        # cv2.imshow("image", grid_image)
        # cv2.waitKey()
        ############################
        # Gaussian to point
        # coord_list shape=(heatmaps, coords, xy)
        coord_list = [centeroid(np_heats[:, :, c]) for c in range(1, np_heats.shape[2])]  # 1 is original image
        img_HWC = img_gray[:, :, np.newaxis] * np.ones((1,1,3))
        for i, coords in enumerate(coord_list):  # Different heatmaps (corners)
            mark_color = bgr_colors[i+1]
            for coord in coords:  # Same kind, different coordinate landmarks
                cv2.circle(img_HWC, center=tuple(coord), radius=3, color=tuple([int(c) for c in mark_color]))
        cv2.imwrite(path.join(out_folder, "7marks_" + img_name), img_HWC)

        coord_list = [centeroid(np_heats[:, :, c]) for c in range(1, 5)]  # 1 is original image
        img_HWC = img_gray[:, :, np.newaxis] * np.ones((1, 1, 3))
        for i, coords in enumerate(coord_list):  # Different heatmaps (corners)
            mark_color = bgr_colors[i + 1]
            for coord in coords:  # Same kind, different coordinate landmarks
                cv2.circle(img_HWC, center=tuple(coord), radius=3, color=tuple([int(c) for c in mark_color]))
        cv2.imwrite(path.join(out_folder, "4marks_" + img_name), img_HWC)




"""
def eval_submit_testset():
    import csv
    result_name_an123 = []  # Parsing results to be wrote
    submit_example = path.join(f.submit_test_img, "sample_submission.csv")
    with open(submit_example, 'r') as example:
        reader = csv.reader(example)
        example_content = list(reader)
        result_name_an123.append(example_content[0])  # Title line
        name_an123 = example_content[1:]  # Exclude first title line "name, an1, an2, an3"

    net_heat = spine_model.SpineModelPAF()
    net_angle = spine_model.CobbAngleModel()
    net_heat.cuda()
    net_heat.eval()
    net_angle.cuda()
    net_angle.eval()

    save_path = f.checkpoint_heat_trainval_path if args.trainval else f.checkpoint_heat_path
    net_heat.load_state_dict(torch.load(save_path))
    save_path_a = f.checkpoint_angle_trainval_path if args.trainval else f.checkpoint_angle_path
    net_angle.load_state_dict(torch.load(save_path_a))

    device = torch.device("cuda")  # Input device

    filename_list = list(zip(*name_an123))[0]
    for filename in filename_list:
        resize_filename = path.join(f.resize_submit_test_img, filename + ".jpg")
        np_img = cv2.imread(resize_filename, cv2.IMREAD_GRAYSCALE)
        np_img = [[np_img]]  # NCHW
        np_img = np.asarray(np_img, np.float32)

        np_norm_img = np_img / 255.
        t_norm_img = torch.from_numpy(np_norm_img).to(device)
        with torch.no_grad():
            out_pcm, out_paf, _, _ = net_heat(t_norm_img)
            an123 = net_angle(out_paf)
        np_an123 = an123.detach().cpu().numpy()
        np_an123 = np_an123[0] * 90.  # batch size 1
        np_an123 = np.clip(np_an123, a_min=0, a_max=100)
        result_line = [filename, np_an123[0], np_an123[1], np_an123[2]]
        result_name_an123.append(result_line)
        print(filename)

    with open(path.join(f.data_root, "submit_result.csv"), "w+", newline='') as result_csv_file:
        writer = csv.writer(result_csv_file)
        [writer.writerow(l) for l in result_name_an123]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval", action='store_true', default=False)
    parser.add_argument("--dataset", type=str, help="Which set to predict? (val, test)", default="val")
    args = parser.parse_args()

    if args.dataset == "val":
        # Validation set
        predict_heatmaps(f.resize_test_img, f.validation_plot_out)
    elif args.dataset == "test":
        # Submit test set
        predict_heatmaps(f.resize_submit_test_img, f.submit_test_plot_out)
    else:
        print("Invalid dataset argument")


