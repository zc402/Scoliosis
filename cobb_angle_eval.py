# Run cobb angle evaluation script based on center point heatmaps
import numpy as np
import load_utils
import ladder_shufflenet
import torch
import os.path as path
import cv2
import folders as f
import os
import argparse
import cobb_angle_parse as cap
import csv


# Run evaluation on submit test set
def run_on_submit_test(net_heat):
    os.makedirs(f.submit_test_plot_pairs, exist_ok=True)
    result_name_an123 = []  # Parsing results to be wrote
    submit_example = path.join(f.submit_test_img, "sample_submission.csv")
    with open(submit_example, 'r') as example:
        reader = csv.reader(example)
        example_content = list(reader)
        result_name_an123.append(example_content[0])  # Title line
        name_an123 = example_content[1:]  # Exclude first title line "name, an1, an2, an3"

    save_path = f.checkpoint_heat_trainval_path
    # save_path = f.checkpoint_heat_path
    net_heat.load_state_dict(torch.load(save_path))

    filename_list = list(zip(*name_an123))[0]
    for filename in filename_list:
        #if '88' not in filename:
        #    continue
        resize_filepath = path.join(f.resize_submit_test_img, filename + ".jpg")
        np_img_ori = cv2.imread(resize_filepath, cv2.IMREAD_GRAYSCALE)
        np_img = [[np_img_ori]]  # NCHW
        np_img = np.asarray(np_img, np.float32)

        np_norm_img = np_img / 255.
        t_norm_img = torch.from_numpy(np_norm_img).cuda()
        with torch.no_grad():
            out_dict = net_heat(t_norm_img)


        np_pcm_lrcenter = out_dict["pcm"].detach().cpu().numpy()[0, 4:6]
        np_paf_center = out_dict["paf"].detach().cpu().numpy()[0, 0:1]
        np_neck = out_dict["pcm"].detach().cpu().numpy()[0, 6]

        cobb_dict = cap.cobb_angles(np_pcm_lrcenter, np_paf_center, np_img_ori, np_neck, use_filter=True)
        pred_angles, pairs_img, pairs_lr_value = cobb_dict["angles"], cobb_dict["pairs_img"], cobb_dict["pair_lr_value"]
        np.save(path.join(f.validation_plot_out, "{}.npy".format(filename)), pairs_lr_value)
        result_line = [filename, float(pred_angles[0]), float(pred_angles[1]), float(pred_angles[2])]
        result_name_an123.append(result_line)
        print(filename)
        # cap.cvsave(pairs_img, "{}".format(filename))
        cv2.imwrite(path.join(f.submit_test_plot_pairs, "{}.jpg".format(filename)), pairs_img)

    with open(path.join(f.data_root, "submit_result.csv"), "w+", newline='') as result_csv_file:
        writer = csv.writer(result_csv_file)
        [writer.writerow(l) for l in result_name_an123]

def run_on_validation(net_heat):
    # Run on validation set
    save_path = f.checkpoint_heat_path
    net_heat.load_state_dict(torch.load(save_path))
    test_data_loader = load_utils.test_loader(1, load_angle=True)
    avg_smape = []
    for step in range(128):
        test_imgs, test_labels, test_angles = next(test_data_loader)
        test_imgs_f = np.asarray(test_imgs, np.float32)[:, np.newaxis, :, :]
        test_imgs_01 = test_imgs_f / 255.0
        test_imgs_tensor = torch.from_numpy(test_imgs_01).cuda()
        with torch.no_grad():
            out_pcm, out_paf, _, _ = net_heat(test_imgs_tensor)  # NCHW
        np_pcm = out_pcm.detach().cpu().numpy()
        np_paf = out_paf.detach().cpu().numpy()

        cobb_dict = cap.cobb_angles(np_pcm[0, 4:6], np_paf[0], test_imgs[0], np_pcm[0, 6], use_filter=False)
        pred_angles, pairs_img, pairs_lr_value = cobb_dict["angles"], cobb_dict["pairs_img"], cobb_dict["pair_lr_value"]
        smape = cap.SMAPE(pred_angles, test_angles[0])
        avg_smape.append(smape)
        print(step, smape)
        print(pred_angles - test_angles[0])
        cap.cvsave(pairs_img, "{}".format(step))
        print("end-----------------------------")
    print("SMAPE:", np.mean(avg_smape))


def parse_cobb_angle_by_annotated_points():
    # Use annotated corner points to parse cobb angle
    # so as to test cobb_angle_parser
    import confidence_map as cm
    test_data_loader = load_utils.test_loader(1, load_angle=True)
    avg_smape = []
    counter_isS = 0
    counter_notS = 0
    for step in range(128):
        test_imgs, test_labels, test_angles = next(test_data_loader)
        # gt_a1, gt_a2, gt_a3 = test_angles[0]
        # gt center points
        # [lr][N][17(joint)][xy]
        l_bcs, r_bcs = cm.ConfidenceMap()._find_LCenter_RCenter(test_labels)
        gt_lc, gt_rc = l_bcs[0], r_bcs[0]
        pair_lr_value = gt_lc, gt_rc

        # -----------------------------Use angle_parse from here
        # Sort pairs by y
        pair_lr_value = cap.sort_pairs_by_y(pair_lr_value)
        # Use sigma of x, interval, length to delete wrong pairs
        # pair_lr_value = rbf.simple_filter(pair_lr_value)
        # rbf_dict = rbf.filter(pair_lr_value)
        # pair_lr_value = rbf_dict["pair_lr_value"]
        # pair_lr_value = reduce_redundant_paris(pair_lr_value)
        # [p_len][xy] vector coordinates. (sorted by bone confidence, not up to bottom)
        bones = cap.bone_vectors(pair_lr_value)
        # Index1(higher), index2(lower) of max angle; a1: max angle value
        max_ind1, max_ind2, a1 = cap.max_angle_indices(bones, pair_lr_value)

        hmids = (pair_lr_value[0] + pair_lr_value[1]) / 2
        if not cap.isS(hmids):
            a2 = np.rad2deg(np.arccos(cap.cos_angle(bones[max_ind1], bones[0])))  # Use first bone
            a3 = np.rad2deg(np.arccos(
                cap.cos_angle(bones[max_ind2], bones[-1])))  # Note: use last bone on submit test set gains better results

        # print(max_ind1, max_ind2)
        else:  # isS
            a2, a3 = cap.handle_isS_branch(pair_lr_value, max_ind1, max_ind2, test_imgs[0].shape[0])
        sub = np.array([a1, a2, a3]) - test_angles[0]
        print(step)
        print(sub)
        print("------------end---------------")
    # print(np.mean(avg_smape))
    # print("number of isS-notS:", counter_isS, counter_notS)


def gen_manual_img_label():
    """Generate img, label of manually marked test set"""
    import glob
    # npy file, point range 0~1
    npy_list = glob.glob(path.join(f.manual_npy_submit_test, "*.npy"))
    name_list = [path.splitext(path.basename(npy))[0] for npy in npy_list]
    img_list = [path.join(f.resize_submit_test_img, npy)+".jpg" for npy in name_list]
    for i in range(len(npy_list)):
        npy = np.load(npy_list[i]) #[p][xy]
        img_path = img_list[i]
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        ori_npy = npy * [img.shape[1], img.shape[0]]
        img_name = path.basename(img_path).replace(".jpg", "")
        yield img, ori_npy, img_name

def eval_manually_marked_submit_test():
    import confidence_map as cmap
    # Prepare the submit csv file
    os.makedirs(f.submit_test_plot_pairs, exist_ok=True)
    result_name_an123 = []  # Parsing results to be wrote
    submit_example = path.join(f.submit_test_img, "sample_submission.csv")
    with open(submit_example, 'r') as example:
        reader = csv.reader(example)
        example_content = list(reader)
        result_name_an123.append(example_content[0])  # Title line
        name_an123 = example_content[1:]  # Exclude first title line "name, an1, an2, an3"

    # Read manually annotated npy
    gen = gen_manual_img_label()
    for img, manual_label, filename in gen:
        # Manually marked labels
        cm = cmap.ConfidenceMap()
        # Classify labels as (top left, top right, bottom left, bottom right, left center, right center)
        heat_scale = 1
        img = [img]  # batch, h, w
        manual_label = [manual_label]
        heat_hw = np.asarray(img).shape[1:3]
        NCHW_corner_gau = cm.batch_gaussian_split_corner(img, manual_label, heat_scale)
        NCHW_center_gau = cm.batch_gaussian_LRCenter(img, manual_label, heat_scale)
        NCHW_c_lines = cm.batch_lines_LRCenter(heat_hw, manual_label, heat_scale)
        NCHW_first_lrpt = cm.batch_gaussian_first_lrpt(img, manual_label)
        NCHW_paf = NCHW_c_lines
        NCHW_pcm = np.concatenate((NCHW_corner_gau, NCHW_center_gau, NCHW_first_lrpt), axis=1)

        np_pcm_lrcenter = NCHW_pcm[0, 4:6]
        np_paf_center = NCHW_paf[0, 0:1]
        np_neck = NCHW_pcm[0, 6]

        cobb_dict = cap.cobb_angles(np_pcm_lrcenter, np_paf_center, img[0], np_neck, use_filter=False)
        pred_angles, pairs_img, pairs_lr_value = cobb_dict["angles"], cobb_dict["pairs_img"], cobb_dict["pair_lr_value"]
        np.save(path.join(f.validation_plot_out, "{}.npy".format(filename)), pairs_lr_value)
        result_line = [filename, float(pred_angles[0]), float(pred_angles[1]), float(pred_angles[2])]
        result_name_an123.append(result_line)
        print(filename)
        # cap.cvsave(pairs_img, "{}".format(filename))
        cv2.imwrite(path.join(f.submit_test_plot_pairs, "{}.jpg".format(filename)), pairs_img)

        with open(path.join(f.data_root, "submit_result.csv"), "w+", newline='') as result_csv_file:
            writer = csv.writer(result_csv_file)
            [writer.writerow(l) for l in result_name_an123]


if __name__ == "__main__":
    net = ladder_shufflenet.LadderModelAdd()
    net.eval()
    net.cuda()
    os.makedirs(f.validation_plot_out, exist_ok=True)

    # run_on_validation(net)
    run_on_submit_test(net)
    # eval_manually_marked_submit_test()
