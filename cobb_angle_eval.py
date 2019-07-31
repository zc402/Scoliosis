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


# Run evaluation on submit test set
def run_on_submit_test(net_heat):
    import csv
    result_name_an123 = []  # Parsing results to be wrote
    submit_example = path.join(f.submit_test_img, "sample_submission.csv")
    with open(submit_example, 'r') as example:
        reader = csv.reader(example)
        example_content = list(reader)
        result_name_an123.append(example_content[0])  # Title line
        name_an123 = example_content[1:]  # Exclude first title line "name, an1, an2, an3"

    # save_path = f.checkpoint_heat_trainval_path  # FIXME: change back to trainval path
    save_path = f.checkpoint_heat_path
    net_heat.load_state_dict(torch.load(save_path))

    filename_list = list(zip(*name_an123))[0]
    for filename in filename_list:
        resize_filename = path.join(f.resize_submit_test_img, filename + ".jpg")
        np_img_ori = cv2.imread(resize_filename, cv2.IMREAD_GRAYSCALE)
        np_img = [[np_img_ori]]  # NCHW
        np_img = np.asarray(np_img, np.float32)

        np_norm_img = np_img / 255.
        t_norm_img = torch.from_numpy(np_norm_img).cuda()
        with torch.no_grad():
            out_pcm, out_paf, _, _ = net_heat(t_norm_img)

        np_pcm = out_pcm.detach().cpu().numpy()
        np_paf = out_paf.detach().cpu().numpy()

        cobb_dict = cap.cobb_angles(np_pcm[0, 0:2], np_paf[0], np_img_ori)
        pred_angles, pairs_img, pairs_lr_value = cobb_dict["angles"], cobb_dict["pairs_img"], cobb_dict["pair_lr_value"]
        np.save(path.join(f.validation_plot_out, "{}.npy".format(filename)), pairs_lr_value)
        result_line = [filename, float(pred_angles[0]), float(pred_angles[1]), float(pred_angles[2])]
        result_name_an123.append(result_line)
        print(filename)
        cap.cvsave(pairs_img, "{}".format(filename))

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

        cobb_dict = cap.cobb_angles(np_pcm[0, 0:2], np_paf[0], test_imgs[0])
        pred_angles, pairs_img, pairs_lr_value = cobb_dict["angles"], cobb_dict["pairs_img"], cobb_dict["pair_lr_value"]
        smape = cap.SMAPE(pred_angles, test_angles[0])
        avg_smape.append(smape)
        print(step, smape)
        print(pred_angles - test_angles[0])
        cap.cvsave(pairs_img, "{}".format(step))
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
        gt_a1, gt_a2, gt_a3 = test_angles[0]
        # gt center points
        # [lr][N][17(joint)][xy]
        l_bcs, r_bcs = cm.ConfidenceMap()._find_LCenter_RCenter(test_labels)
        gt_lc, gt_rc = l_bcs[0], r_bcs[0]
        pair_lr_value = gt_lc, gt_rc
        pair_lr_value = cap.sort_pairs_by_y(pair_lr_value)
        # From cobb_angle_parse
        # [p_len][xy] vector coordinates. (sorted by bone confidence, not up to bottom)
        bones = cap.bone_vectors(pair_lr_value)
        # [len_b][len_b] angle matrix
        am = cap.angle_matrix(bones)
        sort_indices = np.unravel_index(np.argsort(am, axis=None), am.shape)
        # Two indices that composed the largest angle
        max_ind1, max_ind2 = sort_indices[0][-1], sort_indices[1][-1]
        # Find out which one is upper bone
        max_ind1, max_ind2 = cap.make_ind1_upper(max_ind1, max_ind2, pair_lr_value)
        a1 = am[max_ind1, max_ind2]

        # If not "isS" (in matlab)
        hmids = (pair_lr_value[0] + pair_lr_value[1]) / 2
        if not cap.isS(hmids):
            counter_notS = counter_notS + 1
            #a2 = np.rad2deg(np.arccos(cap.cos_angle(bones[max_ind1], np.array([1, 0]))))
            #a3 = np.rad2deg(np.arccos(cap.cos_angle(bones[max_ind2], np.array([1, 0]))))
            a2 = np.rad2deg(np.arccos(cap.cos_angle(bones[max_ind1], bones[0])))
            a3 = np.rad2deg(np.arccos(cap.cos_angle(bones[max_ind2], bones[-1])))
            pred_angles = np.array([a1, a2, a3])
            print(pred_angles - test_angles[0])
            smape = cap.SMAPE(pred_angles, test_angles[0])
            avg_smape.append(smape)
        else:
            print("isS case (not implemented)")
            counter_isS = counter_isS + 1
    print(np.mean(avg_smape))
    print("number of isS-notS:", counter_isS, counter_notS)


if __name__ == "__main__":
    net = ladder_shufflenet.LadderModel()
    net.eval()
    net.cuda()
    os.makedirs(f.validation_plot_out, exist_ok=True)

    # run_on_validation(net)
    run_on_submit_test(net)
    # parse_cobb_angle_by_annotated_points()
