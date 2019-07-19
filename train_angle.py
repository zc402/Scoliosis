"""
Train cobb angle value using heatmaps
"""
import load_utils
import argparse
import spine_model
import folders as f
import os.path as path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default=5, required=False, help="batch size")
    args = parser.parse_args()
    batch_size = args.s
    train_data_loader = load_utils.train_loader(batch_size, load_angle=True)
    test_data_loader = load_utils.test_loader(batch_size, load_angle=True)

    net_heat = spine_model.SpineModelPAF()
    net_heat.cuda()
    net_heat.eval()

    save_path_heat = f.checkpoint_heat_path
    if path.exists(save_path_heat):
        net_heat.load_state_dict(torch.load(save_path_heat))
    else:
        raise FileNotFoundError("Heatmap model checkpoint not found.")

    net_angle = spine_model.CobbAngleModel()
    net_angle.cuda()

    save_path_angle = f.checkpoint_angle_path
    if path.exists(save_path_angle):
        net_angle.load_state_dict(torch.load(save_path_angle))
        print("Load angle net checkpoint")
    else:
        print("Train angle net from scratch")

    optimizer = optim.Adam(net_angle.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=6000, verbose=True)  # Be patient for n steps

    step = 0
    device = torch.device("cuda")
    for train_imgs, train_labels, train_angles in train_data_loader:

        criterion = nn.MSELoss()
        # To numpy, NCHW. normalize to [0, 1]
        norm_train_imgs = np.asarray(train_imgs, np.float32)[:, np.newaxis, :, :] / 255.0
        t_train_imgs = torch.from_numpy(norm_train_imgs).to(device)

        out_pcm, out_paf, _, _= net_heat(t_train_imgs)

        np_train_angles = np.array(train_angles, dtype=np.float32)
        norm_train_angles = np_train_angles / 90.
        t_train_angles = torch.from_numpy(norm_train_angles).to(device)

        predict_angles = net_angle(out_paf)

        loss = criterion(predict_angles, t_train_angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step = step + 1
        loss_value = loss.item()
        scheduler.step(loss_value)
        lr = optimizer.param_groups[0]['lr']
        print("Loss: %f, LR: %f" % (loss_value, lr))
        if lr < 10e-5:
            print("Stop on plateau")
            break

        # Save
        if step % 100 == 0:
            torch.save(net_angle.state_dict(), save_path_angle)
            print("Angle model saved")

        # Test
        if step % 100 == 0:
            net_angle.eval()
            with torch.no_grad:
                test_imgs, _, test_angles= next(test_data_loader)
                norm_test_imgs = np.asarray(test_imgs, np.float32)[:, np.newaxis, :, :] / 255.0
                t_test_imgs = torch.from_numpy(norm_test_imgs).to(device)

                out_pcm, out_paf, _, _= net_heat(t_test_imgs)

                np_test_angles = np.array(test_angles, dtype=np.float32)
                norm_test_angles = np_test_angles / 90.
                t_test_angles = torch.from_numpy(norm_test_angles).to(device)

                norm_predict_angles = net_angle(t_test_imgs)
                norm_predict_angles = norm_predict_angles.detach().cpu().numpy()
                predict_angles = norm_predict_angles * 90.

                #Use SMAPE?
                print(np_test_angles - predict_angles)

            net_angle.train()
