"""
Run this script to train the spine keypoint network
"""
import numpy as np
import load_utils
import spine_augmentation as aug
import confidence_map as cmap
import ladder_shufflenet
import torch.optim as optim
import torch.nn as nn
import torch
import os.path as path
import torchvision
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from PIL import Image
import folders as f
import os
import argparse

def save_grid_images(img, gau, name):
    # gau = F.interpolate(gau, size=(img.size(2), img.size(3)), mode="bilinear")
    gau_img = torch.cat((gau, img), dim=0)
    gau_img = torchvision.utils.make_grid(gau_img, nrow=batch_size)

    npimg = gau_img.detach().cpu().numpy()
    npimg = np.clip(npimg, 0., 1.)
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = (npimg*255.).astype(np.uint8)
    # npimg = cv2.resize(npimg, None, fx=4, fy=4)  # Gaussian
    cv2.imwrite(path.join(f.train_results, "%s.jpg" % name), npimg)

def label_normalize_flatten(batch_labels, batch_imgs):
    """
    Normalize pts to [0,1] for training the prediction network
    :param batch_labels:
    :param batch_imgs:
    :return:
    """
    hw = np.asarray(batch_imgs).shape[2:4]
    bl = np.array(batch_labels, np.float32)
    # Normalization
    bl[:, :, 0] = bl[:, :, 0] / hw[1]
    bl[:, :, 1] = bl[:, :, 1] / hw[0]
    # Flatten
    bl = bl.reshape((bl.shape[0], -1))
    return bl


def plot_norm_pts(batch_imgs, batch_norm_pts, name):
    hw = batch_imgs.shape[2:4]
    plt.style.use('grayscale')
    batch_norm_pts = batch_norm_pts.detach().cpu().numpy()
    batch_norm_pts = batch_norm_pts.reshape((batch_imgs.shape[0], 68, 2))  # Batchsize, joints*4, xy
    for i in range(batch_imgs.shape[0]):
        img = batch_imgs[i,0]  # NCHW -> HW
        # img = np.repeat(img[..., np.newaxis], 3, axis=2)  # HWC
        img = img / 255.
        plt_img = Image.fromarray(img)
        plt.imshow(plt_img)

        xy_list = batch_norm_pts[i]  # [J][XY]
        xy_list *= np.array((hw[1], hw[0]), np.float32)
        x_list, y_list = np.transpose(xy_list, axes=[1, 0]).tolist()  # [XY][J]
        plt.scatter(x_list, y_list, color='yellow', s=9)
        for j in range(len(x_list)):
            plt.annotate(j, (x_list[j], y_list[j]), color='red', size=5)
        plt.axis("off")
        plt.savefig(path.join(f.train_results, "%s_%d_pts.jpg" % (name, i)), dpi=400)
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('-s', type=int, default=4, help='batch size')
    parser.add_argument("--trainval", action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(f.train_results, exist_ok=True)
    os.makedirs(f.checkpoint, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    batch_size = args.s
    print("Training with batch size: %d" % batch_size)
    if args.trainval:  # Final training, use train and val set
        train_data_loader = load_utils.train_loader(batch_size, use_trainval=True)
        print("--- Using [train, val] set as training set!")
    else:
        train_data_loader = load_utils.train_loader(batch_size)
    test_data_loader = load_utils.test_loader(batch_size)
    device = torch.device("cuda")

    net = ladder_shufflenet.LadderModel()
    # Load checkpoint
    # If in trainval mode, no "trainval" checkpoint found,
    # and the checkpoint for "train" mode exists,
    # then load the "train" checkpoint for "trainval" training
    if not args.trainval:
        save_path = f.checkpoint_heat_path
        if path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            print("Model loaded")
        else:
            print("New model created")
    else: # Trainval mode
        save_path = f.checkpoint_heat_trainval_path
        if path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            print("Load model weights from [trainval] checkpoint")
        elif path.exists(f.checkpoint_heat_path):
            net.load_state_dict(torch.load(f.checkpoint_heat_path))
            print("No [trainval] checkpoint but [train] checkpoint exists. Load [train]")
        else:
            print("No [trainval] or [train] checkpoint, training [train, val] from scratch")

    net.cuda().train()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5000, verbose=True)  # Be patient for n steps

    step = 0
    for train_imgs, train_labels in train_data_loader:
        train_imgs, train_labels = aug.augment_batch_img(train_imgs, train_labels)
        cm = cmap.ConfidenceMap()
        # Classify labels as (top left, top right, bottom left, bottom right, left center, right center)
        heat_scale = 1
        # NCHW_corner_gau = cm.batch_gaussian_split_corner(train_imgs, train_labels, heat_scale)
        NCHW_center_gau = cm.batch_gaussian_LRCenter(train_imgs, train_labels, heat_scale)
        NCHW_lines = cm.batch_lines_LRCenter(train_imgs, train_labels, heat_scale)
        # train_gaussian_imgs = np.concatenate((NCHW_corner_gau, NCHW_center_gau, NCHW_lines), axis=1)

        optimizer.zero_grad()
        criterion = nn.MSELoss()
        # To numpy, NCHW. normalize to [0, 1]
        train_imgs = np.asarray(train_imgs, np.float32)[:, np.newaxis, :, :] / 255.0
        # Normalize train labels to [0, 1] to predict them directly
        norm_labels = label_normalize_flatten(train_labels, train_imgs)
        # To tensor
        train_imgs = torch.from_numpy(np.asarray(train_imgs)).cuda()
        tensor_gt_pcm = torch.from_numpy(np.asarray(NCHW_center_gau)).cuda()
        tensor_gt_paf = torch.from_numpy(np.asarray(NCHW_lines)).cuda()

        out_pcm, out_paf = net(train_imgs)

        # Heatmap loss
        loss1 = criterion(out_pcm, tensor_gt_pcm)
        # point regression loss
        norm_labels = torch.from_numpy(norm_labels).to(device)
        loss2 = criterion(out_paf, tensor_gt_paf)
        loss = loss1 + (loss2 / 5)  # pcm + paf
        loss.backward()
        optimizer.step()
        step = step + 1
        loss_value = loss.item()
        scheduler.step(loss_value)
        lr = optimizer.param_groups[0]['lr']
        print("Step: %d, Loss: %f, LR: %f" % (step, loss_value, lr))

        # Save
        if step % 200 == 0:
            torch.save(net.state_dict(), save_path)
            print("Model saved")

        if lr <= 10e-5:
            print("Stop on plateau")
            break

        # Test
        if step % 200 == 0:
            net.eval()
            test_imgs, test_labels = next(test_data_loader)
            test_imgs = np.asarray(test_imgs, np.float32)[:, np.newaxis, :, :]
            test_imgs_01 = test_imgs / 255.0
            with torch.no_grad():
                test_imgs_tensor = torch.from_numpy(test_imgs_01).to(device)
                out_pcm, out_paf = net(test_imgs_tensor)  # NCHW

                save_grid_images(test_imgs_tensor, out_pcm[:, 0:1, ...], str(step))
                # plot_norm_pts(test_imgs, test_out_pts, str(step))
            net.train()
