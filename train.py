"""
Run this script to train the spine keypoint network
"""
import numpy as np
import load_utils
import spine_augmentation as aug
import confidence_map as cmap
import SpineModel
import torch.optim as optim
import torch.nn as nn
import torch
import os.path as path
import torchvision
import matplotlib.pyplot as plt
import cv2

def imshow(img, name):
    # img = img * 255.0     # unnormalize
    npimg = img.detach().cpu().numpy()
    npimg = np.clip(npimg, 0., 1.)
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = (npimg*255.).astype(np.uint8)
    npimg = cv2.resize(npimg, None, fx=4, fy=4)
    cv2.imwrite(path.join("results", "%s.jpg" % name), npimg)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")
    batch_size = 5
    train_data_loader = load_utils.train_loader(batch_size)
    test_data_loader = load_utils.test_loader(batch_size)
    # train_imgs, train_labels = next(train_data_loader)
    device = torch.device("cuda")

    save_path = "checkpoint.pth"
    net = SpineModel.SpineModel()
    if path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("Model loaded")
    else:
        print("New model created")

    net.cuda()
    print(net)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    step = 0
    for train_imgs, train_labels in train_data_loader:
        train_imgs, train_labels = aug.augment_batch_img(train_imgs, train_labels)
        # Classify labels as (top left, top right, bottom left, bottom right)
        train_gaussian_imgs = cmap.ConfidenceMap().batch_gaussian_split_corner(train_imgs, train_labels, zoom=4)

        optimizer.zero_grad()
        # To numpy, NCHW. zoom to [0, 1]
        train_imgs = np.asarray(train_imgs, np.float32)[:, np.newaxis, :, :] / 255.0
        # To tensor
        train_imgs = torch.from_numpy(train_imgs).to(device)
        # To numpy -> to tensor
        train_gaussian_imgs = torch.from_numpy(np.asarray(train_gaussian_imgs)).to(device)
        output = net(train_imgs)
        loss = criterion(output, train_gaussian_imgs)
        loss.backward()
        optimizer.step()
        step = step + 1
        print("%f" % loss.item())

        # Save
        if step % 100 == 0:
            torch.save(net.state_dict(), save_path)
            print("Model saved")

        # Test
        if step % 100 == 0:
            net.eval()
            with torch.no_grad():
                test_imgs, test_labels = next(test_data_loader)
                test_imgs = np.asarray(test_imgs, np.float32)[:, np.newaxis, :, :] / 255.0
                test_imgs = torch.from_numpy(test_imgs).to(device)
                test_out = net(test_imgs)  # NCHW
                img = torchvision.utils.make_grid(test_out[:, 0:1, ...])
                imshow(img, str(step))
            net.train()