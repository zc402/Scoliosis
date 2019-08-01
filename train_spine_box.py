import torchvision.models
import torch.nn as nn
import numpy as np
import load_utils
import spine_augmentation as aug
import confidence_map as cmap
import part_affinity_field_net
import ladder_shufflenet
import torch.optim as optim
import torch
import os.path as path
import torchvision
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import folders as f
import os
import argparse

def draw_box_on_image(image, box, file):
    assert len(image.shape) == 2, "hw"
    h, w = image.shape
    # x_min, x_max, y_min, y_max = box
    x_min, x_max = box[0:2] * w
    y_min, y_max = box[2:4] * h
    cv2.line(image, tuple([x_min, 0]), tuple([x_min, h]), (255), thickness=2)
    cv2.line(image, tuple([x_max, 0]), tuple([x_max, h]), (255), thickness=2)
    cv2.line(image, tuple([0, y_min]), tuple([w, y_min]), (255), thickness=2)
    cv2.line(image, tuple([0, y_max]), tuple([w, y_max]), (255), thickness=2)
    cv2.imwrite("{}.jpg".format(file), image)

def label_normalize(batch_labels, batch_imgs):
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

    return bl

def get_box(labels):
    # labels : N P xy
    labels = np.array(labels)
    xs = labels[:, :, 0]
    x_max = np.max(xs, axis=1)  # N
    x_min = np.min(xs, axis=1)
    ys = labels[:, :, 1]
    y_max = np.max(ys, axis=1)
    y_min = np.min(ys, axis=1)
    box = np.stack([x_min, x_max, y_min, y_max], axis=-1)
    return box

def submit_test(net):
    import glob
    net.eval().cuda()
    test_imgs = glob.glob(path.join(f.resize_submit_test_img, '*'))  # Wildcard of test images
    for img_path in test_imgs:
        base_name = path.basename(img_path)[:-4]
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # HW
        img = [[img_gray]]  # NCHW
        img = np.asarray(img, np.float32)
        img_01 = img / 255.0
        img_01 = img_01 * np.ones([1,3,1,1], np.float32)
        test_imgs_tensor = torch.from_numpy(img_01).cuda()
        with torch.no_grad():
            pred_box = net(test_imgs_tensor)  # NCHW
        pred_box = pred_box.detach().cpu().numpy()
        draw_box_on_image(img_gray, pred_box[0], path.join(f.submit_test_box_plot, base_name))
        print(base_name)
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a box of spine.')
    parser.add_argument('-s', type=int, default=10, help='batch size')
    parser.add_argument("--trainval", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--submit_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(f.train_box_results, exist_ok=True)
    os.makedirs(f.checkpoint, exist_ok=True)
    os.makedirs(f.submit_test_box_plot, exist_ok=True)

    net = torchvision.models.densenet121(pretrained=True)
    num_conv_features = net.features[-1].num_features
    classifier = nn.Sequential(nn.Linear(num_conv_features, 4), nn.Sigmoid())
    net.classifier = classifier

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


    # Load checkpoint
    # If in trainval mode, no "trainval" checkpoint found,
    # and the checkpoint for "train" mode exists,
    # then load the "train" checkpoint for "trainval" training
    if not args.trainval:
        save_path = f.checkpoint_box_path
        if path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            print("Model loaded")
        else:
            print("New model created")
    else: # Trainval mode
        save_path = f.checkpoint_box_trainval_path
        if path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
            print("Load model weights from [trainval] checkpoint")
        elif path.exists(f.checkpoint_box_path):
            net.load_state_dict(torch.load(f.checkpoint_box_path))
            print("No [trainval] checkpoint but [train] checkpoint exists. Load [train]")
        else:
            print("No [trainval] or [train] checkpoint, training [train, val] from scratch")

    if args.submit_test:
        submit_test(net)

    net.cuda().train()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2000, verbose=True)  # Be patient for n steps

    step = 0
    for train_imgs, train_labels in train_data_loader:
        train_imgs, train_labels = aug.augment_batch_img_for_box(train_imgs, train_labels)
        cm = cmap.ConfidenceMap()
        # Classify labels as (top left, top right, bottom left, bottom right, left center, right center)

        optimizer.zero_grad()
        criterion = nn.MSELoss()
        # To numpy, NCHW. normalize to [0, 1]
        train_imgs = np.asarray(train_imgs, np.float32)[:, np.newaxis, :, :] / 255.0
        # To 3 dim color images
        train_imgs = train_imgs * np.ones([1, 3, 1, 1], dtype=np.float32)
        # Normalize train labels to [0, 1] to predict them directly
        norm_labels = label_normalize(train_labels, train_imgs)
        box_labels = get_box(norm_labels)
        # To tensor
        t_train_imgs = torch.from_numpy(np.asarray(train_imgs)).cuda()
        t_train_labels = torch.from_numpy(box_labels).cuda()

        t_pred_labels = net(t_train_imgs)

        # Heatmap loss
        loss = criterion(t_train_labels, t_pred_labels)
        # point regression loss
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

        if lr <= 0.00005:
            print("Stop on plateau")
            break

        # Test
        if step % 200 == 1:
            net.eval()
            test_imgs, test_labels = next(test_data_loader)
            test_imgs = np.asarray(test_imgs, np.float32)[:, np.newaxis, :, :]
            test_imgs_01 = test_imgs / 255.0
            test_imgs_01 = test_imgs_01 * np.ones([1, 3, 1, 1], dtype=np.float32)
            test_norm_labels = label_normalize(test_labels, test_imgs)
            test_box_labels = get_box(test_norm_labels)
            with torch.no_grad():
                test_imgs_tensor = torch.from_numpy(test_imgs_01).cuda()
                t_test_pred_labels = net(test_imgs_tensor)  # NCHW
                test_pred_labels = t_test_pred_labels.detach().cpu().numpy()
                print(test_pred_labels, test_box_labels, test_pred_labels-test_box_labels)
                # print(test_box_labels)

                test_img = test_imgs[0][0]
                test_box_labels = test_pred_labels[0]
                draw_box_on_image(test_img, test_box_labels, path.join(f.train_box_results, str(step)))
            net.train()