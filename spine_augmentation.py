"""
Image augmentation module. Run this script to see augmentation results.
"""
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage
import numpy as np


def augment_batch_img(batch_img, batch_pts, plot=False):
    """
    Image augmentation, used when training
    :param batch_img: [B,H,W,C]
    :param batch_pts: [B,number,xy]
    :return: aug_b_img, aug_b_pts
    """
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.CropAndPad(percent=((0., 0.), (-0.1, 0.1), (0., 0.), (-0.1, 0.1))),
        iaa.Affine(rotate=(-10, 10)),
        iaa.Add((-25, 25))  # change brightness
    ])
    aug_b_imgs, aug_b_pts = seq(images=batch_img, keypoints=batch_pts)

    if plot:
        import cv2
        batch_img = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in batch_img]
        aug_b_imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in aug_b_imgs]
        for i in range(len(batch_img)):
            print("[Image #%d]" % (i,))
            keypoints_before = KeypointsOnImage.from_xy_array(
                batch_pts[i], shape=batch_img[i].shape)
            keypoints_after = KeypointsOnImage.from_xy_array(
                aug_b_pts[i], shape=aug_b_imgs[i].shape)
            image_before = keypoints_before.draw_on_image(batch_img[i])
            image_after = keypoints_after.draw_on_image(aug_b_imgs[i])
            ia.imshow(np.hstack([image_before, image_after]))
    return aug_b_imgs, aug_b_pts

if __name__ == "__main__":
    # Run this script to see augmentation results
    import load_utils
    data_gen = load_utils.train_loader(5)
    for imgs, labels in data_gen:
        augment_batch_img(imgs, labels, plot=True)
