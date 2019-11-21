#### Automated Vertebral Landmarks and Spinal Curvature Estimation using Non-directional Part Affinity Fields

# Preparation
Acquire the datasets (see below).

Unzip the train, val, data into `dataroot/boostnet_labeldata`

Unzip the test data into `dataroot/submit_test_images` folder

Merge train and val csv annotation, put into `dataroot/trainval-labels` folder

By default, `dataroot` = `../ScoliosisData`

## Use "train" set, test on "val" set
Run resize_images.py to apply augmentation and resize

Run train.py to train on "train" set

Run eval.py to produce heatmaps

## Use "train val" set, test on "submit test" set
Run resize_images.py to flipLR, resize

Run train.py --trainval to train

Run eval.py --trainval to produce heatmaps

Run cobb_angle_eval.py to evaluate landmark pairs and Cobb angles

# Dataset

Dataset provided by: 

>Wu, Hongbo, et al. "Automatic landmark estimation for adolescent idiopathic scoliosis assessment using BoostNet." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2017.