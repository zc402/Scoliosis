# Usage

## Train a model and validate
- Extract train-val set as boostnet_labeldata to ../ScoliosisData/
- Run `python3 fliplr_and_points.py` to generate fliplr images
- (Optional) Run `python3 plot_points` to plot labels on images
- Run `python3 resize_images.py` to shrink images in train and val set to same size
- Run `python3 train.py` to train the heatmap predictor
- Run `python3 train_angle.py` to train angle predictor
- Run `python3 eval.py` to plot test output to ../ScoliosisData/evaluation and write csv output to "data_root"

## Train a model with [train, val] set for final submission
- `python3 fliplr_and_points.py` to flip both train and test set
- `python3 resize_images.py` to handle "train, val, test" set images
- `python3 train.py --trainval` to train with train and val set

TODO: merge train and val csv annotation
