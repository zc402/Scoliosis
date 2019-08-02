import os.path as path

# Fliplr
data_root = path.join("..", "ScoliosisData")
data_spine = path.join(data_root, "boostnet_labeldata")

train_img = path.join(data_spine, "data", "training")
train_mat = path.join(data_spine, "labels", "training")

train_img_flip = path.join(data_spine, "image_flip", "training")
train_mat_flip = path.join(data_spine, "labels_flip", "training")

# Create train-val set for final training
val_img_flip = path.join(data_spine, "image_flip", "test")
val_mat_flip = path.join(data_spine, "labels_flip", "test")

# Plot points
plot = path.join(data_root, "plot_label_on_image")

# Resize
val_img = path.join(data_spine, "data", "test")
val_mat = path.join(data_spine, "labels", "test")

resized_data = path.join(data_root, "resized_data")
resize_train_img = path.join(resized_data, "image", "training")
resize_train_label = path.join(resized_data, "labels", "training")
resize_test_img = path.join(resized_data, "image", "test")
resize_test_label = path.join(resized_data, "labels", "test")
submit_test_img = path.join(data_root, "submit_test_images")
resize_submit_test_img = path.join(resized_data, "image", "submit_test")

# Temporal folder for images with less head and leg areas
submit_test_img_lesshead = path.join(data_root, "submit_test_images_lesshead")


# Train
train_results = path.join(data_root, "train_resutls")  # Results output folder
checkpoint = path.join(data_root, "checkpoint")
checkpoint_heat_path = path.join(checkpoint, "checkpoint.pth")  # -heat
checkpoint_angle_path = path.join(checkpoint, "checkpoint-angle.pth")
checkpoint_heat_trainval_path = path.join(checkpoint, "checkpoint-heat-trainval.pth")
checkpoint_angle_trainval_path = path.join(checkpoint, "checkpoint-angle-trainval.path")
checkpoint_box_path = path.join(checkpoint, "checkpoint-box.pth")
checkpoint_box_trainval_path = path.join(checkpoint, "checkpoint-box-trainval.pth")

# Eval
validation_plot_out = path.join(data_root, "validation_plot")
submit_test_plot_out = path.join(data_root, "submit_test_plot")
submit_test_plot_pairs = path.join(data_root, "submit_test_plot_pairs")

# Angle csv
train_angle = path.join(data_spine, "labels", "training")
val_angle = path.join(data_spine, "labels", "test")
trainval_angle = path.join(data_root, "trainval-labels")

# Submit
resize_trainval_img = path.join(resized_data, "image", "train-val")
resize_trainval_label = path.join(resized_data, "labels", "train-val")

# Box
train_box_results = path.join(data_root, "train_box_resutls")
submit_test_box_plot = path.join(data_root, "submit_test_box_plot")
submit_test_trim_images = path.join(data_root, "submit_test_trim_images")