import os.path as path

# Fliplr
data_root = path.join("..", "ScoliosisData")
data_spine = path.join(data_root, "boostnet_labeldata")

train_img = path.join(data_spine, "data", "training")
train_mat = path.join(data_spine, "labels", "training")

train_img_flip = path.join(data_spine, "image_flip", "training")
train_mat_flip = path.join(data_spine, "labels_flip", "training")

# Plot points
plot = path.join(data_root, "plot_label_on_image")

# Resize
test_img = path.join(data_spine, "data", "test")
test_mat = path.join(data_spine, "labels", "test")

resized_data = path.join(data_root, "resized_data")
resize_train_img = path.join(resized_data, "image", "training")
resize_train_label = path.join(resized_data, "labels", "training")
resize_test_img = path.join(resized_data, "image", "test")
resize_test_label = path.join(resized_data, "labels", "test")
submit_test_img = path.join(data_root, "submit_test_images")
resize_submit_test_img = path.join(resized_data, "image", "submit_test")

# Train
train_results = path.join(data_root, "train_resutls")  # Results output folder
checkpoint = path.join(data_root, "checkpoint")
checkpoint_heat_path = path.join(checkpoint, "checkpoint.pth")
checkpoint_angle_path = path.join(checkpoint, "checkpoint-angle.pth")

# Eval
validation_plot_out = path.join(data_root, "validation_plot")
submit_test_plot_out = path.join(data_root, "evaluation-submit_test")

# Angle csv
train_angle = path.join(data_spine, "labels", "training")
test_angle = path.join(data_spine, "labels", "test")