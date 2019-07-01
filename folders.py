import os.path as path
# Fliplr
data_root = path.join("..", "ScoliosisData")
data_spine = path.join(data_root, "data_spine")

train_img = path.join(data_spine, "image", "training")
train_mat = path.join(data_spine, "labels", "training")

train_img_flip = path.join(data_spine, "image_flip", "training")
train_mat_flip = path.join(data_spine, "labels_flip", "training")

# Plot points
plot = path.join(data_root, "plot_label_on_image")

# Resize
test_img = path.join(data_spine, "image", "test")
test_mat = path.join(data_spine, "labels", "test")

resized_data = path.join(data_root, "resized_data")
resize_train_img = path.join(resized_data, "image", "training")
resize_train_label = path.join(resized_data, "labels", "training")
resize_test_img = path.join(resized_data, "image", "test")
resize_test_label = path.join(resized_data, "labels", "test")

# Train
train_results = path.join(data_root, "train_resutls")  # Results output folder
