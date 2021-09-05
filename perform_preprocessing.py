import processing
import os

location = "D:/deepfake_database/dfdc_train_part_49/dfdc_train_part_49"
os.makedirs("train_files")

save_path = "train_files"
capture_sec = 1

processing.processing(location = location, save_path = save_path, capture_sec = capture_sec, num_vids = 500)
