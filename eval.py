import tensorflow as tf
import data_pipeline as dataGen
import time
import numpy as np

with tf.device('/GPU:0'):
    model = tf.keras.models.load_model("model", compile = True)

capture_sec = 1
num_frames = int(capture_sec*60)

BATCH_SIZE = 4
IMG_SIZE = [100, 100]
num_vids = 16

location = "D:/dfdc_reduced_set/test_videos"

strTime = time.time()
eval_dataset = dataGen.build_dataset(location = location, capture_sec = capture_sec, dataset_type = 'eval', batch_size = BATCH_SIZE, img_size = IMG_SIZE, num_vids = num_vids)
totTime = time.time() - strTime
print(f"Time To Build Train Dataset - {int(totTime // 60)}:{int(totTime % 60)}\n\n")


for step, data in enumerate(eval_dataset):
    op = model(data)
    print(op)
