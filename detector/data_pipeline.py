'''
This file generates the dataset needed

This file is generic and can be extended to any of the datasets, the train,
eval or test
'''

import tensorflow as tf
import numpy as np
import generate_preprocessed_files as processing
import cv2
import glob
import json

def build_dataset(file_loc = "train_files", img_size = [100,100], shuffle_buffer = 401, batch_size = 16, capture_sec = 1):

    # <------------------------------------ tf.data parameters ------------------------------------>

    BATCH_SIZE = batch_size
    SHUFFLE_BUFFER = shuffle_buffer

    # <------------------------------------ image parameters ------------------------------------>

    capture_sec = capture_sec
    num_frame = int(capture_sec * 60)
    img_size = img_size

    # <------------------------------------ data preprocessing ------------------------------------>
    #converting to valid filepaths
    #converting float labels to tensor values
    labels = []
    files = glob.glob(file_loc+"/*.mp4")
    for i in range(len(files)):
        files[i] = files[i].replace("\\", "/")

    metadata = json.load(open(file_loc+"/metadata.json"))

    file_set = []

    #going through the directory and appending every file to the dataset
    for file in files:
        all_files = glob.glob(file+'/*.jpeg')

        for i in range(len(all_files)):
            all_files[i] = all_files[i].replace('\\', '/')

        all_files.sort()

        for i in range(len(all_files)):
            all_files[i] = tf.constant(all_files[i], tf.string)

        file_set.append(all_files)

        file_name = file.replace(file_loc + "/", "")

        label = metadata[file_name]["label"]

        if(label == "FAKE"):
            labels.append(tf.constant(1, tf.float32))
        else:
            labels.append(tf.constant(0, tf.float32))

    #helper function
    def load_img_and_resize(list_files, labels):

        #need to use tensor arrays since the processing will generate a list which
        #tensorflow cannot use in computational graph
        img_stack = tf.TensorArray(tf.float32, size = num_frame)

        frame_number = 0

        for img_loc in list_files:
            img_loc = tf.io.read_file(img_loc)
            image = tf.image.decode_jpeg(img_loc, channels = 3)

            #resizing all images to same size
            image=tf.image.resize(image, img_size)

            #casting to float to ensure type safety
            image = tf.cast(image, tf.float32)

            #changing range of values from [0, 255] to [-1, 1]
            #helps in gradient flow
            image = image/127.0 - 1.0

            #stack the tensor array
            img_stack = img_stack.write(frame_number, image)

            #note that the tensor array demands an index value without which it
            #will not work
            frame_number = frame_number + 1

        holder = img_stack.stack()

        return holder, labels

    #making the sequential dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_set, labels))
    print("Built Dataset")
    dataset = dataset.map(load_img_and_resize)
    print("Mapped Dataset")
    dataset = dataset.shuffle(buffer_size = SHUFFLE_BUFFER).batch(BATCH_SIZE, drop_remainder = True).prefetch(tf.data.experimental.AUTOTUNE)
    print("Final Touches")

    return dataset
