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

def build_dataset(location = 'D:\deepfake_database\dfdc_train_part_1', dataset_type = "train", capture_sec = 5, img_size = [100,100], shuffle_buffer = 401):

    # <------------------------------------ tf.data parameters ------------------------------------>

    BATCH_SIZE = batch_size
    SHUFFLE_BUFFER = shuffle_buffer

    # <------------------------------------ image parameters ------------------------------------>
    capture_sec = capture_sec
    num_frame = int(capture_sec * 60)
    img_size = img_size


    # <------------------------------------ data preprocessing ------------------------------------>
    # preprocessing the videos by slicing their frames, detecting faces, and providing
    #names of the images
    names, _ , labels = processing.generate(path_root = location, capture_sec = capture_sec)

    #converting to valid filepaths
    for i in range(len(names)):
        names[i] = dataset_type + "_"+ 'files' + "/" + names[i]

    #converting float labels to tensor values
    for i in range(len(labels)):
        labels[i] = tf.constant(labels[i], tf.float32)

    file_set = []

    #going through the directory and appending every file to the dataset
    for file in names:

        all_files = glob.glob(file+'/*.jpg')

        for i in range(len(all_files)):
            all_files[i] = all_files[i].replace('\\', '/')

        all_files.sort()

        for i in range(len(all_files)):
            all_files[i] = tf.constant(all_files[i], tf.string)

        file_set.append(all_files)

    #making the sequential dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_set, labels))

    #helper function
    def load_img_and_resize(list_files, labels):

        #need to use tensor arrays since the processing will generate a list which
        #tensorflow cannot use in computational graph
        img_stack = tf.TensorArray(tf.float32, size = num_frame)

        frame_number = 0

        for img in list_files:
            img = tf.io.read_file(img)
            image = tf.image.decode_jpeg(img, channels = 3)

            #resizing all images to same size
            image=tf.image.resize_with_pad(image, target_height = img_size[0], target_width = img_size[1])

            #casting to float to ensure type safety
            image = tf.cast(image, tf.float32)

            #changing range of values from [0, 255] to [-1, 1]
            #helps in gradient flow
            image = image/127.0 - 1.0

            #stack the tensor array
            img_stack.write(frame_number, image)

            #note that the tensor array demands an index value without which it
            #will not work
            frame_number = frame_number + 1

        holder = img_stack.stack()

        return holder, labels

    dataset = dataset.map(load_img_and_resize)

    dataset = dataset.shuffle(buffer_size = SHUFFLE_BUFFER).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
