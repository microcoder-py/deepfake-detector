'''
This file contains a function that extracts frames from a set of specified videos,
stores them as .jpg files. and then extracts faces from the said images, builds
out the directory, and provides all file names to the dataset generator
'''

import json
import glob
import os
import cv2
import numpy as np
import video_proc_function as proc

def generate(path_root = 'D:\deepfake_database\dfdc_train_part_1', dataset_type = "train", capture_sec = 5):

    #find all files
    files = glob.glob(path_root + '/*.mp4')
    metadata = json.load(open(path_root + '/metadata.json'))
    list_name = []
    list_file_loc = []
    list_label = []


    for file in files:

        #code written on Windows10 which uses \ instead of / for directory path
        #separator
        file_name=file.replace(path_root, "").replace('\\', '')
        label_str = metadata[file_name]['label']

        label = 0.0

        if label_str == 'FAKE':
            label = 1.0

        list_name.append(file_name)
        list_file_loc.append(file)
        list_label.append(label)

    os.makedirs(dataset_type + "_" + 'files')

    for i in range(len(list_name)):

        #use helper function to slice up the video into numpy frames
        imgs = proc.video_slicer(list_file_loc[i], capture_sec = capture_sec)
        root_name = list_name[i]

        #build directory for storing our training images
        root_folder = dataset_type + "_" + 'files/' + root_name
        os.makedirs(root_folder)

        box = []

        #small function to help with file sorting later
        #basically appends the right number of 0s to filename
        #so that when we want to generate the sequence, it sorts correctly
        def file_name_generator(j):
            filename = ''
            if(j<10):
                file_name = "00"+str(j)+ ".jpg"
            elif(j<100 and j>=10):
                file_name = "0" + str(j) + ".jpg"
            else:
                file_name = str(j)+ ".jpg"
            return file_name

        list_bound_box = []
        list_unchanged_images = []

        for j in range(len(imgs)):
            #write image to location
            file_loc = root_folder + "/" + file_name_generator(j)
            cv2.imwrite(file_loc, imgs[j])

            #read file, extract face
            faceImg = cv2.imread(file_loc)
            faceCut, box = proc.detect_face(faceImg)

            #if face was detected, write the file, and append current box to list
            if(faceCut.size > 0 and np.size(box) > 0):
                cv2.imwrite(file_loc, faceCut)
                list_bound_box.append(box)
            else:
                list_unchanged_images.append(file_loc)

        #if there are any images where faces weren't detected, use the average
        #of the above bounding boxes to generate new bounding box
        #Assumption is that the face of the person does not move too often
        if(len(list_bound_box) > 0):
            boxes = np.array(list_bound_box)
            avg_box = np.array([[int(np.average(boxes[:, :, 0])), int(np.average(boxes[:,:, 1])), int(np.average(boxes[:,:, 2])), int(np.average(boxes[:,:, 3]))]])

            #use above bounding box to find faces
            for img in list_unchanged_images:
                image = cv2.imread(img)
                faceCut = proc.crop_img(image, avg_box)
                cv2.imwrite(img, faceCut)

    #returning all necessary information to generate the dataset
    return list_name, list_file_loc, list_label
