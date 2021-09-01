'''
This file contains multiple helper functions used for preprocessing videos so that
they can be used for training our model
'''

import cv2
import os

#crops image based on bounding box
def crop_img(image, box):
    for x, y, w, h in box:
        image = image[y:y+h, x:x+w, :]

    return image

#face detection using OpenCV
def detect_face(image):

    cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    fc = cv2.CascadeClassifier(cfp)
    grayscaled_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(grayscaled_img, 1.1, 8)
    img = crop_img(image, faces)
    return img, faces

#slices the frames of the video and stores them individually
def video_slicer(vid_path, capture_sec = 5):

    max_frame_len = int(capture_sec * 60)
    cap = cv2.VideoCapture(vid_path)
    frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            if len(frames) == max_frame_len:
                break
    finally:
        cap.release()

    return frames
