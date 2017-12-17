import cv2
import os
import numpy as np


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img, face_recognizer, subjects):
    img = test_img.copy()
    faces, rects = detect_all_faces(img)
    predicted_rects = dict()

    for i, face in enumerate(faces):
        label, confidence = face_recognizer.predict(face)
        print("x: " + str(rects[i][0]) + " y: " + str(rects[i][1]) + " confidence: " + str(confidence))
        if confidence < 60:
            if subjects[label] not in predicted_rects or predicted_rects[subjects[label]][0] > confidence:
                predicted_rects[subjects[label]] = (confidence, rects[i])

    for label_text, (confidence, rect) in predicted_rects.items():
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)

    return img


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]


def detect_all_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(rects) == 0:
        return None, None

    faces = [None] * len(rects)
    for i, rect in enumerate(rects):
        (x, y, w, h) = rects[i]
        faces[i] = gray[y:y+w, x:x+h]

    return faces, rects


def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            print("Training on image...%s" % image_path)

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels
