import cv2
import os
import time
import uuid
import mediapipe
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import time
from Graph_BD import GraphUse

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
a = GraphUse()


offset = 20
imgSize = 300
labels = ["call", "hello", "luck", "stop", "yes"]
# counter = 0
# folder = "Data/call"
def start():
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgRecize = cv2.resize(imgCrop, (wCal, imgSize))
                imgRecizeShape = imgRecize.shape
                wGap = math.ceil((imgSize - wCal) / 2)  # для центрирования изображения
                imgWhite[:, wGap:wCal + wGap] = imgRecize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgRecize = cv2.resize(imgCrop, (imgSize, hCal))
                imgRecizeShape = imgRecize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgRecize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            print(a.get_image_name(labels[index]))
            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 4), 4)

            # cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)


