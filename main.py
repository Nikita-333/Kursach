import cv2
import os
import time
import uuid
import mediapipe
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time

'''Фрагмент где добавляется картинки наших жестов'''

# IMAGES_PATH = 'D:/1/1/workspace/images'
#
# labels = ['hello','thanks','yes','no','iloveyou']
# number_imgs = 15
#
# for label in labels:
#
#     path = 'D:/1/1/workspace/images//'
#     os.mkdir(path+label)
#     cap = cv2.VideoCapture(0)
#     print('Collecting images for {}'.format(label))
#     time.sleep(5)
#     for imgnum in range(number_imgs):
#         ret,frame = cap.read()
#         imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
#         cv2.imwrite(imgname,frame)
#         cv2.imshow('frame',frame)
#         time.sleep(2)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0
folder = "Data/call"
while True:
    success, img = cap.read()
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

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgRecize = cv2.resize(imgCrop, (imgSize, hCal))
            imgRecizeShape = imgRecize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgRecize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s") and counter<=150:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
