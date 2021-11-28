import cv2
import numpy as np
from HandDetector import HandDetector
import autopy

vid = cv2.VideoCapture(0)

wScr, hScr = autopy.screen.size()
wCam, hCam = 640, 480
vid.set(3, wCam)
vid.set(4, hCam)

detector = HandDetector(detectionCon=0.75, trackCon=0.75)

while True:
    success, img = vid.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)
    if landmark_list:
        distance = np.linalg.norm(np.array(landmark_list[8][1:]) - np.array(landmark_list[12][1:]))
        finger_up = detector.finger_up(landmark_list)
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        x3 = np.interp(x1, (0, wCam), (0, wScr))
        y3 = np.interp(y1, (0, hCam), (0, hScr))

        if finger_up[0] and not finger_up[2]:
            try:
                autopy.mouse.smooth_move(wScr - x3, y3)
                cv2.circle(img, tuple(landmark_list[8][1:]), 15, (250, 0, 250), cv2.FILLED)
            except:
                continue

        if distance < 40:
            cv2.circle(img, tuple(landmark_list[12][1:]), 15, (250, 0, 250), cv2.FILLED)
            autopy.mouse.click()

    cv2.imshow('Mouse', img)
    cv2.waitKey(1)
