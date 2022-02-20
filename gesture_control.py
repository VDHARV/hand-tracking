import cv2
import numpy as np
from HandDetector import HandDetector

detector = HandDetector(detectionCon=0.75, trackCon=0.75)
vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)
    if landmark_list:
        finger_up = detector.finger_up(landmark_list)
        if finger_up == [0]*5:
            print("move")
        elif finger_up == [0,0,0,1,0]:
            print("right")
        elif finger_up == [0,0,0,0,1]:
            print("left")
        elif finger_up == [1]*5:
            print("back")
        else:
            print("stop")
    cv2.imshow("cam", img)
    cv2.waitKey(1)
