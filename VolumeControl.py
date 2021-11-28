import numpy as np
import cv2
from HandDetector import HandDetector
import osascript

detector = HandDetector()
vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)
    if landmark_list:
        distance = np.linalg.norm(np.array(landmark_list[4][1:]) - np.array(landmark_list[8][1:]))
        print(distance)
        cv2.circle(img, tuple(landmark_list[4][1:]), 15, (120, 220, 220), cv2.FILLED)
        cv2.circle(img, tuple(landmark_list[8][1:]), 15, (120, 220, 220), cv2.FILLED)
        cv2.line(img, tuple(landmark_list[4][1:]), tuple(landmark_list[8][1:]), (220, 220, 220), 3)
        mid = (int((landmark_list[4][1] + landmark_list[8][1]) / 2),
               int((landmark_list[4][2] + landmark_list[8][2]) / 2))
        cv2.circle(img, mid, 15, (120, 220, 220), cv2.FILLED)
        if distance < 50:
            cv2.circle(img, mid, 15, (120, 0, 220), cv2.FILLED)
        volume = np.interp(distance, [30, 350], [0, 100])
        print(volume)
        osascript.osascript(f'set volume output volume {int(volume)}')

    cv2.imshow('Volume Control', img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
