import numpy as np
import cv2
from HandDetector import HandDetector


detector = HandDetector(detectionCon=0.75, trackCon=0.75)
vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)
    if landmark_list:
        distance_8 = np.linalg.norm(np.array(landmark_list[8][1:]) - np.array(landmark_list[0][1:]))
        distance_12 = np.linalg.norm(np.array(landmark_list[12][1:]) - np.array(landmark_list[0][1:]))
        distance_16 = np.linalg.norm(np.array(landmark_list[16][1:]) - np.array(landmark_list[0][1:]))
        distance_20 = np.linalg.norm(np.array(landmark_list[20][1:]) - np.array(landmark_list[0][1:]))
        distance_4 = np.linalg.norm(np.array(landmark_list[4][1:]) - np.array(landmark_list[17][1:]))
        distances = [(distance_8, 300), (distance_12, 320), (distance_16, 300), (distance_20, 300), (distance_4, 250)]
        finger_up = [0, 0, 0, 0, 0]
        for index, distance in enumerate(distances):
            if distance[0] > distance[1]:
                finger_up[index] = 1
        if finger_up[0]:
            if finger_up[1]:
                if finger_up[2]:
                    if finger_up[3]:
                        if finger_up[4]:
                            cv2.putText(img, '5', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
                        else:cv2.putText(img, '4', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)

                    else: cv2.putText(img, '3', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
                else: cv2.putText(img, '2', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
            else:cv2.putText(img, '1', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
        else:cv2.putText(img, '0', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
    cv2.imshow('Volume Control', img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break