import cv2
import mediapipe as mp
import time

import numpy as np


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        landmark_list = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, model_complexity=1,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True):

        self.landmark_List = []
        if self.result.multi_hand_landmarks:
            single_Hand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(single_Hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_List.append([id, cx, cy])
                if draw:
                    self.mpDraw.draw_landmarks(img, single_Hand,
                                               self.mpHands.HAND_CONNECTIONS)

        return self.landmark_List

    def finger_up(self, landmark_list):

        self.fingers = [(landmark_list[8][2], landmark_list[7][2], 0),
                   (landmark_list[12][2], landmark_list[11][2], 1),
                   (landmark_list[16][2], landmark_list[15][2], 2),
                   (landmark_list[20][2], landmark_list[19][2], 3),
                   (landmark_list[5][1], landmark_list[4][1], 4)]

        self.fingerup = [0, 0, 0, 0, 0]
        for finger in self.fingers:
            if finger[0] < finger[1]:
                self.fingerup[finger[2]] = 1
        return self.fingerup
    
    def distance_tips(self, index1, index2, landmark_list):
        return np.linalg.norm(np.array(landmark_list[index1][1:]) - np.array(landmark_list[index2][1:]))

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
