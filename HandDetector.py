import cv2
import mediapipe as mp
import time


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

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        if len(lmList) != 0:
            # getting the tracking points of hand
            pass

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()