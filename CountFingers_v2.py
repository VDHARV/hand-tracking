import cv2
from HandDetector import HandDetector


def CountFingersV2(detector, vid):
    detector = HandDetector(detectionCon = 0.75, trackCon = 0.75)
    vid = cv2.VideoCapture(0)
    i = 0
    while True:
        success, img = vid.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        if landmark_list:
            fingers = [(landmark_list[8][2], landmark_list[7][2], 0), 
                   (landmark_list[12][2], landmark_list[11][2], 1), 
                   (landmark_list[16][2], landmark_list[15][2], 2), 
                   (landmark_list[20][2], landmark_list[19][2], 3), 
                   (landmark_list[5][1], landmark_list[4][1], 4)]
            finger_up = [0, 0, 0, 0, 0]
            for finger in fingers:
                if finger[0] < finger[1]:
                    finger_up[finger[2]] = 1
            if finger_up[0]:
                if finger_up[1]:
                    if finger_up[2]:
                        if finger_up[3]:
                            if finger_up[4]:
                                cv2.putText(img, '5', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
                            else:
                                cv2.putText(img, '4', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
                        else:
                            cv2.putText(img, '3', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
                    else:
                        cv2.putText(img, '2', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
                else:
                    cv2.putText(img, '1', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
            else:
                cv2.putText(img, '0', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 5, (250, 0, 250), 3)
        cv2.imshow('Finger Count', img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
