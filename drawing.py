import cv2
import numpy as np
from HandDetector import HandDetector


def palette(img):
    cv2.circle(img, (200, 100), 50, (0, 255, 0), cv2.FILLED)
    cv2.circle(img, (400, 100), 50, (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (600, 100), 50, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (800, 100), 50, (0, 255, 100), cv2.FILLED)

def draw(img, store_line):    
    for sl in store_line:
        cv2.line(img, sl[0], sl[1], (100, 255, 100), 8)

def main():

    vid = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8)
    store_points = list()
    store_line = list()
    while True:
    
        _, img = vid.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img) # [index, height, width]
        palette(img)
        
        if landmark_list:
            
            distance = detector.distance(8, 12, landmark_list)
            if distance > 50:
                store_points.append(landmark_list[8][1:])
                if len(store_points) >= 2:
                    store_line.append([store_points[-1], store_points[-2]])
                draw(img, store_line)

        cv2.imshow('window', img)
        cv2.waitKey(1)
        
        
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
    

