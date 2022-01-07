import cv2
import numpy as np
from numpy.lib.type_check import imag
from HandDetector import HandDetector

green = (0, 128, 0) 
red = (255, 0, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)

def palette(img):
    
    colors = (green, red, blue, yellow)
    pos = (200, 400, 600, 800)
    
    cv2.circle(img, (pos[0], 100), 50, green, cv2.FILLED)
    cv2.circle(img, (pos[1], 100), 50, red, cv2.FILLED)
    cv2.circle(img, (pos[2], 100), 50, blue, cv2.FILLED)
    cv2.circle(img, (pos[3], 100), 50, yellow, cv2.FILLED)
    
    return pos, colors

def draw(img, store_line):    
    for sl in store_line:
        cv2.line(img, sl[0], sl[1], sl[2], 8)
        
def selector(pc, landmark_list, recent_color):
    detector = HandDetector()
    if detector.distance([pc[0][0], 100], landmark_list[8][1:]) < 50:
        return pc[1][0]
    elif detector.distance([pc[0][1], 100], landmark_list[8][1:]) < 50:
        return pc[1][1]
    elif detector.distance([pc[0][2], 100], landmark_list[8][1:]) < 50:
        return pc[1][2]
    elif detector.distance([pc[0][3], 100], landmark_list[8][1:]) < 50:
        return pc[1][3]
    else:
        return recent_color


def main():
    vid = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon = 0.8)     
    # store_points = list()
    # store_line = list()
    color = green
    while True:
        _, img = vid.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        pc = palette(img)
        
        
        # if hand is not visible
        if not landmark_list:
            pass
            # draw(img, store_line) # color should be selected one      
        else: # if hand is visible
            finger = detector.finger_up(landmark_list)
            if finger[0] and finger[1] == 1:
                color = selector(pc, landmark_list, color)
                cv2.rectangle(img, landmark_list[8][1:], landmark_list[12][1:], color, cv2.FILLED)
                print('Selection Mode')
            elif finger[0] == 1 and finger[1] != 1:
                cv2.circle(img, landmark_list[8][1:], 8, color, cv2.FILLED)
                print('Drawing Mode')
                # color = selector(pc, landmark_list)
                # draw(img, store_line)
            # else:
                # store_points.append(landmark_list[8][1:])
                # if len(store_points) > 2:
                #     store_line.append([store_points[-2], store_points[-1], color])
                #     draw(img, store_line)
                
        
        
        

        cv2.imshow('w', img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()
    

