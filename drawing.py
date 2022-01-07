import cv2
import numpy as np
from HandDetector import HandDetector

# Screen Size
SIZE = (720, 1280, 3)

# Thickness
BRUSH_THICKNESS = 5
ERASER_THICKNESS = 50

# Colors
GREEN = (0, 128, 0) 
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (0, 255, 255)
ERASER = (0, 0, 0)


def palette(img):
    """ Show palette colors. """
    colors = (GREEN, RED, BLUE, ERASER)
    pos = (200, 400, 600, 800)
    
    cv2.circle(img, (pos[0], 100), 50, GREEN, cv2.FILLED)
    cv2.circle(img, (pos[1], 100), 50, RED, cv2.FILLED)
    cv2.circle(img, (pos[2], 100), 50, BLUE, cv2.FILLED)
    cv2.circle(img, (pos[3], 100), 50, ERASER, cv2.FILLED)
    
    return pos, colors
        
def selector(pc, landmark_list, recent_color):
    """ Select the color. """
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
    xp, yp = 0, 0
    color = GREEN
    blackboard = np.zeros(SIZE, np.uint8)
    
    while True:
        _, img = vid.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)
        pc = palette(img)

        # if hand is not visible
        if not landmark_list:
            cv2.putText(img, 'No hand Detected', (600, 600), cv2.FONT_HERSHEY_PLAIN, 3, YELLOW, BRUSH_THICKNESS)
            
        else: # if hand is visible
            finger = detector.finger_up(landmark_list)
            
            if finger[0] and finger[1]: # selection mode
                color = selector(pc, landmark_list, color)
                cv2.rectangle(img, landmark_list[8][1:], landmark_list[12][1:], color, cv2.FILLED)
                xp, yp = landmark_list[8][1:]
                print('Selection Mode')
            
            elif finger[0] and not finger[1]: # drawing mode
                cv2.circle(img, landmark_list[8][1:], 8, color, cv2.FILLED)
            
                if xp == 0 and yp == 0: # first point 
                    xp, yp = landmark_list[8][1:]
                
                if color == ERASER: # selecting eraser
                    cv2.line(blackboard, (xp, yp), landmark_list[8][1:], color, ERASER_THICKNESS) # drawing on canvas
                else: # selecting any color
                    cv2.line(blackboard, (xp, yp), landmark_list[8][1:], color, BRUSH_THICKNESS) # drawing on canvas
                    
                xp, yp = landmark_list[8][1:]
                print('Drawing Mode')
                
        imgGRAY = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
        
        _, imgINV = cv2.threshold(imgGRAY, 0, 255, cv2.THRESH_BINARY_INV)
        imgINV = cv2.cvtColor(imgINV, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgINV)
        img = cv2.bitwise_or(img, blackboard)
        cv2.imshow('w', img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()
    

