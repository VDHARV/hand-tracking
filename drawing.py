import cv2
from HandDetector import HandDetector

detector = HandDetector()

vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()
    img = detector.find_hands(img)
    cv2.imshow('window', img)
    cv2.waitKey(1)
    
    
cv2.destroyAllWindows()
