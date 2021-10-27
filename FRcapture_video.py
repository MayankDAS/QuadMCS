import cv2
import numpy as np
import os
from datetime import datetime
cap = cv2.VideoCapture(0)
saveFrames = True
i = 0

while(True):
    ret, frame = cap.read()
    
    faceCascade = cv2.CascadeClassifier('G:\Github\opencv\data\lbpcascades\lbpcascade_frontalface.xml')  
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
    
    for (x, y, w, h) in faces:
        minFrame = w > 200 and h > 200
        if minFrame:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if saveFrames == True:
                faceslice = frame[y:y+h, x:x+w]
                faceslice = cv2.resize(faceslice, (100, 100))
#                 path = 'C:/SPB_Data/RTSP/dataset/faces-' + datetime.now().strftime('%d-%m-%Y')
                path = 'C:/SPB_Data/RTSP/dataset/faces-real'
                if os.path.exists(path):
                    pass
                else:
                    os.makedirs(path)
#                 image_name = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
                image_name = i;
                i=i+1
                filename = "%s.jpg" % (image_name)
                cv2.imwrite(os.path.join(path , filename), faceslice)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
