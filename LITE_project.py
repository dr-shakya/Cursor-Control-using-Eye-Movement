import numpy as np
import cv2
import pyautogui as m
m.FAILSAFE = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture( 0)

while 1:
    ret, img = cap.read()

    img2 = cv2.flip(img, 1)
    img2 = cv2.resize(img2,(200,100))
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        x = np.int(x)
        y = np.int(y)
        #just to check the data type-------------------------
        print(type(x))
        #----------------------------------------------------


        # Viewport conversion---------------------------------
        xvMax, yvMax = m.size()
        xvMin, yvMin = 0, 0
        xwMax , xwMin = 240, 80
        ywMax , ywMin = 205, 70
        sx = (xvMax-0)//(xwMax-xwMin)
        sy = (yvMax - 0) // (ywMax - ywMin)
        xv =700 - xvMin + (x - xwMin) * sx
        yv = yvMin + (y - ywMin) * sy
        #Not working


        print('x = {} and y = {}'.format(xv, yv))


        m.moveTo(sx*x,sy*y,.1)

    cv2.imshow('img',img2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()