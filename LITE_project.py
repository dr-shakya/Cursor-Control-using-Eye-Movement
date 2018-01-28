import numpy as np
import cv2
import pyautogui as m

m.FAILSAFE = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def calculateView(x,y):
    xvMax, yvMax = m.size()
    xvMin, yvMin = 0, 0
    xwMax, xwMin = 120, 70
    ywMax, ywMin = 90, 60
    sx = (xvMax - 0) // (xwMax - xwMin)
    sy = (yvMax - 0) // (ywMax - ywMin)
    xv = xvMin + (x - xwMin) * sx
    yv = yvMin + (y - ywMin) * sy
    return xv,yv


cap = cv2.VideoCapture( 0)

while 1:
    ret, img = cap.read()

   # img2 = cv2.flip(img, 1)
   # img2 = cv2.resize(img2,(320,240))
   # img2 = cv2.bilateralFilter(img2,9,75,75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)
    gray = cv2.resize(gray, (320, 240))
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
      #  roi_color = img2[y:y+h, x:x+w]
        x = np.int(x)
        y = np.int(y)
        #just to check the data type-------------------------
        print(type(x))
        #----------------------------------------------------

        print('x = {} and y = {}'.format(x, y))
        xv, yv = calculateView(x,y)
       # Viewport conversion---------------------------------

        #Not working


        print('sx = {} and sy = {}'.format(xv, yv))


        m.moveTo(xv,yv,tween =  m.linear(.5))

    cv2.imshow('img',gray)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()