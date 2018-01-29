import numpy as np
import cv2
import pyautogui as m

m.FAILSAFE = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


(scrx,scry)=m.size()
(camx, camy)=(32,300)

mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
DampingFactor = 3

def calculateView(x,y):
    xvMax, yvMax = m.size()
    xvMin, yvMin = 0, 0
    xwMax, xwMin = 182, 120
    ywMax, ywMin = 170, 140
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
        w = np.int(w)
        h = np.int(h)
        #just to check the data type-------------------------
        #print(type(x))
        #----------------------------------------------------


        #creating centre
        cx = x+w/2
        cy = y+h/2

        #print('x = {} and y = {}'.format(x, y))
        xv, yv = calculateView(cx,cy)
       # Viewport conversion---------------------------------

        #Not working

        print('cx = {} and cy = {}'.format(cx, cy))

        mouseLoc = mLocOld + ((xv,yv)-mLocOld)//DampingFactor


        print('nx = {} and ny = {}'.format(mouseLoc[0], mouseLoc[1]))

        m.moveTo(mouseLoc[0],mouseLoc[1],tween =  m.linear(.5))
        mLocOld = mouseLoc

    cv2.imshow('img',gray)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()