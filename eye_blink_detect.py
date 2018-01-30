# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyautogui as m

#taking size of screen
(scrx,scry)=m.size()

mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
DampingFactor = 15          #Supposed to make mouse smooth

def calculateView(x,y):
    xvMax, yvMax = m.size()
    xvMin, yvMin = 0, 0
    xwMax, xwMin = 400, 280
    ywMax, ywMin = 350, 280
    sx = (xvMax - 0) // (xwMax - xwMin)
    sy = (yvMax - 0) // (ywMax - ywMin)
    xv = xvMin + (x - xwMin) * sx
    yv = yvMin + (y - ywMin) * sy
    return xv,yv




def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # return the eye aspect ratio
    return ear
# construct the argument parse and parse the arguments


#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "shape_predictor_68_face_landmarks.dat", required=True,
#    help="path to facial landmark predictor")
#ap.add_argument("-v", "--video", type=str, default="",
#    help="path to input video file")
#args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = .15
EYE_AR_CONSEC_FRAMES = 7
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# start the video stream thread
print("[INFO] starting video stream thread...")

vs = cv2.VideoCapture(0)

#vs = FileVideoStream(0).start()
#fileStream = True

# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
#time.sleep(1.0)
# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    #if fileStream and not vs.more():
    #    break
 
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    ret, frame = vs.read()
    #frame = imutils.resize(frame, width=450)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    
        # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        
        print(leftEye[0])
        
        #get window coordinates
        
        xv, yv = leftEye[0]
        
        xw = np.int(xv)
        yw = np.int(yv)
        print(type(xv))
        xv,yv = calculateView(xw,yw)
         
#For mouse control        
        mouseLoc = mLocOld + ((xv,yv)-mLocOld)//DampingFactor
        print('nx = {} and ny = {}'.format(mouseLoc[0], mouseLoc[1]))
        m.moveTo(mouseLoc[0],mouseLoc[1],pause = 0,tween =  m.linear(.5))
        
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
 
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
       
        if rightEAR < .15:
            m.click(mouseLoc[0],mouseLoc[1],clicks = 1, button = 'left', pause = 0  )
            
        if leftEAR < .15:
            m.click(mouseLoc[0],mouseLoc[1],clicks = 1, button = 'right', pause = 0)
            
        mLocOld = mouseLoc

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
#        if ear < EYE_AR_THRESH:
#            COUNTER += 1
# 
#        # otherwise, the eye aspect ratio is not below the blink
#        # threshold
#        else:
#            # if the eyes were closed for a sufficient number of
#            # then increment the total number of blinks
#            if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                TOTAL += 1
# 
#            # reset the eye frame counter
#            COUNTER = 0
#            # draw the total number of blinks on the frame along with
#        # the computed eye aspect ratio for the frame
#        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
#            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
#            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()