# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import pyautogui as m

m.FAILSAFE = False
#taking size of screen
(scrx,scry)=m.size()

mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
DampingFactor = 15          # supposed to make mouse smooth

def calculateView(x,y):
    xvMax, yvMax = m.size()
    xvMin, yvMin = 0, 0
    xwMax, xwMin = 370, 270
    ywMax, ywMin = 290, 200
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
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
# start the video stream thread
print("[INFO] starting video stream thread...")

vs = cv2.VideoCapture(0)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    ret, frame = vs.read()
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
        nose = shape[nStart:nEnd]
        rightEye = shape[rStart:rEnd]

        print('nose')
        print(nose[0])
        
        # get window coordinates
        
        xv, yv = nose[0]
        
        xw = np.int(xv)
        yw = np.int(yv)
        print(type(xv))
        xv,yv = calculateView(xw,yw)
         
        # for mouse control        
        mouseLoc = mLocOld + ((xv,yv)-mLocOld)//DampingFactor
        print('nx = {} and ny = {}'.format(mouseLoc[0], mouseLoc[1]))
        m.moveTo(mouseLoc[0],mouseLoc[1],pause = 0,tween =  m.linear(.5))
    
        # calculate the eye-aspect-ratio(EAR) for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
 
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
        
        if rightEAR < .15:
            m.click(mouseLoc[0],mouseLoc[1],clicks = 1, button = 'left', pause = 0  )
            
        if leftEAR < .15:
            m.click(mouseLoc[0],mouseLoc[1],clicks = 1, button = 'right', pause = 0)
            
        mLocOld = mouseLoc

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` or 'esc' key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == 27:
        break
 
cv2.destroyAllWindows()