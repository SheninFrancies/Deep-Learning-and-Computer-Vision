import cv2
import PoseModule as PM
import imutils
import numpy as np

capture = cv2.VideoCapture('Images and Videos\squat.mp4')

detector = PM.poseDetector()
count = 0
direction = 0

while True:
    success, frame = capture.read()
    frame = imutils.resize(frame, width=1280, height=720)

    frame = detector.findPose(frame, draw = False)
    landmarks_list = detector.findPosition(frame, draw = False)

    if len(landmarks_list) != 0:
        # This does not start from zero
        # angle = detector.findAngle(frame, 23, 25, 27)
        angle = detector.findAngle(frame, 24, 26, 28)

    percentage = np.interp(angle, (45, 170), (0, 100))
    
    if percentage == 100:
        if direction == 0:
            count  += 0.5
            direction = 1
    if percentage == 0:
        if direction == 1:
            count += 0.5
            direction = 0
    print(count)

    cv2.putText(frame, str(int(count)), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(frame, str(int(percentage)), (1090, 90), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    bar = np.interp(angle, (45, 170), (700, 100))

    cv2.rectangle(frame, (1100, 100), (1200, 700), (0, 255, 0), 3)
    cv2.rectangle(frame, (1100, int(bar)), (1200, 700), (0, 255, 0), cv2.FILLED)

    cv2.imshow('Squat Counter', frame)
    if cv2.waitKey(1) == 13:
        break
