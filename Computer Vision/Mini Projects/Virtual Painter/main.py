import cv2
import numpy as np
import HandTrackingModule as HTM

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)    # Width
# cap.set(4, 720)     # Height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1020)

detector = HTM.handDetector()

# Default Color if No Selection is made
drawingColor = (0, 0, 0)

# Eraser Size & Brush Size
EraserSize = 30
BrushSize = 10

# Image Canvas
Canvas = np.zeros((720, 1080, 3), np.uint8)


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1080, 720))
    
    # Painter Selection Panel
    cv2.rectangle(frame, (0, 0), (1080, 110), (0, 0, 0), cv2.FILLED)    # Black Background
    cv2.rectangle(frame, (10, 10), (200, 100), (0, 0, 255), cv2.FILLED)     # Red
    cv2.rectangle(frame, (210, 10), (400, 100), (0, 255, 0), cv2.FILLED)    # Green
    cv2.rectangle(frame, (410, 10), (600, 100), (255, 0, 0), cv2.FILLED)    # Blue
    cv2.rectangle(frame, (610, 10), (800, 100), (0, 255, 255), cv2.FILLED)  # Yellow
    cv2.rectangle(frame, (810, 10), (1070, 100), (255, 255, 255), cv2.FILLED)   # White for Eraser
    cv2.putText(frame, "Eraser", (890, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)  # Eraser Text
    
    # Find Hand Landmarks
    frame = detector.findHands(frame)
    landmark_list = detector.findPosition(frame)    # List of 21 landmark values on the hand, x and y coordinates
    if len(landmark_list) != 0:
        # Landmark Index Finger Tip
        x1, y1 = landmark_list[8][1:]
        # Landmark Middle Finger Tip
        x2, y2 = landmark_list[12][1:]

    # Check which finger is UP
    fingers = detector.fingersUp()

    # Selection Mode - Index & Middle Fingers are UP
    if fingers[1] and fingers[2]:
        # Point on the finger
        xp, yp = (0, 0)
        # print('Selection Mode')
        
        if y1 < 120:
            # Select Red
            if  10 < x1 < 200:
                drawingColor = (0, 0, 255)
            
            # Select Green
            elif 210 < x1 < 400:
                drawingColor = (0, 255, 0)

            # Select Blue
            elif 410 < x1 < 600:
                drawingColor = (255, 0, 0)
            
            # Select Yellow
            elif 610 < x1 < 800:
                drawingColor = (0, 255, 255)

            elif 810 < x1 < 1070:
                drawingColor = (0, 0, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), drawingColor, cv2.FILLED)
    
    # Drawing Mode - ONLY Index finger is UP
    if (fingers[1] and not fingers[2]):
        cv2.circle(frame, (x1, y1), 5, drawingColor, thickness = -1)
        if xp == 0 and yp == 0:
            xp = x1
            yp = y1

        if drawingColor == (0, 0, 0):    
            cv2.line(frame, (xp, yp), (x1, y1), drawingColor, EraserSize)
            cv2.line(Canvas, (xp, yp), (x1, y1), drawingColor, EraserSize)
        else:
            cv2.line(frame, (xp, yp), (x1, y1), drawingColor, BrushSize)
            cv2.line(Canvas, (xp, yp), (x1, y1), drawingColor, BrushSize)
        
        xp, yp = x1, y1

    # Canvas Grayscale, Inverse and back to BGR
    GrayCanvas = cv2.cvtColor(Canvas, cv2.COLOR_BGR2GRAY)
    _ , CanvasInverse = cv2.threshold(GrayCanvas, 20, 255, cv2.THRESH_BINARY_INV)
    CanvasInverse = cv2.cvtColor(CanvasInverse, cv2.COLOR_GRAY2BGR)

    # Merge the Frame together
    frame = cv2.bitwise_and(frame, CanvasInverse)
    frame = cv2.bitwise_or(frame, Canvas)
    frame = cv2.addWeighted(frame, 1, Canvas, 0.5, 0)

    cv2.imshow('Virtual Painter', frame)
    if cv2.waitKey(1) == 13:
        break