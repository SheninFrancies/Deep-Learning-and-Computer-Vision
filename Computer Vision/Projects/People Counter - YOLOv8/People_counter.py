import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort

capture = cv2.VideoCapture("../Images&Videos/people.mp4")

model = YOLO("../YOLOWeights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread('../Images&Videos/mask_people.png')

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []

while True:
    success, frame = capture.read()
    frameRegion = cv2.bitwise_and(frame,mask)
    GraphicsCard = cv2.imread('../Images&Videos/graphics_people.png', cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, GraphicsCard, (730, 260))
    results = model(frameRegion, stream=True)

    detections = np.empty((0,5))

    for result in results:
        boxes = result.boxes
        # You can also use box.xywh
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.3:
                cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 2)
    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 2)

    for resultTracker in resultsTracker:
        x1, y1, x2, y2, ID = resultTracker
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(resultTracker)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'{int(ID)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        # cv2.circle(frame, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(ID) == 0:
                totalCountUp.append(ID)
                cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 3)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(ID) == 0:
                totalCountDown.append(ID)
                cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 3)


    # # cvzone.putTextRect(frame, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(frame, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_COMPLEX, 2, (139, 195, 75), 5)
    cv2.putText(frame, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_COMPLEX, 2, (50, 50, 230), 5)

    cv2.imshow("Webcam", frame)
    # cv2.imshow("Region", frameRegion)
    if cv2.waitKey(1) == 13:
        break
