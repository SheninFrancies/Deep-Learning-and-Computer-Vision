import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import torch
from torchvision import ops

# Video
capture = cv2.VideoCapture("../Images&Videos/cars.mp4")
# YOLOv8 Nano Model
model = YOLO("../YOLOWeights/yolov8n.pt")

# Define the classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]

# Mask for Region of Interest
mask = cv2.imread('../Images&Videos/mask_car.png')

# Tracker for Cars
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Crossing Line
limits = [400, 297, 673, 297]
# Counter for Cars
totalCount = []

while True:
    success, frame = capture.read()
    frameRegion = cv2.bitwise_and(frame,mask)
    # Get the mask and overlay it onto the frame
    GraphicsCard = cv2.imread('../Images&Videos/graphics_car.png', cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, GraphicsCard, (0, 0))
    results = model(frameRegion, stream=True)
    detections = np.empty((0,5))

    for result in results:
        boxes = result.boxes
        # You can also use box.xywh instead of box.xyxy
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # Ground Truth Bounding Box for IOU
            ground_truth_bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
    for resultTracker in resultsTracker:
        x1, y1, x2, y2, ID = resultTracker
        # Prediction Bounding Box for IOU
        prediction_bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Get the IOU value
        iou = ops.box_iou(ground_truth_bbox, prediction_bbox)
        if iou > 0.3:
            print('IOU : ', iou.numpy()[0][0])
            # cvzone.putTextRect(frame, f'IOU: {math.ceil(iou.numpy()[0][0] * 100)/100}', (1020, 50),
            #                    scale=2, thickness=3, offset=15, colorT=(139, 195, 75), colorR= (255, 255, 255))
            # A nicer text instead of the Rectangle
            cv2.putText(frame, f'IOU:{str(math.ceil(iou.numpy()[0][0] * 100) / 100)}', (1000, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (139, 195, 75), 5)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Center of the Car crosses the line limit --> Count the car
        # Also need to check the ID
        cx, cy = x1 + w // 2, y1 + h // 2
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(ID) == 0:
                totalCount.append(ID)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)

    cv2.putText(frame, str(len(totalCount)), (220, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (50, 50, 255), 5)

    cv2.imshow("Car Counter with IOU", frame)
    if cv2.waitKey(1) == 13:
        break
