from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Images&Videos/surgery-4.mp4")  # For Video
model = YOLO("surgicalequipment.pt")

classNames = ['Army_navy', 'Bulldog', 'Castroviejo', 'Forceps', 'Frazier',
              'Hemostat', 'Iris', 'Mayo_metz', 'Needle', 'Potts', 'Richardson',
              'Scalpel', 'Towel_clip', 'Weitlaner', 'Yankauer']


BoxColor = (0, 0, 255)

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',
                               (max(0, x1), max(35, y1)), scale=2, thickness=2,colorB=BoxColor,
                               colorT=(255,255,255),colorR=BoxColor, offset=10)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), BoxColor, 3)

    cv2.imshow("Surgical Equipment Detection", frame)
    if cv2.waitKey(1) == 13:
        break