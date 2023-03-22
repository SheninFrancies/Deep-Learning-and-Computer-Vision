from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("../Images&Videos/ppe-3.mp4")  # For Video

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO Hardhat', 'NO Mask', 'NO Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

BoxColor = (0, 0, 255)

while True:
    success, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if conf > 0.5:
                if currentClass =='NO Hardhat' or currentClass =='NO Safety Vest' or currentClass == "NO Mask":
                    BoxColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    BoxColor =(0,255,0)
                else:
                    BoxColor = (255, 0, 0)

                cvzone.putTextRect(frame, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=2,colorB=BoxColor,
                                   colorT=(255,255,255),colorR=BoxColor, offset=10)
                cv2.rectangle(frame, (x1, y1), (x2, y2), BoxColor, 3)

    cv2.imshow("PPE Detection", frame)
    if cv2.waitKey(1) == 13:
        break