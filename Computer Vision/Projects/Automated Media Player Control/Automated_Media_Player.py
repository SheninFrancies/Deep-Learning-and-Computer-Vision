import cv2
from cvzone import HandTrackingModule as HTM
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import time
import pyautogui as gui

'''
Use the Left Hand to control the Volume:
    Adjust the volume using the Thumb and Index Finger.
    Set the volume by holding down the Pinky Finger.

Use the Right Hand to Pause/Unpause the Video as well as Fast Forward/Rewind the Video screentime.
    Hold up the Thumb to Rewind the Video and the Pinky Finger to Forward the Video.
    Swiftly Hold up the Index Finger for a mini-second to Pause the Video. Remove the hand from the screen simultaneously.
        Repeat the Action to Unpause the Video.

Note: The Media Player is configured to work on Media Player that accomodates the keyboard's Right/Left/Spacebar combination to Forward/Rewind/Pause-Unpause the Video.
'''



capture = cv2.VideoCapture(0)
detector = HTM.HandDetector(detectionCon=False, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Range yields (-65.25, 0.0, 0.03125)
volumeRange = volume.GetVolumeRange()
volumeMinimum = volumeRange[0]
volumeMaximum = volumeRange[1]
volumeBar = 150
volumePercentage = 0
volumeSetColor = (255, 0, 0)

while True:
    success, frame = capture.read()
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, flipType=False)
    if hands:
        if hands[0]['type'] == 'Left':
            length, Info, frame = detector.findDistance(hands[0]['lmList'][4], hands[0]['lmList'][8], frame)        
            # Hand Range: 10, 130
            # Volume Range: -65.25, 0.0
            volumeBar = np.interp(length, [10, 130], [400, 150])
            volumePercentage = np.interp(length, [10, 130], [0, 100])
            volumeSmoothened = 10
            volumePercentage = volumeSmoothened * round(volumePercentage / volumeSmoothened)
            fingers_up = detector.fingersUp(hands[0])
            print(fingers_up)
            if not (fingers_up[4]):
                volume.SetMasterVolumeLevelScalar(volumePercentage / 100, None)
                cv2.circle(frame, (Info[4], Info[5]), 15, (0, 255, 0), cv2.FILLED)
                volumeSetColor = (0, 255, 0)
            else:
                volumeSetColor = (255, 0, 0)


            cv2.rectangle(frame, (20, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(frame, (20, int(volumeBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, f'{int(volumePercentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)    
    
            currentVolume = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(frame, f'Volume Set: {int(currentVolume)}', (350, 50), cv2.FONT_HERSHEY_COMPLEX, 1, volumeSetColor, 3)


        if hands[0]['type'] == "Right":
            fingers_up = detector.fingersUp(hands[0])
            if (fingers_up[0] and not (fingers_up[1] or fingers_up[2] or fingers_up[3] or fingers_up[4])):
                gui.press("left")
            if (fingers_up[4] and not (fingers_up[0] or fingers_up[1] or fingers_up[2] or fingers_up[3])):
                gui.press("right")
            if (fingers_up[1] and not (fingers_up[0] or fingers_up[2] or fingers_up[3] or fingers_up[4])):
                gui.press("space")
                time.sleep(0.5)


    cv2.imshow('Automated Media Player', frame)
    if cv2.waitKey(1) == 13:
        break    