'''
"Drowsiness Detection"

Utilizes OpenCV and Mediapipe to evaluate if the eyes of a person is closed or not for a particular interval.
One could use a timeframe, but here a the camera's framerate is taken as an influencing factor.
When the eyes are closed for a particular period, an audio alert message is (triggered and/or) played automatically.

The FaceMeshModule file needs to be in the same directory as this file in order to seamlessly run the program.
'''

import cv2
import FaceMeshModule as FMM
import threading
import pyttsx3

capture = cv2.VideoCapture(0)
detector = FMM.eyeDetector(maxFaces=1)

# Face and Eyes landmarks
LEFT_EYE_TOP_BOTTOM = detector.LEFT_EYE_TOP_BOTTOM
LEFT_EYE_LEFT_RIGHT = detector.LEFT_EYE_LEFT_RIGHT
RIGHT_EYE_TOP_BOTTOM = detector.RIGHT_EYE_TOP_BOTTOM
RIGHT_EYE_LEFT_RIGHT = detector.RIGHT_EYE_LEFT_RIGHT
FACE = detector.FACE


speech = pyttsx3.init()

def speech_alert(speech, speech_message):
    speech.say(speech_message)
    speech.runAndWait()


# Range Parameters
min_tolerance = 5.0
frame_count = 0
min_frame = 6

while True:
    success, frame = capture.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.faceMesh.process(frameRGB)

    if results.multi_face_landmarks:
        detector.draw_landmarks(frame, results, FACE, (0, 255, 0))
        detector.draw_landmarks(frame, results, LEFT_EYE_TOP_BOTTOM, (0, 0, 255))
        detector.draw_landmarks(frame, results, LEFT_EYE_LEFT_RIGHT, (0, 0, 255))
        # Get the aspect ratio of the left eye
        ratio_left =  detector.get_aspect_ratio(frame, results, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
        
        detector.draw_landmarks(frame, results, RIGHT_EYE_TOP_BOTTOM, (0, 0, 255))
        detector.draw_landmarks(frame, results, RIGHT_EYE_LEFT_RIGHT, (0, 0, 255))
        # Get the aspect ratio of the right eye
        ratio_right =  detector.get_aspect_ratio(frame, results, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
        
        # Average Aspect Ratio - eyes opening/closure 
        aspect_ratio = (ratio_left + ratio_right) / 2.0

        # A Neat Tesselation that blankets the face
        for face_landmarks in results.multi_face_landmarks:
            detector.mpDraw.draw_landmarks(frame, face_landmarks, detector.mpFaceMesh.FACEMESH_TESSELATION, None, detector.mpDrawStyles.get_default_face_mesh_tesselation_style())
        
        if aspect_ratio > min_tolerance:
            frame_count += 1
        else:
            frame_count = 0
        if frame_count > min_frame:
            message = 'Drowsy Alert: Wake up!'
            saymessage = threading.Thread(target = speech_alert, args = (speech, message))
            saymessage.start()

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) == 13:    # Enter key to Exit
        break
    
capture.release()
cv2.destroyAllWindows()