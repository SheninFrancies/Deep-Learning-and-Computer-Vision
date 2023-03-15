import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

class eyeDetector():
    def __init__(self, mode=False, maxFaces=1, refineLandmarks=False, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.LEFT_EYE_TOP_BOTTOM = [386, 374]
        self.LEFT_EYE_LEFT_RIGHT = [263, 362]
        self.RIGHT_EYE_TOP_BOTTOM = [159, 145]
        self.RIGHT_EYE_LEFT_RIGHT = [133, 33]
        self.FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
              377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.refineLandmarks, 
                                                 self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles


    def draw_landmarks(self, image, results, multi_face_landmarks, color):
        height, width = image.shape[:2]
             
        for face_landmark in multi_face_landmarks:
            point = results.multi_face_landmarks[0].landmark[face_landmark]
            point_scale = ((int)(point.x * width), (int)(point.y * height))
            cv2.circle(image, point_scale, 2, color, 1)

    def euclidean_distance(self, image, top, bottom):
        height, width = image.shape[0:2]
        point1 = int(top.x * width), int(top.y * height)
        point2 = int(bottom.x * width), int(bottom.y * height)
        distance = dist.euclidean(point1, point2)
        return distance
    
    def get_aspect_ratio(self, image, results, distance_top_bottom, distance_left_right):
        landmark = results.multi_face_landmarks[0]     
        top = landmark.landmark[distance_top_bottom[0]]
        bottom = landmark.landmark[distance_top_bottom[1]]
        distance_TB = eyeDetector.euclidean_distance(self, image, top, bottom)
        left = landmark.landmark[distance_left_right[0]]
        right = landmark.landmark[distance_left_right[1]]
        distance_LR = eyeDetector.euclidean_distance(self, image, left, right)
        aspect_ratio = distance_LR / distance_TB
        return aspect_ratio
    

def main():
        capture = cv2.VideoCapture(0)
        detector = eyeDetector()
        LEFT_EYE_TOP_BOTTOM = detector.LEFT_EYE_TOP_BOTTOM
        LEFT_EYE_LEFT_RIGHT = detector.LEFT_EYE_LEFT_RIGHT
        RIGHT_EYE_TOP_BOTTOM = detector.RIGHT_EYE_TOP_BOTTOM
        RIGHT_EYE_LEFT_RIGHT = detector.RIGHT_EYE_LEFT_RIGHT
        FACE = detector.FACE

        while True:
            _, frame = capture.read()
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.faceMesh.process(frameRGB)
            if results.multi_face_landmarks:
                detector.draw_landmarks(frame, results, FACE, (0, 255, 0))
                detector.draw_landmarks(frame, results, LEFT_EYE_TOP_BOTTOM, (0, 0, 255))
                detector.draw_landmarks(frame, results, LEFT_EYE_LEFT_RIGHT, (0, 0, 255))
                detector.draw_landmarks(frame, results, RIGHT_EYE_TOP_BOTTOM, (0, 0, 255))
                detector.draw_landmarks(frame, results, RIGHT_EYE_LEFT_RIGHT, (0, 0, 255))

                # ratio_left =  detector.get_aspect_ratio(frame, results, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                # ratio_right =  detector.get_aspect_ratio(frame, results, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                # aspect_ratio = (ratio_left + ratio_right) / 2.0

            for face_landmarks in results.multi_face_landmarks:
                detector.mpDraw.draw_landmarks(frame, face_landmarks, detector.mpFaceMesh.FACEMESH_TESSELATION, None, 
                                               detector.mpDrawStyles.get_default_face_mesh_tesselation_style())
        
            cv2.imshow("Eye Detection", frame)
            if cv2.waitKey(1) == 13:
                break

if __name__ == "__main__":
    main()
