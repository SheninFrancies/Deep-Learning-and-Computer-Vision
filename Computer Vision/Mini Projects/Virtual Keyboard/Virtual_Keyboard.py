import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller
import cvzone
import imutils
import pygame

pygame.init()
pygame.mixer.init()
click_sound = pygame.mixer.Sound("click_sound.wav")
capture = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=False)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

KeyboardText = ""
 
keyboard = Controller()

class Button():
    def __init__(self, position, text, size = [85, 85]):
        self.position = position
        self.size = size
        self.text = text

def drawButtons(frame, buttonList):
    for button in buttonList:
        x, y = button.position
        w, h = button.size
        cvzone.cornerRect(frame, (button.position[0], button.position[1], button.size[0], 
                                  button.size[1]), 20, rt=0)
        cv2.rectangle(frame, button.position, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(frame, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return frame

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width = 1280, height = 720)

    frame = detector.findHands(frame)
    landmarks_list, bbox_Info = detector.findPosition(frame)
    frame = drawButtons(frame, buttonList)
    
    if landmarks_list:
        for button in buttonList:
            x, y = button.position
            w, h = button.size
 
            if x < landmarks_list[8][0] < x + w and y < landmarks_list[8][1] < y + h:
                cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(frame, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                length, _, _ = detector.findDistance(8, 12, frame, draw=False)
                print(length)
            
            # Button Click Action
            # Note: Length depends on distance from the screen
                if length < 30:
                    keyboard.press(button.text)
                    pygame.mixer.Sound.play(click_sound)
                    cv2.rectangle(frame, button.position, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    KeyboardText += button.text
                    sleep(0.2)
    
    cv2.rectangle(frame, (50, 350), (1000, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(frame, KeyboardText, (60, 420), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
 
    cv2.imshow('Virtual Keyboard', frame)
    if cv2.waitKey(1) == 13:
        break
