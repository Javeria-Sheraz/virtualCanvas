import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import os

##### FUNCTIONS
def drawHandLandmarks(image, results, hands_module, draw_utility):
    if results.multi_hand_landmarks:
        for dots in results.multi_hand_landmarks:
            draw_utility.draw_landmarks(image, dots, hands_module.HAND_CONNECTIONS,
                                        draw_utility.DrawingSpec((64, 64, 64), -1, 4),
                                        draw_utility.DrawingSpec((255, 255, 255), 1))


def get_landmarkList(results, image):
    lmList = []
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        handLmks = results.multi_hand_landmarks[0]
        # loops through indexes(0,8,12...) and lm(x,y,z values)
        for idx, lm in enumerate(handLmks.landmark):
            # finds position of lm (Landmarks) in pixels and converts them in integers
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmList.append([idx, cx, cy])
    return lmList


def fingersUp(lmlist):
    fingers = []
    tipsIdx = [4, 8, 12, 16, 20]
    if len(lmlist) ==0:
        return []
    #As thumb moves sideways on the screen, we check it x coodinates [1]
    # if lmlist[4][1] < lmlist[3][1]:
    if lmlist[tipsIdx[0]][1] < lmlist[tipsIdx[0]-1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Left a finger(tip) is, the smaller its x-number becomes

    # Checking for other fingers with tipidx 1 to 4
    for id in range(1, 5):
        if lmlist[tipsIdx[id]][2] < lmlist[tipsIdx[id]-2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    # Higher a finger is, the smaller its y-number becomes. This checks the tip and two indexes below it
    return fingers

##### Other variables initilization
Brushtext = "Red"
drawColour = (0, 0, 255)
Brushthickness = 3
xp, yp = 0, 0
# Taking a blank canvas to draw
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


# Initialization
widthCam = 1280
heightCam = 720

#THESE LINES CAPTURES FRAME FROM WEBCAM
webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, widthCam)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, heightCam)

#THESE LINES INITIALIZE THE DRAWING OF LANDMARKS ON HANDS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# This gets a list of all files inside the Header folder
folderPath = "Header"
myList = os.listdir(folderPath)
#This reads all the images from Header file and adds to overlayList so all UI images are loaded
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList)) #for debugging (should be 7)
header = overlayList[0]


while True:
    success, captures = webcam.read()
    captures = cv.flip(captures, 1)
    if success:
        colorConversion = cv.cvtColor(captures, cv.COLOR_BGR2RGB)
        result = hands.process(colorConversion)

        # THIS FUNCTION DRAWS LANDMARKS ON HANDS IF A FRAME IS SUCCESSFULLY CAPTURED BY WEBCAM
        # Uncomment this line to draw them
        # drawHandLandmarks(captures, result, mp_hands, mp_drawing)

        # This function gets the list of landmarks
        LmList = get_landmarkList(result, captures)

        # Get 1st and 2nd (last) element from list Lmlist and assign x, y coordinates to identify index and middle finger tip
        if len(LmList) !=0:
            # tip of index finger, taking points from 1 till the end
            x1, y1 = LmList[8][1], LmList[8][2]
            # tip of middle finger
            x2, y2 = LmList[12][1:]

        fingers = fingersUp(LmList)
        # print(fingers)

        #Makes sure fingers list has 5 fingers detected
        if len(fingers) >= 5:

            # For selection both index and middle finger should be up others should be down (optional for thumb)
            if  fingers[1] ==1 and fingers[2] ==1 and fingers[3] == 0 and fingers[4] == 0:
                xp, yp =0, 0
                cv.putText(captures, "Selection Mode", (captures.shape[1]-270, captures.shape[0]-13), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 8)
                cv.putText(captures, "Selection Mode", (captures.shape[1]-270, captures.shape[0]-13), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
                # checking if our click is in the header
                if y2 < 120:
                    if 120< x2< 250:
                        header = overlayList[0]
                        Brushtext = "Red"
                        drawColour = (0, 0, 255)
                    elif 320< x2< 400:
                        header = overlayList[1]
                        Brushtext = "Orange"
                        drawColour = (0, 94, 255)
                    elif 450< x2< 520:
                        header = overlayList[2]
                        Brushtext = "Purple"
                        drawColour = (255, 0, 154)
                    elif 690 < x2 < 760:
                        header = overlayList[3]
                        Brushtext = "Blue"
                        drawColour = (150, 0, 0)
                    elif 810 < x2 < 880:
                        header = overlayList[4]
                        Brushtext = "Pink"
                        drawColour = (102, 0, 204)
                    elif 920 < x2 < 1050:
                        header = overlayList[5]
                        Brushtext = "Green"
                        drawColour = (0, 255, 0)
                    elif 1130<x2<1200:
                        header = overlayList[6]
                        Brushtext = "Erasing"
                        drawColour = (64, 64, 64)
                        if Brushtext == "Erasing":
                            imgCanvas = cv.addWeighted(imgCanvas, 0.5, np.zeros_like(captures), 0.15, 0)
                cv.putText(captures, text=Brushtext, org=(captures.shape[1] - 180, captures.shape[0] - 50),
                           fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1.7, color=(255, 255, 255), thickness=7)
                cv.putText(captures, text=Brushtext, org =(captures.shape[1] - 180, captures.shape[0] - 50),
                           fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1.7, color=drawColour, thickness=3)
                cv.circle(captures, (x2, y2), 13, drawColour, -1)


            # For drawing only index finger should be up (middle finger should be down for drawing)
            elif fingers[1] ==1 and fingers[2] == 0:
                cv.circle(captures, (x1, y1), 12, drawColour, -1)
                if Brushtext == "Erasing":
                    cv.putText(captures, "Erasing Mode", (captures.shape[1]-250, captures.shape[0]-13), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 8)
                    cv.putText(captures, "Erasing Mode", (captures.shape[1]-250, captures.shape[0]-13), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
                else:
                    cv.putText(captures, "Drawing Mode", (captures.shape[1]-250, captures.shape[0]-13), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 8)
                    cv.putText(captures, "Drawing Mode", (captures.shape[1]-250, captures.shape[0]-13), cv.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
                    #capture.shapes = [h, w, c] has three elements in list, h = y = [1] and w = x = [0], put text requires origin (x,y)
                    cv.putText(captures, text=Brushtext, org=(captures.shape[1] - 180, captures.shape[0] - 50),
                               fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1.7, color=(255, 255, 255), thickness=7)
                    cv.putText(captures, text=Brushtext, org=(captures.shape[1] - 180, captures.shape[0] - 50),
                               fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1.7, color=drawColour, thickness=3)

                # Without this condition, the first line on screen will be drawn from point (0,0) to the current point
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                # if anything other than eraser is selected start drawing the line
                if Brushtext!= "Erasing":
                    cv.line(captures, (xp, yp), (x1, y1), drawColour, Brushthickness)
                    cv.line(imgCanvas, (xp, yp), (x1, y1), drawColour, Brushthickness)
                xp, yp = x1, y1

            # If drawing mode not active, reset starting point
            else:
                xp, yp = 0, 0

    imgGrey = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGrey, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    captures = cv.bitwise_and(captures, imgInv)
    captures = cv.bitwise_or(captures, imgCanvas)
    # Combine the drawing with the camera feed:
    # First, erase the drawing area from the camera image using a mask
    #         We use and to keep only the parts of camera that was not drawn on
    #         AND works like a stencil; it returns true for unused areas on both camera feed and imgInv
    # Then, add only the drawing on top so it looks neat and not messy.
    #         We use or to add imgCanvas drawing back on top of camera image


    #Setting the header image
    captures[0:120, 0:1280] = header
    # This will add two images then blend them
    captures = cv.addWeighted(captures, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("WELCOME TO THE PAINTING STATION", captures)
    # cv.imshow("Canvas", imgCanvas)
    # cv.imshow("Inverse", imgInv)
    key = cv.waitKey(20)

    #PRESS Esc key to EXIT the video frame
    if key == 27:
        break

webcam.release()
cv.destroyAllWindows()


#Brush colours in BGR format
#Red = (0, 0, 255)
#Orange = (0, 94, 255)
#Green = (0, 255, 0)
#Purple = (255, 0, 154)
#Blue = (150, 0, 0)
#Pink = (102, 0, 204)
#Eraser = (64, 64, 64)



