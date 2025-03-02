import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import os, os.path
import traceback
import time

capture = cv2.VideoCapture(0) # Get image from Camera as Capture

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
# Hand detect as hd & hd2

offset = 30
step = 1
flag=False
suv=0
white=np.ones((300,300),np.uint8)*255 # white image board of 300 x 300

cv2.imwrite("D:\\Pycharm\\Testing\\white.jpg",white) # store at image @ folder- Pycharm / Testing

folder = "Data/Z"
counter = 0

while True:
    try:
        _, frame = capture.read() # read image at frame
        frame = cv2.flip(frame, 1)
        hands= hd.findHands(frame, draw=False, flipType=True)
        img_final=img_final1=img_final2=0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]


            roi = image     #rgb image without drawing

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # #for simple gray image without draw
            blur = cv2.GaussianBlur(gray, (1, 1), 2)


            gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # #for binary image
            blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            test_image1=blur
            img_final1 = np.ones((300, 300), np.uint8) * 148
            h = test_image1.shape[0]
            w = test_image1.shape[1]
            img_final1[((300 - h) // 2):((300 - h) // 2) + h, ((300 - w) // 2):((300 - w) // 2) + w] = test_image1

            img_final = np.ones((300, 300), np.uint8) * 255
            h = test_image.shape[0]
            w = test_image.shape[1]
            img_final[((300 - h) // 2):((300 - h) // 2) + h, ((300 - w) // 2):((300 - w) // 2) + w] = test_image


        hands = hd.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            white = cv2.imread("D:\\Pycharm\\Testing\\white.jpg")
            handz = hd2.findHands(image, draw=False, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']


                os = ((300 - w) // 2) - 15
                os1 = ((300 - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 255), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 255), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 255), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 255), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 255), 3)
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 255),
                         3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 255),
                         3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1),
                         (0, 255, 255), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 255),
                         3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 255),
                         3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 0), 1)

        cv2.imshow("skeleton", white)
        cv2.imshow("binary", img_final)
        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        key = cv2.waitKey(1)
        if key == ord("s"):  # S key is pressed
           counter += 1  # increment counter
           cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', white)
           # Saving image with current timestamp in specific folder
           print(counter)  # print how many times the img has been saved

        if interrupt & 0xFF == 27:    # esc key

            break
    except Exception:
        print("==",traceback.format_exc() )



capture.release()
cv2.destroyAllWindows()
