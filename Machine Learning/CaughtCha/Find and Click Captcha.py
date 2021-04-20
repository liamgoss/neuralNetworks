from python_imagesearch.imagesearch import imagesearch
from scipy import interpolate
from PIL import ImageGrab
import numpy as np
import math
import random
import pyautogui
import time
import traceback
import sys
import pytesseract
import cv2

def findCaptcha(dark, light):
    posDark = imagesearch(dark)
    posLight = imagesearch(light)
    if posDark[0] != -1 and posDark[1] != -1:
        print("Position (Dark): ", posDark[0], posDark[1])
        return posDark[0], posDark[1], True
    elif posLight[0] != -1 and posLight[1] != -1:
        print("Position (Light): ", posLight[0], posLight[1])
        return posLight[0], posLight[1], True
    else:
        print("No Captcha's Found")

def clickToPoint(xIn, yIn):
    def pointDistance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Uncomment this section to speed up the mouse movement (it is uber fast lmao)
    # '''
    # Any duration less than this is rounded to 0.0 to instantly move the mouse.
    pyautogui.MINIMUM_DURATION = 0  # Default: 0.1
    # Minimal number of seconds to sleep between mouse moves.
    pyautogui.MINIMUM_SLEEP = 0  # Default: 0.05
    # The number of seconds to pause after EVERY public function call.
    pyautogui.PAUSE = 0  # Default: 0.1
    # '''

    controlPoints = random.randint(3, 8)  # number of control points (must be 2 or more)
    x1, y1 = pyautogui.position()  # starting position of mouse



    # Distribute our control points evenly between start and end points
    x = np.linspace(x1, xIn, num=controlPoints, dtype='int')
    y = np.linspace(y1, yIn, num=controlPoints, dtype='int')

    rnd = 10
    xr = [random.randint(-rnd, rnd) for k in range(controlPoints)]
    yr = [random.randint(-rnd, rnd) for k in range(controlPoints)]
    xr[0] = yr[0] = xr[-1] = yr[-1] = 0
    x += xr
    y += yr

    # Bezier spline approximation
    degree = 3 if controlPoints > 3 else controlPoints - 1  # A degree of 3 is recommended, must be less than control pts
    tck, u = interpolate.splprep([x, y], k=degree)

    # Move upto a certain number of points
    u = np.linspace(0, 1, num=2 + int(pointDistance(x1, y1, xIn, yIn) / 50))
    points = interpolate.splev(u, tck)

    # Move mouse
    duration = 0.1
    timeout = duration / len(points[0])
    point_list = zip(*(i.astype(int) for i in points))
    for point in point_list:
        pyautogui.moveTo(*point)
        time.sleep(timeout)
    pyautogui.click()

def findInitialCaptcha():
    try:
        x, y, success = findCaptcha("captchaDark.png", "captchaLight.png")
        print(x, ", ", y)
        #x = x + 25  # If one monitor
        x = x - 1250  # If three monitors
        y = y + 30
        clickToPoint(x, y)
    except TypeError:
        pass
def findAccessibilityOptions():
    try:
        time.sleep(2)
        x, y, success = findCaptcha("iconsDark.png", "iconsLight.png")
        print(x, ", ", y)
        #x = x + 200  # If one monitor
        x = x - 935  # If three monitors
        print(x, ", ", y)
        print("Distance: ")
        y = y + 25
        clickToPoint(x, y)
    except TypeError:
        pass

def textRecognition():
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    i = 1
    while i < 2:
        # ImageGrab ~ To capture the screen image in a loop
        # Bbox ~ To capture a specific area
        # (left_x, top_y, right_x, bottom_y)
        time.sleep(2)
        cap = ImageGrab.grab(bbox=None)
        cap.show()
        #Converted the image to monochrome for easy OCR

        tesstr = pytesseract.image_to_string(cv2.cvtColor(np.array(cap), cv2.COLOR_BGR2GRAY), lang="eng")
        print(tesstr)
        i = i + 1
textRecognition()
#findAccessibilityOptions()
#findInitialCaptcha()