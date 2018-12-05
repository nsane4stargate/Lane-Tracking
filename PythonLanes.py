
import numpy as np
import cv2
import matplotlib.pyplot as plt


def loadImage(str):
    image = cv2.imread(str)
    return image


def canny(image):
    grey = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    # Make an array of polygons
    polygons = np.array([[(200,height),(1100, height),(550,250)]])
    # Make a mask that has the same dimensions as the original image 
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    maskedImage = cv2.bitwise_and(image,mask)
    return maskedImage;

def displayLine(image, lines): 
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(lineImage,(x1,y1),(x2,y2), (255,0,0),10)
    return lineImage;

def getHoughLines(image):
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines;

def makeCoordinate(image,line_parameters):
    slope, intercept  = line_parameters
    y1 = image.shape[0]  
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def averageSlopeIntercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2),(y1, y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        # Left side will have -slope. Right side +slope
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = makeCoordinate(image,left_fit_average)
        right_line = makeCoordinate(image, right_fit_average)
        average_lines = [left_line,right_line]
        return average_lines


cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = getHoughLines(cropped_image)
    average_lines = averageSlopeIntercept(frame,lines)
    line_image = displayLine(frame,average_lines)
    lane_on_colorImage = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Lane', lane_on_colorImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows
