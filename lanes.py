import cv2
import numpy as np
#import matplotlib.pyplot as plt

# imread() and imshow() = to load and display the image 
image = cv2.imread(r'C:\Users\Admin\Desktop\Python\Lane Detection\road.jpg',1)
lane_img = np.copy(image)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines): #for optimization
    left_fit = []
    right_fit= []
    for line in lines:
        x1,y1,x2,y2 =line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope <0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
#Canny Edge Detector
#Step 1- Convert to greyscale - to convert the picture from 3 channels to one channel- more fast processing (AlREADY dONE)
#Step 2- Apply Gaussian blur on the image and Reduce Image noice
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_img, (x1,y1),(x2,y2), (0,255,0),10)
    return line_img

def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([[(200,height), (1100,height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img
canny_img = canny(lane_img)
cropped_img = region_of_interest(canny_img)
#Hough Transform - we need a region of interest in the image that defines our lanes
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100 , np.array([]), minLineLength=40,maxLineGap=5)
avg_lines = average_slope_intercept(lane_img, lines)
line_img = display_lines(lane_img, avg_lines)
combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1,1) #line img will have 20%more weight
#imshow - to render the image
cv2.imshow('result', combo_img)
cv2.waitKey(0) 
#plt.imshow(canny)
#plt.show()




