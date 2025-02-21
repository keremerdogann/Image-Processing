import cv2 as cv
import numpy as np

#rengin hsv kodunu bulma , cunku ona göre islem yapıyoruz.
green = np.uint8([[[0,255,0]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
print(hsv_green)

cap = cv.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue) #sadece mavi rengi algılar

    lower_green = np.array([50,50,50])
    upper_green = np.array([70,255,255])

    green_mask = cv.inRange(hsv,lower_green,upper_green) # sadece yeşil rengi algılar

    mixed_mask = cv.bitwise_or(blue_mask,green_mask) # hem mavi , hem yeşil

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mixed_mask)

    #cv.imshow("hsv",hsv)
    cv.imshow('frame',frame)
    cv.imshow('mask',mixed_mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()