import cv2 as cv
import numpy as np

#rengin hsv kodunu bulma , cunku ona göre islem yapıyoruz.
green = np.uint8([[[0,255,0]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
print(hsv_green)

cap = cv.VideoCapture(r"C:\Users\Kerem\Videos\GreenBlue Screen Transition Pack.mp4")

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

    green_res = cv.bitwise_and(frame,frame, mask= green_mask) # yeşil renkli videoyu göster
    blue_res = cv.bitwise_and(frame,frame,mask=blue_mask) # mavi renkli videoyu göster

    not_blue = cv.bitwise_not(blue_mask) #maviyi alma , kalan tüm renkleri al
    not_green = cv.bitwise_not(green_mask) #yesili alma , kalan tüm renkleri al

    all_except_blue = cv.bitwise_and(frame,frame,mask=not_blue)
    all_except_green = cv.bitwise_and(frame,frame,mask=not_green)

    #cv.imshow("hsv",hsv)
    #cv.imshow('manuel',frame)
    cv.imshow('only_green',green_res)
    #cv.imshow('blue_mask',blue_res)

    cv.imshow('all_except_blue',all_except_blue)
    cv.imshow('all_except_green',all_except_green)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()