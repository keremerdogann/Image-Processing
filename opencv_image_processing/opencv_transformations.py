"""""

Dönüşümler
OpenCV , her türlü dönüşümü gerçekleştirebileceğiniz cv.warpAffine ve cv.warpPerspective adında iki dönüşüm fonksiyonu sunar .
cv.warpAffine girdi olarak 2x3'lük bir dönüşüm matrisi alırken, cv.warpPerspective girdi olarak 3x3'lük bir dönüşüm matrisi alır.

"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

"""""
Ölçekleme

Ölçekleme, yalnızca görüntünün yeniden boyutlandırılmasıdır. 
OpenCV, bu amaç için cv.resize() fonksiyonuyla birlikte gelir . 
Görüntünün boyutu elle belirtilebilir veya ölçekleme faktörünü belirtebilirsiniz. Farklı enterpolasyon yöntemleri kullanılır.
Tercih edilen enterpolasyon yöntemleri, küçültme için cv.INTER_AREA ve yakınlaştırma için cv.INTER_CUBIC (yavaş) ve cv.INTER_LINEAR'dır . 
Varsayılan olarak, tüm yeniden boyutlandırma amaçları için cv.INTER_LINEAR enterpolasyon yöntemi kullanılır. 
Bir giriş görüntüsünü aşağıdaki yöntemlerden biriyle yeniden boyutlandırabilirsiniz:

"""

img = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\Historical_peninsula_and_modern_skyline_of_Istanbul.jpg")

res = cv.resize(img,None,fx=0.5,fy=0.5,interpolation = cv.INTER_CUBIC)

#OR

""""
height,width = img.shape[:2]
res = cv.resize(img,(2*width,2*height),interpolation = cv.INTER_CUBIC)

"""

cv.imshow("ilk hali",img)
cv.imshow("son hali",res)


cv.waitKey(0)


#dokumantasyonun bu kısmında belirli islemler var ama sadece bu kısmını gerekli gordum
  # https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html

#C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\sudoku.png

sudoku = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\sudoku.png")
rows,cols,channel_number = sudoku.shape

pts1 = np.float32([[56,56],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(sudoku,M,(300,300))

plt.subplot(121),plt.imshow(sudoku),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

# RESİM OLCEKLERİ DOGRU OLMADIGI ICIN SONUC DOGRU GİBİ GOZUKMUYOR ANCAK SONUC DOGRU


