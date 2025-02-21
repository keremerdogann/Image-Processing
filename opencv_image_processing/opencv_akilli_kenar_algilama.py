import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

messi_img = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\roi.jpg")
k_size = (3,3)

cv.imshow("messi",messi_img)

messi_img_rgb = cv.cvtColor(messi_img,cv.COLOR_BGR2RGB)

cv.imshow("messi_rgb",messi_img_rgb)
#bu kod olmasa direkt messi_img yi kullanırsam renkler cv.imshow ile dogru gözüküyor ancak plt ile yanlış gözüküyor.

#görüntüye 3x3 lik bir gauss temizleme uyguluyorum k size degerine bagli

messi_img_to_gray = cv.cvtColor(messi_img_rgb,cv.COLOR_RGB2GRAY)

gaussian_blur_to_messi_image = cv.GaussianBlur(messi_img_to_gray,ksize=k_size,sigmaX=0,sigmaY=0)

plt.subplot(1,2,1),plt.imshow(gaussian_blur_to_messi_image,cmap="gray"),plt.title("GAUSS lu hali")
plt.subplot(1,2,2),plt.imshow(messi_img_rgb),plt.title("NORMAL HALİ")
plt.show()

#kendi fikrimce bileteral blur uygulamak daha doğru olacak çünkü bileteral blur onceki calismalarimda söyledigim gibi
#gauss filtrelemenin aksine görüntüyü yumuşatırken kenarları korur

messi_img_to_gray = cv.cvtColor(messi_img_rgb,cv.COLOR_RGB2GRAY)

bileteral_filter = cv.bilateralFilter(messi_img_to_gray,d=3,sigmaColor=50,sigmaSpace=50)

edges_wout_filter = cv.Canny(messi_img_to_gray,100,200)
edges_with_bilfilter = cv.Canny(bileteral_filter,100,200)
edges_with_gausfilter = cv.Canny(gaussian_blur_to_messi_image,100,200)

plt.subplot(2,3,1),plt.imshow(edges_wout_filter,cmap="gray"),plt.title("FİLTRESİZ KOSE ALGİLAMA")
plt.subplot(2,3,2),plt.imshow(edges_with_bilfilter,cmap="gray"),plt.title("BİLETERAL FİLTRELİ KOSE ALGİLAMA")
plt.subplot(2,3,3),plt.imshow(edges_with_gausfilter,cmap="gray"),plt.title("GAUSS FİLTRELİ KOSE ALGİLAMA")
plt.subplot(2,3,4),plt.imshow(messi_img_to_gray,cmap="gray"),plt.title("FİLTRESİZ GRİ FOTO")
plt.subplot(2,3,5),plt.imshow(bileteral_filter,cmap="gray"),plt.title("BİLETERAL FİLTRELİ FOTO")
plt.subplot(2,3,6),plt.imshow(gaussian_blur_to_messi_image,cmap="gray"),plt.title("GAUSS FİLTRELİ FOTO")
plt.show()
#cmap gray yazma sebebimiz plt fotoları her daim rgb göstermeye çalışır bu sebeple gri fotoları göstermek istedigimiz zaman
#fotolar gri renkte gözükmez , plt bunları rgb gibi göstermeye çalışır bundan dolayı cmap gray yazıyoruz ki gri kalsın







