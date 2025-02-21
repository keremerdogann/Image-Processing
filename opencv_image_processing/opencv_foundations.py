# BURAYI OPENCV DOKUMANTASYONUYLA ÇALIŞACAĞIM
import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\roi.jpg"

img = cv2.imread(img)
assert img is not None , "file could not be read , chech with os.path.exists"

img2 = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\Historical_peninsula_and_modern_skyline_of_Istanbul.jpg")


"""""
px = img[10,15]
print(px)

blue = img[100,100,0]
print(f"blue: {blue}")

print(f"shape : {img.shape}")

ball = img[220:280, 330:390]
#img[273:333, 100:160] = ball

cv.imshow("sonuc",img)
cv.waitKey(0)

#RENK RENK AYIRMA

# b,g,r = cv2.split(img) # Maliyetli bir islem oldugu soylenıyor

cv.imshow("splitting",img)

# TEKRARDAN BİRLESTİRME

# img = cv.merge((b,g,r))

cv.imshow("merging",img)

cv.waitKey(0)

# Belirli sınır islemleri yapacagiz , padding denmıs buna , agırlıklı olarak köşe falan dolduruyoruz

blue = [255,0,0]

b , g , r = cv.split(img2)

r = cv.add(r,100)

img2 = cv.merge((b,g,r))

img3 = cv.merge

assert img2 is not None , "file couldnt be read."

replicate = cv.copyMakeBorder(img2,50,50,50,50,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img2,50,50,50,50,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img2,50,50,50,50,cv.BORDER_REFLECT101)
wrap = cv.copyMakeBorder(img2,50,50,50,50,cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img2,50,50,50,50,cv.BORDER_CONSTANT)

plt.subplot(231), plt.imshow(img2, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant), plt.title('CONSTANT')

plt.show()

"""""

# img2 istanbul , img1 messi


# simdi iki resmi üst üste koyup , agırlık fonksiyonunu kullanarak hangisinin birbiri üzerinde daha saydam gözüküp gözükmediğini hesaplayacağız.
# bunu l.img1 + b.img2 + l olarak düşünebiliriz.


#dsize kaç x kaç piksel yaptıracak onu ayarlar
img = cv.resize(img,(1000,1000),interpolation=cv.INTER_LINEAR)
img2 = cv.resize(img2,(1000,1000),interpolation=cv.INTER_LINEAR)

dst = cv.addWeighted(img,0.95,img2,0.05,0)

cv.imshow("agırlıklandırılmıs resim",dst)
cv.waitKey(0)
cv.destroyAllWindows()

#img1 ve messi aslında aynı ama ben messi demek istiyorum

messi = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\roi.jpg") # Messi fotoğrafını okur ve 'messi' değişkenine atar.
opencv_logo = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\opencv_logo.png") # OpenCV logosunu okur ve 'opencv_logo' değişkenine atar.

rows,cols,channels = opencv_logo.shape # OpenCV logosunun boyutlarını (satır, sütun, kanal) alır.
print(f"opencvlogosunun satır sütün ve renk kanalı değerleri : {opencv_logo.shape} size degeri : {opencv_logo.size}")
roi = messi[0:rows,0:cols] # Messi fotoğrafının sol üst köşesinden logo boyutlarında bir bölgeyi (Region of Interest - ROI) seçer.

opencv_logo_to_gray = cv.cvtColor(opencv_logo,cv.COLOR_BGR2GRAY) # OpenCV logosunu gri tonlamaya dönüştürür.

ret , mask = cv.threshold(opencv_logo_to_gray,10,255,cv.THRESH_BINARY)
# Gri tonlamalı logoya eşikleme uygular. 10 eşik değerinin altındaki pikseller 0 (siyah), üstündekiler 255 (beyaz) olur. Bu, bir maske oluşturur.
# ret: Eşik değerini döndürür (bu örnekte 10).
# mask: Eşikleme sonucunda oluşan maske.

mask_inv = cv.bitwise_not(mask)
# Maskeyi tersine çevirir. Siyah pikseller beyaz, beyaz pikseller siyah olur.

messi_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# ROI bölgesindeki Messi fotoğrafının, ters maskeye göre arka planını (logo olmayan kısımlarını) alır.
# Yani, maskenin beyaz olduğu yerlerdeki ROI pikselleri korunur, siyah olduğu yerlerdeki pikseller siyah olur.

opencv_logo_fg = cv.bitwise_and(opencv_logo,opencv_logo,mask=mask)
# Orijinal logonun, maskeye göre ön planını (logo olan kısımlarını) alır.
# Yani, maskenin beyaz olduğu yerlerdeki logo pikselleri korunur, siyah olduğu yerlerdeki pikseller siyah olur.

dst = cv.add(messi_bg,opencv_logo_fg)
# Arka plan (messi_bg) ve ön planı (opencv_logo_fg) toplar. Bu, logoyu Messi fotoğrafının üzerine yerleştirir.

roi = dst
# Oluşturulan yeni görüntüyü (dst) Messi fotoğrafının ROI bölgesine yerleştirir.

cv.imshow("res",messi_bg) # Sonucu gösterir.

cv.waitKey(0) # Pencere kapanana kadar bekler.

cv.destroyAllWindows() # Tüm pencereleri kapatır.


# ŞU BİTWİSE LARA TEKRARDAN ÇALIŞMAM LAZIM NEYİN NE İŞE YARADIĞINI PEK ANLAYAMADIM AÇIKÇASI





