import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

j = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\j.png")

kernel_3 = np.ones((3,3),np.uint8)
erosion_j_3 = cv.erode(j, kernel_3, iterations=1)

kernel_5 = np.ones((5,5),np.uint8)
erosion_j_5 = cv.erode(j,kernel_5,iterations=1)

kernel_7 = np.ones((7,7),np.uint8)
erosion_j_7 = cv.erode(j,kernel_7,iterations=1)

#chatgpt kernel boyutunun islem üzerinden büyük bir etkiye sahip oldugundan bahsediyor
#bu sebeple farklı kernel boyutlarını tek bir grafik üzerinde gösterip farklarını gözlemleyeceğim

#EROZYON İLGİLİ FOTOĞRAFTAKİ BEYAZ PİKSELLERİN AZALIP , SİYAHLARIN ARTMASINA SEBEP OLUR ,
# BU SEBEPLE AŞAĞIDA , BEYAZ J HARFİNİ DAHA İNCELMİS GORMEKTEYİZ


plt.subplot(1,4,1),plt.imshow(j),plt.title("EN SADE HALİ")
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2),plt.imshow(erosion_j_3),plt.title("kernel 3")
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3),plt.imshow(erosion_j_5),plt.title("kernel 5")
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4),plt.imshow(erosion_j_7),plt.title("kernel 7")
plt.xticks([]), plt.yticks([])
plt.show()

#kernel boyutu arttıkca daha da siliklestigini görüyoruz
#simdi kernel boyutunu sabit tutarak , iterasyon sayısını arttıralım

const_kernel = np.ones((3,3),np.uint8)
erosion_j_i1 = cv.erode(j,kernel=const_kernel,iterations=1)
erosion_j_i3 = cv.erode(j, kernel=const_kernel, iterations=3)
erosion_j_i5 = cv.erode(j, kernel=const_kernel, iterations=5)
erosion_j_i9 = cv.erode(j,kernel=const_kernel,iterations=9)

plt.subplot(1,5,1),plt.imshow(j),plt.title("EN SADE HALİ")
plt.xticks([]), plt.yticks([])
plt.subplot(1,5,2),plt.imshow(erosion_j_i1),plt.title("iter1")
plt.xticks([]), plt.yticks([])
plt.subplot(1,5,3),plt.imshow(erosion_j_i3),plt.title("iter3")
plt.xticks([]), plt.yticks([])
plt.subplot(1,5,4),plt.imshow(erosion_j_i5),plt.title("iter5")
plt.xticks([]), plt.yticks([])
plt.subplot(1,5,5),plt.imshow(erosion_j_i9),plt.title("iter9")
plt.show()

#burda da gozuktugu üzere , iterasyon sayısı arttırıldığı gibi kernelden de belirgin sekilde siliklik artıyor.



#ŞİMDİ DE EROZYONUN TERSİ DİLASYONU YAPİCAZ

dilation = cv.dilate(j,kernel=const_kernel,iterations=5)

plt.subplot(1,2,1),plt.imshow(j),plt.title("Orijinal")
plt.subplot(1,2,2),plt.imshow(dilation),plt.title("Dilation")
plt.show()

#dilasyonda da aynı sekilde kernel boyutu ve iterasyon arttıkca bu sefer doldurma islemi yaptıgımız icin doldurma artar.

# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
 # opening , closing , diye tür var onları buradan bakarsın ben kendi istediklerime devam ediyorum.

#morfolojik gradyan , cismin kose kenarlarini alır

gradyan = cv.morphologyEx(j,cv.MORPH_GRADIENT,kernel=kernel_3)

plt.subplot(1,2,1),plt.imshow(j),plt.title("Orijinal")
plt.subplot(1,2,2),plt.imshow(gradyan),plt.title("Gradient")
plt.show()

#kernel boyutu buyudukce , kenarlar daha kalın hala geliyor , bunu gözlemledim

# yarın da akıllı kenar algılama calısacagım

