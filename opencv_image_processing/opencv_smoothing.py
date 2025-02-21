import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

logo = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\opencv_logo.png")
kernel = np.ones((5,5),np.float32)/25

dst = cv.filter2D(logo,-1,kernel)

plt.subplot(121),plt.imshow(logo),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtering')
plt.xticks([]), plt.yticks([])
plt.show()

kernel_size = (8,8)
blur = cv.blur(logo,ksize=kernel_size)

plt.subplot(121),plt.imshow(logo),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Normal Blurring')
plt.xticks([]), plt.yticks([])
plt.show()


gauss_kernel_size = (5,5) # bu degerleri rastgele veriyorum, ve tek sayı olmalı
gaussian_blur = cv.GaussianBlur(logo,ksize=gauss_kernel_size,sigmaX=0,sigmaY=0) #sigmaX veya sigmaY ile standart sapma degerleri verilebilir


plt.subplot(121),plt.imshow(logo),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gaussian_blur),plt.title('Gauss Blurring')
plt.xticks([]), plt.yticks([])
plt.show()


def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size
    salt_pixels = int(salt_prob * total_pixels)
    pepper_pixels = int(pepper_prob * total_pixels)

    # Salt (beyaz) gürültü ekle
    for _ in range(salt_pixels):
        x = np.random.randint(0, image.shape[1])  # Sütun
        y = np.random.randint(0, image.shape[0])  # Satır
        noisy_image[y, x] = [255, 255, 255]

    # Pepper (siyah) gürültü ekle
    for _ in range(pepper_pixels):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        noisy_image[y, x] = [0, 0, 0]

    return noisy_image


# Salt & Pepper Gürültüsü Ekle
salt_prob = 0.02  # %2 Salt
pepper_prob = 0.02  # %2 Pepper
noisy_image = add_salt_pepper_noise(logo, salt_prob, pepper_prob)

# Median Blur Uygula
median_blur = cv.medianBlur(noisy_image, ksize=5)  # Kernel boyutu tek sayı olmalı (ör. 3, 5, 7)
#median blur tuz biber görültülerini temizlemekte basariliymis.


#bir de gürültülü resime Gaussian blur uygulayalım

gaussian_blur_to_noisy_image = cv.GaussianBlur(noisy_image,ksize=gauss_kernel_size,sigmaX=0,sigmaY=0)

# Görüntüleri Göster ( buradan da goruldugu ve yukarıda da yazdıgım gibi median blur tuz,biber gürültülerine temizlemekte basarilidir.
plt.figure(figsize=(12, 6))
plt.subplot(1,4,1), plt.imshow(logo), plt.title("Original") #1.satır 4.sütün 1.hücre
plt.axis("off")
plt.subplot(1,4,2), plt.imshow(noisy_image), plt.title("Noisy (Salt & Pepper)") #1.satır 4.sütün 2.hücre
plt.axis("off")
plt.subplot(1,4,3), plt.imshow(median_blur), plt.title("Median Blurred") # 1.satır 4.sütün 3.hücre
plt.axis("off")
plt.subplot(1,4,4),plt.imshow(gaussian_blur_to_noisy_image),plt.title("Gaussian Blurred") # 1.satır 4.sütün 4.hücre
plt.axis("off")
plt.show()

#bilateral filtreleme , gauss filtremenin aksine görüntüyü yumuşatırken kenarları ve köşeleri korur.

ucgen_fotosu = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\Triangle_illustration.svg.png")

bilateral_blur = cv.bilateralFilter(logo,d=9,sigmaColor=75,sigmaSpace=75)

plt.subplot(121),plt.imshow(logo),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bilateral_blur),plt.title('Bilateral Blur')
plt.xticks([]), plt.yticks([])
plt.show()


"""""

src:
Giriş görüntüsü (gri tonlama veya renkli olabilir).

d:
Filtreleme çapı. Filtrenin kaç piksel etrafında etkili olacağını belirtir.

Örnek:
d = 5: 5x5 filtre çekirdeği.
Çok büyük bir değer verilirse otomatik olarak sigmaSpace tarafından kontrol edilir.

sigmaColor:

Yoğunluk (renk) farkının etkisini kontrol eder.
Büyük bir değer verilirse, daha geniş bir yoğunluk aralığı dikkate alınır ve daha fazla pikselle ortalama alınır.
Küçük bir değer verilirse, sadece benzer renkteki pikseller etkilenir.


sigmaSpace:

Mekansal mesafenin etkisini kontrol eder.
Büyük bir değer verilirse, daha uzak pikseller filtreleme işlemine dahil edilir.
Küçük bir değer verilirse, sadece yakın çevredeki pikseller dikkate alınır.

borderType (isteğe bağlı):

Kenarların nasıl işleneceğini belirtir (örn. cv.BORDER_DEFAULT, cv.BORDER_REFLECT).


E  T  K  İ  L  E  R 
-------------------------


sigmaColor'un etkisi:

Yüksek değer: Daha fazla renk grubu karıştırılır, kenar koruma azalır.
Düşük değer: Kenar korunur, ancak daha az yumuşatma sağlanır.

sigmaSpace'in etkisi:

Yüksek değer: Daha geniş bir alan etkilenir, uzak pikseller dikkate alınır.
Düşük değer: Yalnızca yakın pikseller etkilenir, kenar koruma artar.

d'nin etkisi:(kernel sayilir)

Büyük değer: Daha geniş bir çekirdek kullanılır.
Küçük değer: Daha dar bir çekirdek kullanılır.


"""""









