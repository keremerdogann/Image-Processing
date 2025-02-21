import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

messi_img = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\roi.jpg",cv.IMREAD_GRAYSCALE)

messi_img_copy = messi_img.copy()

template = cv.imread(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\messi_face.jpg",cv.IMREAD_GRAYSCALE)

h = template.shape[0] #yukseklık
w = template.shape[1] #genıslık


methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
            'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']

for meth in methods:
    messi_img = messi_img_copy.copy()
    method = getattr(cv,meth)

    res = cv.matchTemplate(messi_img,template,method) #şablonun ne kadar eşleştiğini gösteren  bir matris verir.
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res) #bu matrisin en düşük ve en yüksek değerlerini ve koordinatlarını bulur.

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc #en düşük değerin olduğu koordinat (sol ust kose)
    else:
        top_left = max_loc #en yüksek değerin olduğunu koordinat (sol üst kose)

    bottom_right = (top_left[0]+w,top_left[1]+h) #[0] x ekseni , [1] y ekseni

    cv.rectangle(messi_img,top_left,bottom_right,255,2)

    plt.subplot(1,2,1),plt.imshow(res,cmap="gray"),plt.title("Matching Result")
    plt.subplot(1,2,2),plt.imshow(messi_img,cmap="gray"),plt.title("Detected Point")
    plt.suptitle(meth)

    plt.show()


