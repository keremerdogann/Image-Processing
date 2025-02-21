import cv2
import cv2 as cv
import numpy as np


def rescaleFrame(frame,scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions , interpolation=cv.INTER_AREA)

img = cv.imread(r'C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\Resources\Photos\park.jpg')

cv.imshow("İlk deneme",img)

cv.waitKey(0)

capture = cv.VideoCapture(r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\Resources\Videos\dog.mp4")

resized_image = rescaleFrame(img)

cv.imshow("Resized deneme",resized_image)

cv.waitKey(0)

while True :
    isTrue , frame = capture.read()

    cv.imshow('Video',frame)

    if cv.waitKey(20) &  0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

"""""

blank = np.zeros((500,500,3),dtype='uint8')

cv.imshow('Blank',blank)

#blank[:] = 0,255,0 # 1-) rgb kodları 0 255 0 olarak eşleşir ama burada [:] yazan yere sayılar koyarak tüm resmin rengini degil belli bir kısmın rengini
                                                                                                            #ilgili renk yapabiliriz

# cv.imshow('Green',blank)  2-) daha sonrada bu yapılabilir

# YAZI YAZDIRMA
cv.putText(blank,'MERHABA',(255,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=1)
cv.imshow('Text',blank)

cv.waitKey(0)

# EKSENLERDE FOTO HAREKET ETTİRME
def translate(img,x,y):
    transMat = np.float32([[1,0,-x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

# -x left | -y up | x right | y down

translated = translate(resized_image,255,255)
cv.imshow("Kayik",translated)
cv.imshow("normali",resized_image)

cv.waitKey(0)

"""""

# ROTASYON


def rotate(img,angle,rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint     is None :
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)

    return cv.warpAffine(img,rotMat,dimensions)

rotated = rotate(resized_image,angle=72)
cv.imshow('Rotated',rotated)
cv.waitKey(0)

#RESİZİNG

resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Hazir resize',resized)
cv.waitKey(0)

#FLİPPİNG / aynalama

flip = cv.flip(resized,1) # 1-) yatay aynalama , sayıya göre eksenler degisir
cv.imshow("Flipped",flip)
cv.imshow("Düz hali",resized)
cv.waitKey(0)

#KIRPMA

print(img.size)

cropped = img[100:200,100:200]
cv.imshow('Cropped',cropped)
cv.waitKey(0)


cat_photo = r"C:\Users\Kerem\PycharmProjects\cuda_solution\opencv-course-master\Resources\Photos\cats.jpg"

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #GRİ YAPMA
cv.imshow('Gray',gray)

blur = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT) #BLUR KOYMA
cv.imshow('Blur',blur)

ret , thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('Thresh',thresh)

canny = cv.Canny(thresh,125,175) #görüntünün kenar kısmının ne kadar detaylı gözükeceğiyle oynarız
cv.imshow('Canny Edges',canny)


contours,hiararchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

cv.waitKey(0)




