import cv2
from MyCanny import Canny
path = "picture.jpg"
savepath = "picture_result/"

Guassian_kernal_size = 3
HT_high_threshold = 45
HT_low_threshold = 15

if __name__ == '__main__':
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_RGB = cv2.imread(path)
    
    canny = Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)
    canny.canny_EdgeDetection()
    cv2.imwrite(savepath + "Mycanny.jpg", canny.img)
    