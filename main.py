import cv2
import numpy as np

class Canny:
    
    def __init__(self, Guassian_kernal_size, img, HT_high_threshold, HT_low_threshold):
        self.Guassian_kernal_size = Guassian_kernal_size
        self.img = img
        self.HT_high_threshold = HT_high_threshold
        self.HT_low_threshold = HT_low_threshold
        self.y, self.x = img.shape[0:2]    #读取图像的高度和宽度
        self.angle = np.zeros([self.y, self.x])  #记录每个像素点的梯度方向
        self.img_origin = None
        self.x_kneral = np.array([[-1, 1]]) #注意这里是二维数组
        self.y_kneral = np.array([[-1],[1]])
        
    def Get_gradient_img(self):
        
        print('输出梯度图像')
        
        # Mycode
        new_img_x = np.zeros([self.y, self.x], dtype = np.float)  #存储倒数值，用浮点数存储
        new_img_y = np.zeros([self.y, self.x], dtype = np.float)
        
        for i in range(self.x):
            for j in range(self.y):
                if j == 0:
                    new_img_y[j][i] = 1
                else:
                    new_img_y[j][i] = np.sum(np.array([[self.img[j - 1][i]], [self.img[j][i]]]) * self.y_kernal) #数组相乘，保持维度一致，均为二维数组
                if i == 0:
                    new_img_x[j][i] = 1
                else:
                    new_img_x[j][i] = np.sum(np.array([[self.img[j][i - 1], self.img[j][i]]]) * self.x_kernal) #数组相乘，保持维度一致，均为二维数组
        
        gradient_img, angle = cv2.cartToPolar(new_img_x, new_img_y) #返回的每个值都是二维数组
        self.angle = np.tan(angle) #用tan值表示
        self.img = gradient_img.astype(np.uint8)
        return self.img    