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
        
        #计算图像的梯度值和方向
        
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
        self.img = gradient_img.astype(np.uint8) #8进制编码
        return self.img   
    
    def Non_maximum_suppression(self):
         #对梯度图进行非极大抑制，并确定最终梯度方向
         
        print('Non_maximum_suppression')
        result = np.zeros([self.y, self.x])
         
        for i in range(1, self.y-1):
            for j in range(1, self.x-1):
                
                #小于4认为不是边缘？
                if abs(self.img[i][j]) <= 4:
                    result[i][j] = 0
                    continue        
                
                #找到变换最大的方向，注意坐标原点在左上角
                #注意此处用的是|tan|，故判断完角度后还需判断其正负值，才可确定其象限
                
                elif abs(self.angle[i][j])>1:
                    gradient2 = self.img[i - 1][j]
                    gradient4 = self.img[i + 1][j]
                    
                    #第一三象限正方向
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                    
                    #第二四象限方向    
                    else:
                        gradient1 = self.img[i - 1][j + 1]
                        gradient3 = self.img[i + 1][j - 1]
                    
                else:
                    #倾斜角度小于45°时
                    gradient2 = self.img[i][j - 1] 
                    gradient4 = self.img[i][j + 1]
                    
                    if self.angle[i][j] > 0:
                        gradient1 = self.img[i - 1][j - 1]
                        gradient3 = self.img[i + 1][j + 1]
                        
                    else:
                        gradient1 = self.img[i + 1][j - 1]
                        gradient3 = self.img[i - 1][j + 1]
                
                #此处注意角度判断，g1 g2 g3 g4的设置也方便了代码的统一化        
                temp1 = abs(self.angle[i][j]) * gradient1 + (1 - abs(self.angle[i][j])) * gradient2
                temp2 = abs(self.angle[i][j]) * gradient3 + (1 - abs(self.angle[i][j])) * gradient4
                
                if  self.img[i][j] >= temp1 and self.img[i][j]>= temp2:
                    result[i][j] = self.img[i][j]
                else:
                    result[i][j] = 0
                
                self.img = result
                
                return self.img  
        
        
                