from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from test import Ui_MainWindow
import numpy as np
import cv2

dog_strong = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg')
dog_weak = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg')
picture = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
lenna_whitenoise = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')
lenna_peppersalt = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg')
house = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
square = cv2.imread('D:/lian_homework/Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
         super().__init__()
         self.setupUi(self)
         self.pushButton.clicked.connect(self.pushButton_clicked)
         self.pushButton_2.clicked.connect(self.pushButton_2_clicked)
         self.pushButton_3.clicked.connect(self.pushButton_3_clicked)
         self.pushButton_4.clicked.connect(self.pushButton_4_clicked)
         self.pushButton_5.clicked.connect(self.pushButton_5_clicked)
         self.pushButton_6.clicked.connect(self.pushButton_6_clicked)
         self.pushButton_7.clicked.connect(self.pushButton_7_clicked)
         self.pushButton_11.clicked.connect(self.pushButton_11_clicked)
         self.pushButton_12.clicked.connect(self.pushButton_12_clicked)
         self.pushButton_13.clicked.connect(self.pushButton_13_clicked)
         self.pushButton_14.clicked.connect(self.pushButton_14_clicked)
         self.pushButton_15.clicked.connect(self.pushButton_15_clicked)
         self.pushButton_16.clicked.connect(self.pushButton_16_clicked)
         self.pushButton_17.clicked.connect(self.pushButton_17_clicked)
         self.pushButton_18.clicked.connect(self.pushButton_18_clicked)



    def on_Trackbar(self,x):
        w = cv2.getTrackbarPos('Blend', 'Blend')
        min_ = w/ 255
        max_ = (1 - min_)
        dst = cv2.addWeighted(dog_strong, max_, dog_weak, min_, 0)
        cv2.imshow('Blend', dst)    

    def convolution2d(self, image, kernel, bias):
        m, n = kernel.shape[0], kernel.shape[1]
        x, y = image.shape[0], image.shape[1]
        x = x- m+ 1
        y = y- m+ 1
        new_image = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                new_image[i][j] = np.sum(image[i: i+ m, j:j+ m]* kernel)+ bias
        return new_image

    def translate(self, image, x, y):
        m = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, m, (400,300), (image.shape[1], image.shape[0]), )
        return shifted



    def pushButton_clicked(self):
        cv2.imshow("Hw1-1", picture)
        print(f"Height :  {picture.shape[0]}")
        print(f'Width :  {picture.shape[1]}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_2_clicked(self):
        b, g, r = cv2.split(picture)
        zeros = np.zeros(picture.shape[:2], dtype = np.uint8)
        b = cv2.merge([b, zeros, zeros])
        g = cv2.merge([zeros, g, zeros])
        r = cv2.merge([zeros, zeros, r])
        cv2.imshow("Blue", b)
        cv2.imshow("Green", g)
        cv2.imshow("Red", r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_3_clicked(self):
        picture_cvfunction = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        picture_average = np.mean(picture, axis = 2)
        cv2.imshow('OpenCV function', picture_cvfunction)
        cv2.imshow('Average show', picture_average/255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_4_clicked(self):
        cv2.namedWindow('Blend')
        cv2.createTrackbar('Blend', 'Blend', 0, 255, self.on_Trackbar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_5_clicked(self):
        blur_lenna = cv2.GaussianBlur(lenna_whitenoise, (3, 3), 0)
        cv2.imshow('Gaussian_Blur', blur_lenna)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_6_clicked(self):
        bilateral_blur= cv2.bilateralFilter(lenna_whitenoise,9, 90, 90 )
        cv2.imshow('Bilateral_Blur', bilateral_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_7_clicked(self):
        median_blur_3 = cv2.medianBlur(lenna_peppersalt, 3)
        median_blur_5 = cv2.medianBlur(lenna_peppersalt, 5)
        cv2.imshow('median Filter 3*3', median_blur_3)
        cv2.imshow('median Filter 5*5', median_blur_5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_11_clicked(self):
        gray_house = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)
        gaussian_kernel = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]])
        x, y = gaussian_kernel[:, :, 0], gaussian_kernel[:, :, 1]
        gaussian_kernel = np.exp(-(x** 2+ y** 2))
        gaussian_kernel = gaussian_kernel/ gaussian_kernel.sum()
        gaussian_blur_image = self.convolution2d(gray_house, gaussian_kernel, 0)
        cv2.imshow('Grayscale', gray_house)
        cv2.imshow('Gaussian Blur', gaussian_blur_image/ 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def pushButton_12_clicked(self):
        gray_house = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)
        gaussian_kernel = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]])
        x, y = gaussian_kernel[:, :, 0], gaussian_kernel[:, :, 1]
        gaussian_kernel = np.exp(-(x** 2+ y** 2))
        gaussian_kernel = gaussian_kernel/ gaussian_kernel.sum()
        gaussian_blur_image = self.convolution2d(gray_house, gaussian_kernel, 0)

        sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_image = self.convolution2d(gaussian_blur_image, sobel_kernel, 0)

        cv2.imshow('Sobel X', sobel_image/ 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_13_clicked(self):
        gray_house = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)
        gaussian_kernel = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]])
        x, y = gaussian_kernel[:, :, 0], gaussian_kernel[:, :, 1]
        gaussian_kernel = np.exp(-(x** 2+ y** 2))
        gaussian_kernel = gaussian_kernel/ gaussian_kernel.sum()
        gaussian_blur_image = self.convolution2d(gray_house, gaussian_kernel, 0)
        sobel_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_image = self.convolution2d(gaussian_blur_image, sobel_kernel, 0)
        cv2.imshow('Sobel Y', sobel_image/ 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_14_clicked(self):
        gray_house = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)
        gaussian_kernel = np.array([[(-1, -1), (0, -1), (1, -1)], [(-1, 0), (0, 0), (1, 0)], [(-1, 1), (0, 1), (1, 1)]])
        x, y = gaussian_kernel[:, :, 0], gaussian_kernel[:, :, 1]
        gaussian_kernel = np.exp(-(x** 2+ y** 2))
        gaussian_kernel = gaussian_kernel/ gaussian_kernel.sum()
        gaussian_blur_image = self.convolution2d(gray_house, gaussian_kernel, 0)

        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_image_x = self.convolution2d(gaussian_blur_image, sobel_kernel_x, 0)
        sobel_image_x = cv2.convertScaleAbs(sobel_image_x)

        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_image_y = self.convolution2d(gaussian_blur_image, sobel_kernel_y, 0)
        sobel_image_y = cv2.convertScaleAbs(sobel_image_y)

        magnitude = cv2.addWeighted(sobel_image_x, 0.5, sobel_image_y, 0.5, 0)
        cv2.imshow('Magnitude', magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_15_clicked(self):
        image = cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_16_clicked(self):
        image = cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
        translation_image = self.translate(image, 0 , 60)
        cv2.imshow("img_1", translation_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_17_clicked(self):
        image = cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
        translation_image = self.translate(image, 0 , 60)
        coordinate = (128, 188)
        rotate_matrix = cv2.getRotationMatrix2D(coordinate, 10, 0.5)
        rotate_image = cv2.warpAffine(translation_image, rotate_matrix, (400, 300))
        cv2.imshow("img_2", rotate_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def pushButton_18_clicked(self):
        image = cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
        translation_image = self.translate(image, 0 , 60)
        coordinate = (128, 188)
        rotate_matrix = cv2.getRotationMatrix2D(coordinate, 10, 0.5)
        rotate_image = cv2.warpAffine(translation_image, rotate_matrix, (400, 300))
        old_location = np.float32([[50, 50], [200, 50], [50, 200]])
        new_location = np.float32([[10, 100], [200, 50], [100, 250]])
        shear_matrix = cv2.getAffineTransform(old_location, new_location)
        shear_image = cv2.warpAffine(rotate_image, shear_matrix, (400, 300))
        cv2.imshow("img_3", shear_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
            







if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())