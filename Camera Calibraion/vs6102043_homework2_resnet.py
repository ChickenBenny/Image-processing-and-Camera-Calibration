from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from resnet50 import Ui_MainWindow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras.utils.data_utils import get_file
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model
import cv2

filenames = os.listdir('D:/Python/lian_homework_2/cat_dog/train/')
label = []

for filename in filenames:
  target = filename.split('.')[0]
  if target == 'cat':
    label.append(0)
  else:
    label.append(1)                                                                                                                                           

all_data = pd.DataFrame({'filename': filenames, 'label': label})
all_data['label'] = all_data['label'].replace({0: 'cat', 1: 'dog'})
train, test = train_test_split(all_data, test_size = 0.2, random_state = 42)
val, test = train_test_split(test, test_size = 0.5, random_state = 42)

NUM_CLASS = 2
BATCH_SIZE = 8
EPOCH = 5
FREEZE_LAYERS = 2
image_width = 224
image_height = 224
image_size = (image_width, image_height)

'''
test_datagen = ImageDataGenerator(rescale=1./255,)

test_generator = test_datagen.flow_from_dataframe(
    test, 
    'D:/Python/lian_homework_2/cat_dog/train/', 
    x_col='filename',
    y_col='label',
    target_size=image_size,
    class_mode='categorical',
    batch_size=BATCH_SIZE
)
'''
model = load_model('D:/Python/lian_homework_2/none_aug_resent.h5')
model_aug = load_model('D:/Python/lian_homework_2/aug_resent.h5')

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
         super().__init__()
         self.setupUi(self)
         self.pushButton.clicked.connect(self.pushButton_clicked)
         self.pushButton_3.clicked.connect(self.pushButton_3_clicked)
         self.pushButton_4.clicked.connect(self.pushButton_4_clicked)
         self.pushButton_2.clicked.connect(self.pushButton_2_clicked)


    def pushButton_clicked(self):
        print('Summary of ResNet50')
        print(model.summary())

    def pushButton_2_clicked(self):
        tensorboard = cv2.imread('D:/Python/lian_homework_2/tensor.PNG')
        plt.axis('off')
        plt.title('Tensorboard_information')
        plt.imshow(tensorboard)
        plt.show()

    def pushButton_3_clicked(self):
        answer_list = ['Cat', 'Dog']
        x = self.textEdit.toPlainText()
        target = test.iloc[int(x), 0]
        target = f'D:/Python/lian_homework_2/cat_dog/train/{target}'
        img = cv2.imread(target)
        process = cv2.resize(img, (224, 224))
        process.astype(np.float32)
        process = process/ 255.
        process = np.reshape(process, (-1, 224, 224, 3))
        pred = model.predict(process)  
        print(pred)      
        if pred[0, 0] > pred[0, 1]:
            answer = answer_list[0]
        else :
            answer = answer_list[1]
        plt.title(answer)
        plt.imshow(img)
        plt.show()
        



    def pushButton_4_clicked(self):
        accuracy_score = [0.9671, 0.9162]
        tick_list = ['Before Random-Erasing', 'After Random-Erasing']
        plt.figure(figsize= (10, 6))
        plt.bar(np.arange(2), np.array(accuracy_score))
        plt.xticks(np.arange(2), np.array(tick_list))      
        plt.ylabel('Accuracy')
        plt.show()
    





























if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())