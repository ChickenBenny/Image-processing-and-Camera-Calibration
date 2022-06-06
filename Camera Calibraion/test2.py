# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test2.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
import glob


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 50, 211, 391))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 40, 181, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 100, 181, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(250, 340, 120, 80))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(280, 50, 211, 391))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei UI")
        font.setPointSize(12)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 40, 181, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 100, 181, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_5.setGeometry(QtCore.QRect(10, 260, 181, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 320, 181, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 150, 191, 101))
        self.groupBox_4.setObjectName("groupBox_4")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit.setGeometry(QtCore.QRect(110, 30, 71, 21))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 101, 21))
        self.label_2.setObjectName("label_2")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_7.setGeometry(QtCore.QRect(20, 60, 161, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 560, 58, 15))
        self.label.setObjectName("label")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(510, 50, 231, 191))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(12)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.textEdit_2 = QtWidgets.QTextEdit(self.groupBox_5)
        self.textEdit_2.setGeometry(QtCore.QRect(20, 30, 161, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 80, 211, 41))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_9.setGeometry(QtCore.QRect(10, 130, 211, 41))
        self.pushButton_9.setObjectName("pushButton_9")
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(510, 260, 231, 181))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setPointSize(12)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_12.setGeometry(QtCore.QRect(10, 40, 211, 41))
        self.pushButton_12.setObjectName("pushButton_12")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Find Contour"))
        self.pushButton.setText(_translate("MainWindow", "1.1 Draw Contour"))
        self.pushButton_2.setText(_translate("MainWindow", "1.2 Count Rings"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2. Calibration"))
        self.pushButton_3.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.pushButton_4.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.pushButton_5.setText(_translate("MainWindow", "2.4 Find Distortion"))
        self.pushButton_6.setText(_translate("MainWindow", "2.5 Show result"))
        self.groupBox_4.setTitle(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.label_2.setText(_translate("MainWindow", "Select image"))
        self.pushButton_7.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.groupBox_5.setTitle(_translate("MainWindow", "3. Augmented Reality"))
        self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Microsoft JhengHei\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">OPENCV</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.pushButton_8.setText(_translate("MainWindow", "3.1 Show Words on Board"))
        self.pushButton_9.setText(_translate("MainWindow", "3.2 Show Words Vertically"))
        self.groupBox_6.setTitle(_translate("MainWindow", "4. Stereo Disparity Map"))
        self.pushButton_12.setText(_translate("MainWindow", "4.1 Stereo Disparity Map"))









if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())