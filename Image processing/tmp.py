import numpy as np
import cv2
import glob

images = glob.glob('D:\lian_homework\Dataset_OpenCvDl_Hw2\Q2_Image\*')

height, width = cv2.imread(r'D:\lian_homework\Dataset_OpenCvDl_Hw2\Q2_Image\1.bmp').shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', fourcc, 2, (height, width))

for fname in images:
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    if ret:
        fnl = cv2.drawChessboardCorners(img, (11, 8), corners, ret)
        video.write(fnl)
video.release()
capture = cv2.VideoCapture(r'D:\lian_homework\Dataset_OpenCvDl_Hw2\Q2_Image\output.mp4')
while(capture.isOpened()):
    ret_cap, frame = capture.read()
    if not ret_cap:
        break
    cv2.namedWindow('Corner', 0)
    cv2.resizeWindow('Corner', 640, 480)
    cv2.imshow('Corner', frame)
    cv2.waitKey(500)
cv2.destroyAllWindows()
capture.release()
