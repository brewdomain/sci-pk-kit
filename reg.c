import cv2 as cv

cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

img = cv.imread('example.jpg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

