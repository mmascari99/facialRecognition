import cv2


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
success, imgTrain = cap.read()
cv2.imwrite('MichaelTrain.png', imgTrain)