import cv2


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
success, imgTest = cap.read()
cv2.imwrite('MichaelTest.png', imgTest)