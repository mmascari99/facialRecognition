import cv2
import numpy as np
import face_recognition as fr

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#use these 4 lines to capture a training img and a img to recognize
#success, imgTrain = cap.read()
#cv2.imwrite('MichaelTrain.png', imgTrain)
#success, imgTest = cap.read()
#cv2.imwrite('MichaelTest.png', imgTest)
imgTest = cv2.imread('MichaelTest.png')
imgTrain = fr.load_image_file('MichaelTrain.png', 'RGB')
imgTrain = cv2.cvtColor(imgTrain, cv2.COLOR_BGR2RGB)

faceLocTrain = fr.face_locations(imgTrain)[0]
encodeFace = fr.face_encodings(imgTrain)[0]
cv2.rectangle(imgTrain,(faceLocTrain[3], faceLocTrain[0]), (faceLocTrain[1], faceLocTrain[2]), (255, 0, 255), 2)
#print(faceLoc)

faceLocTest = fr.face_locations(imgTest)[0]
encodeTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = fr.compare_faces([encodeFace], encodeTest)
print(results)

while True:
    #success, img = cap.read()
    #cv2.imshow("Video", img)
    cv2.imshow("train", imgTrain)
    cv2.imshow("test", imgTest)
    cv2.waitKey(1)