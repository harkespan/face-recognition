import cv2
import numpy as np
import face_recognition

jkw1 = face_recognition.load_image_file('assets/images/datasets/harkespan5.jpg')
jkw1 = cv2.cvtColor(jkw1,cv2.COLOR_BGR2RGB)
jkw2 = face_recognition.load_image_file('assets/images/datasets/harkespan4.jpg')
jkw2 = cv2.cvtColor(jkw2,cv2.COLOR_BGR2RGB)

imgLoc = face_recognition.face_locations(jkw1)[0]
encodeJkw = face_recognition.face_encodings(jkw1)[0]
cv2.rectangle(jkw1,(imgLoc[3],imgLoc[0]),(imgLoc[1],imgLoc[2]),(255,0,255),2)

faceLocTrain = face_recognition.face_locations(jkw2)[0]
encodeJkwTrain = face_recognition.face_encodings(jkw2)[0]
cv2.rectangle(jkw2,(faceLocTrain[3],faceLocTrain[0]),(faceLocTrain[1],faceLocTrain[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeJkw],encodeJkwTrain)
faceDistance = face_recognition.face_distance([encodeJkw],encodeJkwTrain)
print(results,faceDistance)
cv2.putText(jkw2,f'{results} {round(faceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
cv2.imshow('Jokowi 1',jkw1)
cv2.imshow('Jokowi 2',jkw2)
cv2.waitKey(0)
