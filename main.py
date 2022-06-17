import cv2
import numpy as np
import face_recognition

imgMessi = face_recognition.load_image_file("Face Recognition\img\MESSI.jpg")
imgMessi = cv2.cvtColor(imgMessi,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Face Recognition\img\MessiTest.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMessi)[0]
encodeMessi = face_recognition.face_encodings(imgMessi)[0]
cv2.rectangle(imgMessi,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#print(faceLoc)
result = face_recognition.compare_faces([encodeMessi],encodeTest)
print(result)
cv2.imshow("Messi",imgMessi)
cv2.imshow("Test",imgTest)
cv2.waitKey(0)