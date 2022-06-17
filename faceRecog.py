import cv2
import numpy
import face_recognition
import os

path = "img"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for i in myList:
    #loads the images
    currImg = cv2.imread(f'{path}/{i}')
    images.append(currImg)
    #without .jpg
    classNames.append(os.path.splitext(i)[0])
print(images)
print(classNames)

def findEncoding(images):
    encodeList = []
    
    for img in images:
        # for better Results
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        # recognize the face
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeList = findEncoding(images)
print("Encoding Completed\n")

cap = cv2.VideoCapture(0)

while True:
    # cap.read() return 2 content
    success, img = cap.read()

    # size reduce will speed the process 1/4th
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)

    for encodeFace,faceLoc in zip(encodeCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeList,encodeFace)
        faceDis = face_recognition.face_distance(encodeList,encodeFace)
        #print(faceDis)
        matchIndex = numpy.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            # form rectange around the face
            cv2.rectangle(img,(x1,y1),(x2,y2),(564,255,0),2)
            # form rectangle in which we write name
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            # for name
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()