import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileDetect=cv2.CascadeClassifier('haarcascade_smile.xml')
cam=cv2.VideoCapture(0)

facerec=cv2.face.LBPHFaceRecognizer_create()
#eigenfacerec = cv2.face.EigenFaceRecognizer_create()

agerec = cv2.face.LBPHFaceRecognizer_create()
natrec = cv2.face.LBPHFaceRecognizer_create()
genrec = cv2.face.LBPHFaceRecognizer_create()

facerec.read("facemodel/trainingData.yml")
#eigenfacerec.read("eigenfacemodel/trainingData.yml")

agerec.read("agemodel/trainingData.yml")
natrec.read("nationalitymodel/trainingData.yml")
genrec.read("gendermodel/trainingData.yml")

shift = 0
while(True):
    s=0
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)

    for (fx,fy,fw,fh) in faces:
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = img[fy:fy + fh, fx:fx + fw]

        smile = smileDetect.detectMultiScale(roi_gray,scaleFactor=1.7,minNeighbors=22,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 1)
            s = 1

        id, conf = genrec.predict(gray[fy:fy + fh, fx:fx + fw])

        if (id == 1):
            gendisplay = "Female"
        elif (id == 2):
            gendisplay = "Male"


        id, conf = natrec.predict(gray[fy:fy + fh, fx:fx + fw])

        if (id == 1):
            natdisplay = "Indian"
        elif (id == 2):
            natdisplay = "Chinese"
        elif (id == 3):
            natdisplay = "African"

        id, conf = agerec.predict(gray[fy:fy + fh, fx:fx + fw])

        if (id == 1):
            agedisplay = "BABY"
        elif (id == 2):
            agedisplay = "Teenager"
        elif (id == 3 or id == 4):
            agedisplay = "Adult"

        id,conf=facerec.predict(gray[fy:fy + fh, fx:fx + fw])
        if conf<66:
            if(id==1):
                display="Username_1"
            elif(id==2):
                display="Username_2"
            elif(id==3):
                display="Username_3"
            elif(id==4):
                display="Username_4"
            elif(id==5):
                display="Username_5"
            elif(id==6):
                display="Username_6"
        else:
            display="UNKNOWN"
        if s != 1:
            cv2.putText(img," SAY CHEESE", (fx, fy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 50), 2)

        cv2.putText(img, str(display), (fx, fy+fh+30), cv2.FONT_HERSHEY_COMPLEX, 1, (60, 180, 180), 2)
        cv2.putText(img, str(agedisplay), (fx, fy + fh + 60), cv2.FONT_HERSHEY_COMPLEX, 1, (60, 180, 180), 2)
        cv2.putText(img, str(natdisplay), (fx, fy + fh + 90), cv2.FONT_HERSHEY_COMPLEX, 1, (60, 180, 180), 2)
        cv2.putText(img, str(gendisplay), (fx, fy + fh + 120), cv2.FONT_HERSHEY_COMPLEX, 1, (60, 180, 180), 2)
        
    cv2.imshow("Your Mirror",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()


"""     id,conf=facerec.predict(gray[fy:fy+fh,fx:fx+fw])
        print(eigenfacerec.predict(gray[fy:fy + fh, fx:fx + fw]))
        print(facerec.predict(gray[fy:fy + fh, fx:fx + fw]))"""
