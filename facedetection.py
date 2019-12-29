import cv2
import numpy
eye_cascade=cv2.CascadeClassifier('C:/Users/ankush/Documents/GitHub/opencv-tutorial\haar-cascade-files-master/haarcascade_eye.xml')
face_casecade=cv2.CascadeClassifier('C:/Users/ankush/Documents/GitHub/opencv-tutorial\haar-cascade-files-master/haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0)
scale_factor=1.3

while 1:
    ret,pic =video.read()
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    face=face_casecade.detectMultiScale(pic,scale_factor,5)
    # for (x,y,w,h) in face:
    #     front=cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    for (x,y,w,h) in face:

        cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        front=cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(pic,(x,y),front,2,(255,255,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = pic[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    print("number of found{} ".format(len(face)))    
    cv2.imshow('face',pic)

    k=cv2.waitKey(30) & 0xFF
    if k ==2:

        break
cv2.destroyAllWindows()    