import cv2
import numpy as np


classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

while(1):
    # get a frame
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # img_name = '%s/%d.jpg'%(path_name, num)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)
            # cv2.imwrite(img_name, image)

    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
