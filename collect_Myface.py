#-*- coding: utf-8 -*-

import cv2
import sys

from PIL import Image

classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)
num=0
while(1):
    # get a frame
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像

    #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(faceRects) > 0:          #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect

                #将当前帧保存为图片
                img_name = '%s/%d.jpg'%("me", num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)

                num += 1
                if num > (1000):   #如果超过指定最大保存数量退出循环
                    break

                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0,255,0), 2)

                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)

    if num > 1000: break

        #显示图像
    cv2.imshow("collect", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
