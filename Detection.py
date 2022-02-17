import cvlib as cv
import cv2
import numpy as np


gbr = cv2.imread('assets/rame.png')

faces, confidences = cv.detect_face(gbr)

padding = 20

for wjh,conf in zip(faces,confidences):
    (startX,startY) = max(0, wjh[0]-padding),max(0, wjh[1]-padding)
    (endX,endY) = min(gbr.shape[1]-1, wjh[2]+padding),min(gbr.shape[0]-1, wjh[3]+padding)

    #membuat kotak di sekitar wajah
    cv2.rectangle(gbr, (startX,startY),(endX,endY),(255,0,0),2)

    crop_wjh = np.copy(gbr[startY:endY, startX:endX])

    #cek gender
    (label,conf) = cv.detect_gender(crop_wjh)

    print(conf)
    print(label)

    idx = np.argmax(conf)
    label = label[idx]

    label = "{}: {:.2f}%".format(label,conf[idx]*100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    cv2.putText(gbr, label, (startX, Y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0),2)


#output

cv2.imshow("deteksi",gbr)
cv2.waitKey(0)
