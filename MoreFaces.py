import PIL.Image
import PIL.ImageDraw
import cv2
from cv2 import imshow

import face_recognition

gbr = face_recognition.load_image_file('assets/rame.png')
gbr = cv2.cvtColor(gbr,cv2.COLOR_BGR2GRAY)
loc = face_recognition.face_locations(gbr)
wjh = len(loc)
print("Ada {} wajah ".format(wjh))
# cv2.imshow('gambar 1',gbr)
pil_image = PIL.Image.fromarray(gbr)

for fl in loc:
    atas,kanan,bawah,kiri = fl
    print("Wajah terdeteksi di lokasi Atas: {}, Kiri: {}, Bawah: {}, Kanan: {}".format(atas,kiri,bawah,kanan))
    kotak = PIL.ImageDraw.Draw(pil_image)
    kotak.rectangle([kiri,atas,kanan,bawah], outline="black")
    # cv2.rectangle(gbr,(atas,kanan),(bawah,kiri),(0,255,255),2)

pil_image.show()

cv2.waitKey(0)
