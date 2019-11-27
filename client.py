import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import json
import requests
import http.client as httplib
from requests_toolbelt import MultipartEncoder
from PIL import Image
import base64
from gpiozero import LED
ledgreen=LED(26)
ledred=LED(19)
isKnown=False
start=0
boolProcess=False
ledred.on()
cam = cv2.VideoCapture(0)
cam.set(3, 320);
cam.set(4, 240);
while cam.isOpened():
    ret, frame = cam.read()

    if ret:
        frame = cv2.imencode('.jpg', frame)[1]
        # print(frame.tostring())
        
        frame=base64.b64encode(frame)
        data = {"company" : "yeah", "image" : frame}
        r1=requests.post(url="http://0.0.0.0:2998/recognize",data=data)
        # print(r1.text)
        data=r1.json()
        try:
            person_id=data['prediction']
            print(person_id)
        except(IndexError,KeyError):
            print("Something wrong")
        if(person_id!="-1"):
            isKnown=True
            start=time.time()
        if(isKnown==True):
            boolProcess=True
        if(time.time()-start<10):
            ledred.off()
            ledgreen.on()
        if(time.time()-start>11):
            start=0
            ledred.on()
            ledgreen.off()
            print("ledoff")
            boolProcess=False
            isKnown=False
        
    

cam.release()
cv2.destoryAllWindows()