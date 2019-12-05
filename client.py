import cv2
import io
import struct
import time
import pickle
import zlib
import json
import requests
import base64
from gpiozero import LED
while True:
    try:
        ledgreen=LED(26)
        ledred=LED(19)
        isKnown=False
        start=0
        boolProcess=False
        ledred.on()
        error_handler=""
    except:
        print("Something wrong with leds")
    try:
        cam = cv2.VideoCapture(0)
        cam.set(3, 320);
        cam.set(4, 240);
        person_id=""
    except:
        print("Something wrong with Camera initialization")
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
                frame = cv2.imencode('.jpg', frame)[1]
                # print(frame.tostring
                frame=base64.b64encode(frame)
                data = {"company" : "yeah", "image" : frame}
                r1=requests.post(url="http://192.168.0.151:2998/recognize",data=data)
                # print(r1.text)
                data=r1.json()
                try:
                    person_id=data['prediction']
                    error_handler=data['result']
                    print(person_id)
                    print(error_handler)
                except(IndexError,KeyError):
                    print("Something wrong")
                    boolProcess=True
                if(person_id!="-1" and error_handler=="OK"):
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
                error_handler=""
                person_id=""
        cam.release()
