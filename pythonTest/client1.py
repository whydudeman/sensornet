import cv2
import io
import socket
import struct
import time
import pickle
import zlib
from gpiozero import LED

ledgreen=LED(26)
ledred=LED(19)
isKnown=False
start=0
boolProcess=False
ledred.on()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect(('192.168.0.152', 8485))
    connection = client_socket.makefile('wb')
    cam = cv2.VideoCapture(0)
    cam.set(3, 320);
    cam.set(4, 240);
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while True:
        ret, frame = cam.read()
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        send_data = pickle.dumps(frame, 0)
        size = len(send_data)
        client_socket.sendall(struct.pack(">L", size) + send_data)
        client_data = client_socket.recv(1024)
        client_data.decode()
        client_data=str(client_data)
        if(client_data!="b'unknown'"):
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
        print(client_data)
    cam.release()

