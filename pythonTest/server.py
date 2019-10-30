import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import math
from sklearn import neighbors
import os
import os.path
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
def predict(get_frame, knn_clf=None, model_path="/dataset", distance_threshold=0.389):
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_face_locations = face_recognition.face_locations(get_frame)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(get_frame, known_face_locations=X_face_locations)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



HOST=''
PORT=8485
identificator="unknown"
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print('Socket created')

    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn,addr=s.accept()
    with conn:
        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))
        while True:
            while len(data) < payload_size:
                print("Recv: {}".format(len(data)))
                data += conn.recv(4096)
            
            print("Done Recv: {}".format(len(data)))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            predictions=predict(frame,knn_clf=None,model_path="dataset/trained_knn_model.clf",distance_threshold=0.389)
            print(predictions)
            if not predictions:
                identificator="unknown"
                print("There is no face")
            for name, (top, right, bottom, left) in predictions:
                identificator=name
                print("Nameis"+name)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('ImageWindow',frame)
            cv2.waitKey(1)
            conn.sendall(identificator.encode())

