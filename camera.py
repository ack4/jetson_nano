#!/usr/bin/env python3
# coding: utf-8
import cv2
#import tensorflow as tf
#import numpy as np


#***************************************
face_cascade=cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_eye.xml")
def face_detection(input_frame):
    # ファイルパスが渡された場合
    # img = cv2.imread(input)
    frame = input_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #検出した顔の座標リスト(原点左上x座標, y座標, 幅, 高さ)が返却される
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=5)
        face_color = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0),thickness=3)
    return frame

#object_cascade=cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_upperbody.xml")
object_cascade=cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
def object_detection(input_frame):
    frame=input_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #objects = object_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    objects = object_cascade.detectMultiScale(gray)
    for x,y,w,h in objects:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=5)
    return frame

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
GST_STR2 = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)800, height=(int)600, format=(string)BGRx \
    ! videoconvert \
    ! appsink'

def camera_capture(frame_method=None):
    capture = cv2.VideoCapture(GST_STR2, cv2.CAP_GSTREAMER)
    if frame_method is None:
        exit(0)

    count=0
    while True:
        ret, frame = capture.read()
        frame = frame_method(frame)
        cv2.imshow("jetson_camera", frame)
        cv2.moveWindow("jetson_camera", 200, 100)

        #save files
        #path="./jpg/test" + str(count).zfill(8) + ".jpg"
        #cv2.imwrite(path,frame)

        key = cv2.waitKey(10)
        if key == 27: # ESC
            break

        count=count+1

    capture.release()
    cv2.destroyAllWindows()

def test_frame(input_frame):
    #frame = cv2.rectangle(input_frame, (200, 200), (500, 500), (0,255,255),thickness=2)
    frame = cv2.rectangle(input_frame, (150, 150), (300, 300), (0,0,255),thickness=2)
    #frame = frame[150:300, 150:300]
    return frame


#camera_capture(face_detection)
camera_capture(object_detection)
#camera_capture(test_frame)
#camera_capture()


