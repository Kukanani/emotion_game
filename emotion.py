#!/usr/bin/env python
import cv2
import sys
import numpy
import requests
import time
import operator
import math
import random

api_key = "***REMOVED***1"
emotions = {
    "neutral": "neutral",
    "happiness": "happy",
    "contempt": "contempt",
    "sadness": "sad",
    "disgust": "disgusted",
    "anger": "angry",
    "surprise": "surprised",
    "fear": "surprised" }
target_emotion = ""
faces = []
processed = False

def send_pic(img):
    global emotion
    global faces
    global processed

    img_str = cv2.imencode('.jpg', img)[1].tostring()
    data = img_str
    res = requests.post(url='https://api.projectoxford.ai/emotion/v1.0/recognize',
                        data=data,
                        headers={'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': api_key})

    if res.status_code == 200:
        faces = res.json()
    processed = True

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
pic_width = int(video_capture.get(3))

lastTime = time.time()
interval = 5
timeCount= 0
target_emotion = emotions[random.choice(emotions.keys()[:-1])]


while True:
    faces = []
    target_emotion = emotions[random.choice(emotions.keys()[:-1])]
    lastTime = time.time()
    while timeCount <= interval:
        # Capture frame-by-frameC
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
       # for (x, y, w, h) in faces:
           # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
        cv2.putText(frame, str(int(math.ceil(interval-timeCount))), (pic_width-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Make a " + target_emotion.upper() + " face...", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        timeCount = time.time() - lastTime


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
    cv2.putText(frame, "Processing...", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    send_pic(frame)
    timeCount = 0
    lastTime = time.time()

    ret, frame = video_capture.read()
    cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
    cv2.putText(frame, "Processing...", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    while not processed:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    processed = False

    if len(faces) > 0:
        correct = 0
        for face in faces:
            emotion = emotions[max(face['scores'].iteritems(), key=operator.itemgetter(1))[0]]
            left = int(face['faceRectangle']['left'])
            top = int(face['faceRectangle']['top'])
            width = int(face['faceRectangle']['width'])
            height = int(face['faceRectangle']['height'])
            color = (0, 0, 255)
            print emotion, target_emotion
            if emotion == target_emotion:
                correct += 1
                color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
            cv2.rectangle(frame, (left, top), (left + width, top - 40), color, -1)
            cv2.putText(frame, emotion.title(), (left,top-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)


        percentage = float(correct)/float(len(faces))
        print percentage
        cv2.rectangle(frame, (0, 0), (pic_width, 50), (255, 255, 255, 128), -1)
        if percentage > 0.5:
            cv2.putText(frame, "Good job!", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Try again!", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()