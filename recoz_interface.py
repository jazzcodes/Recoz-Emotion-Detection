from flask import Flask, render_template, request, redirect, url_for, session
import attendanceProject
import face_recognition
import os
from datetime import datetime
import mysql.connector as db
import my_speech
import pyttsx3
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import keras
from keras.models import load_model
from time import sleep
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def front():
    return render_template("recoz.html")

@app.route('/home')
def home():
    connection = db.connect(user='root', password='Gurleen@123', host='127.0.0.1', database='Recoz')
    cursor = connection.cursor()
    print("[DB] CONNECTION ESTABLISHED")

    text_speech_2 = pyttsx3.init()
    my_speech.Recoz_speech()

    path = 'TrainingImageLabel'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    nameSet = set(["abc"])

    def markAttendance(name):
        if name not in nameSet:
            s_name = name
            insrt_state = (
                "insert into student (s_name, time_of_attendance)"
                "values(%s, %s)")
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            data = (s_name, dtString)
            cursor.execute(insrt_state, data)
            connection.commit()
            nameSet.add(name)
            print("[DB] SQL EXECUTED")
        else:
            nameSet.add(name)

        """
        with open('Attendance.csv', 'r+') as f: 
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
                # text_speech_2.say(entry[0])
                # text_speech_2.runAndWait()

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                text_speech_2.say(name)
                att="Your attendance has been marked"
                text_speech_2.say(att)
                text_speech_2.runAndWait()
        """

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    classifier = load_model('./Emotion_Detection.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            # print(classification_report(encodeListKnown, encodeFace))
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        # ret, frame = cap.read()
        frame = img
        ret = success
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                print("\nprediction = ", preds)
                label = class_labels[preds.argmax()]
                print("\nprediction max = ", preds.argmax())
                print("\nlabel = ", label)
                if (label == "Happy"):
                    att = "Always be happy like this"
                    text_speech_2.say(att)
                    text_speech_2.runAndWait()
                elif (label == "Sad"):
                    att = "Hey be cheerful, have a great day ahead"
                    text_speech_2.say(att)
                    text_speech_2.runAndWait()
                elif (label == "Angry"):
                    att = "Its good to forgive some mitakes"
                    text_speech_2.say(att)
                    text_speech_2.runAndWait()
                elif (label == "Neutral"):
                    att = "Its okay not to feel anything sometimes"
                    text_speech_2.say(att)
                    text_speech_2.runAndWait()
                elif (label == "Surprise"):
                    att = "yes! i know this system is great donot be surprised"
                    text_speech_2.say(att)
                    text_speech_2.runAndWait()
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            print("\n\n")
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template("recoz.html")

if __name__ == "__main__":
    app.run(debug = True)






