import cv2
import numpy as np
import tensorflow as tf
import csv
from tkinter import *
from tkinter import ttk

# Load pre-trained model
# save_model = tf.keras.models.load_model("khuonmat.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
webcam = cv2.VideoCapture(0)
webcam.set(3,640)
webcam.set(4,480)

imgBackground = cv2.imread('background.jpg')

# Initialize list to store attendance
attendance_list = []

while True:
    status, frame = webcam.read()
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    img = imgBackground[162:162 + 480, 55:55 + 640]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        face_gray = cv2.resize(face_gray, (100, 100))
        face_gray = np.array(face_gray).reshape((1, 100, 100, 1)) / 255.0
        result = save_model.predict(face_gray)
        final = np.argmax(result)
        labels = ["Hoang", "Hao"]  # Update with your class labels
        label_text = labels[final]

        cv2.putText(img, label_text, (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if 'q' key is pressed to mark attendance
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            attendance_list.append(label_text)
            cv2.putText(imgBackground, "Attended", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            break


    
    cv2.imshow('Face Detection', imgBackground)

    # Press 'ESC' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Write attendance list to CSV file
with open('attendance.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(attendance_list)

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
