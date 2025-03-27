import cv2
import numpy as np
import rembg
import time
import webbrowser
from svgpathtools import Path, Line, wsvg


# Capture Image from Webcam
cap = cv2.VideoCapture(0)  # Ensure Bluetooth is off (may connect to phone if it is on and cause errors where it tries to take the photo from your iPhone)
time.sleep(2)  # Necessary delay for the camera to run on my local Mac

ret, image = cap.read()
cap.release()
cv2.imshow('initial image', image)

# Remove Background Using rembg -- https://www.youtube.com/watch?v=HzwNMYvUXi4
image_nobg = rembg.remove(image)

if image_nobg is None:
    print("Error: Image not found. Check the file path!")
    exit()

# Convert RGBA to BGR 
image_nobg = cv2.cvtColor(image_nobg, cv2.COLOR_BGRA2BGR)
cv2.imshow('initial image', image_nobg)

#https://www.datacamp.com/tutorial/face-detection-python-opencv
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    image_nobg, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
for (x_face, y_face, w_face, h_face) in face:
    cv2.rectangle(image_nobg, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
    cv2.circle(image_nobg, (x_face, y_face), 5, (0, 255, 0), -1)

cv2.imshow('Face Detection', image_nobg)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
