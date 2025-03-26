import cv2
import numpy as np
import rembg
import time
import webbrowser
from svgpathtools import Path, Line, wsvg
import lineGenerator


# Capture Image from Webcam
cap = cv2.VideoCapture(0)  # Ensure Bluetooth is off (may connect to phone if it is on and cause errors where it tries to take the photo from your iPhone)
time.sleep(2)  # Necessary delay for the camera to run on my local Mac

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, image = cap.read()
cap.release()
cv2.imshow('initial image', image)

if not ret:
    print("Error: Could not read image from camera.")
    exit()

# Remove Background Using rembg -- https://www.youtube.com/watch?v=HzwNMYvUXi4
image_nobg = rembg.remove(image)

# Convert RGBA to BGR 
image_nobg = cv2.cvtColor(image_nobg, cv2.COLOR_BGRA2BGR)

#https://www.datacamp.com/tutorial/face-detection-python-opencv
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    image_nobg, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
x_face, y_face, w_face, h_face = face[0]
print (x_face, y_face, w_face, h_face)

lineImage = lineGenerator.lineGenerator(image_nobg)






# ------- Trim lines to keep only points inside the face bounding box ---- WORK IN PROGRESSS
trimmed_Lines = []
for contour in lineImage:
    for point in contour:
        x, y = point[0]
        # Keep points inside the bounding box
        if x_face <= x <= x_face + w_face and y_face <= y <= y_face + h_face:
            trimmed_Lines.append(point)


# Create paths using svgpathtools
paths = []
for contour in filtered_contours:
    path = Path()
    for i in range(len(contour) - 1):
        start = complex(contour[i][0][0], contour[i][0][1])
        end = complex(contour[i + 1][0][0], contour[i + 1][0][1])
        path.append(Line(start, end))
    paths.append(path)

# Save the paths to an SVG file
padding = 10  # Add padding around the paths
viewbox = (x_face - padding, y_face - padding, w_face + 2 * padding, h_face + 2 * padding)
wsvg(paths, filename='output.svg', dimensions=(w_face, h_face), viewbox=viewbox, stroke_widths=[1] * len(paths))

# Open the SVG file in the default web browser
webbrowser.open('output.svg')