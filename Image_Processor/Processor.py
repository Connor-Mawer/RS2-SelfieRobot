import cv2
import numpy as np
import rembg
import time
import webbrowser
from svgpathtools import Path, Line, wsvg


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

if image_nobg is None:
    print("Error: Image not found. Check the file path!")
    exit()

# Convert RGBA to BGR 
image_nobg = cv2.cvtColor(image_nobg, cv2.COLOR_BGRA2BGR)

# Convert to Grayscale
gray = cv2.cvtColor(image_nobg, cv2.COLOR_BGR2GRAY)

#  Normalize for Better Edge Detection -- Trying to solve the issue where chin is not detected in the image. May be able to solve with lighting
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_norm, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
x_face, y_face, w_face, h_face = face[0]
print (x_face, y_face, w_face, h_face)

# ✅ (5) Apply CLAHE to Enhance Face Details
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray_norm)

# ✅ (6) Apply Gaussian Blur to Smooth Edges

blurred = cv2.GaussianBlur(gray_clahe, (9, 9), 0)  # Use a small kernel size
# ✅ (7) Canny Edge Detection
edges = cv2.Canny(blurred, 30, 140)  # Adjust the thresholds as needed but this seems the best so far

cv2.imshow('Edges', edges)

# ✅ (8) Find and Filter Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Adjust the area threshold as needed

smoothed_contours = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in filtered_contours]

# Trim lines to keep only points inside the face bounding box
trimmed_Lines = []
for contour in smoothed_contours:
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