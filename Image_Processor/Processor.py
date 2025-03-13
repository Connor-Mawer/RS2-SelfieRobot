import cv2
import numpy as np
import rembg
import time

# Capture Image from Webcam


cap = cv2.VideoCapture(0) #Ensure bluetooth is off (may connect to phone if it is on)
time.sleep(2)  # Necessary delay for the camera to run
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, image = cap.read()
cap.release()
cv2.imshow('initial image', image)

if not ret:
    print("Error: Could not read image from camera.")
    exit()


# ✅ (1) Remove Background Using rembg
image_nobg = rembg.remove(image)

if image_nobg is None:
    print("Error: Image not found. Check the file path!")
    exit()

# ✅ (2) Convert RGBA to BGR (Fix for OpenCV errors)
if image_nobg.shape[2] == 4:
    image_nobg = cv2.cvtColor(image_nobg, cv2.COLOR_BGRA2BGR)

# ✅ (3) Convert to Grayscale
gray = cv2.cvtColor(image_nobg, cv2.COLOR_BGR2GRAY)

# ✅ (4) Normalize for Better Edge Detection
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

# ✅ (5) Apply CLAHE to Enhance Face Details
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray_norm)

# ✅ (6) Canny Edge Detection
edges = cv2.Canny(gray_clahe, 50, 150)

# Display the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()