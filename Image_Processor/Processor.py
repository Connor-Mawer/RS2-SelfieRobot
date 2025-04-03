import cv2
import numpy as np
import rembg
import time
import webbrowser
from svgpathtools import Path, Line, wsvg
import lineGenerator
import faceDetector


cropped_face = faceDetector.detect_and_crop_face()

lineImage = lineGenerator.lineGenerator(cropped_face)

cv2.imshow('Face Detection', lineImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------- Trim lines to keep only points inside the face bounding box ---- WORK IN PROGRESSS

