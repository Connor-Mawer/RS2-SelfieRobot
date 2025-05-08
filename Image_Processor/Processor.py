import cv2
import numpy as np
import rembg
import time
from svgpathtools import Path, Line, wsvg
import lineGenerator
import faceDetector
import svgSaver

def ImageProcessor():
    cropped_face = faceDetector.detect_and_crop_face()

    lineImage = lineGenerator.lineGenerator(cropped_face)
    svg = svgSaver.save_as_svg(lineImage, "output.svg")
    return svg

svg = ImageProcessor()

