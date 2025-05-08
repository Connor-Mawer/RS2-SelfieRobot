import svgwrite
import cv2
import numpy as np

def save_as_svg(line_image):
    """
    Converts a binary edge image into an SVG string with connected paths.

    Args:
        line_image (numpy.ndarray): Binary edge image (e.g., from Canny edge detection).

    Returns:
        str: SVG content as a string.
    """
    height, width = line_image.shape
    dwg = svgwrite.Drawing(size=(width, height))

    # Find contours in the binary edge image
    contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour
    for contour in contours:
        if len(contour) > 1:  # Only process contours with more than one point
            path_data = []
            for point in contour:
                x, y = point[0]  # Extract x, y coordinates
                path_data.append((float(x), float(y))) # Append as a tuple of coordinates
            # Add the path to the SVG
            dwg.add(dwg.polyline(points=path_data, stroke="black", fill="none", stroke_width=1))

    # Return the SVG content as a string
    return dwg.tostring()