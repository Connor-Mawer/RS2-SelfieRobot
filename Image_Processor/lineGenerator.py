from cv2 import cv2
def lineGenerator(image):
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  Normalize for Better Edge Detection -- Trying to solve the issue where chin is not detected in the image. May be able to solve with lighting
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply CLAHE to Enhance Face Details
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray_norm)

    #  Apply Gaussian Blur to Smooth Edges

    blurred = cv2.GaussianBlur(gray_clahe, (9, 9), 0)  # Use a small kernel size

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 140)  # Adjust the thresholds as needed but this seems the best so far

    # ✅ (8) Find and Filter Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            filtered_contours.append(contour) 
