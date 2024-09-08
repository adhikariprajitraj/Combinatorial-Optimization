import cv2
import numpy as np


def enhance_fingerprint(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image has loaded correctly
    if img is None:
        print("Error: Image did not load.")
        return

    # Step 1: Adaptive Thresholding to handle different lighting conditions in the image
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # Step 2: Morphological operation to clean small noises
    kernel = np.ones((3, 3), np.uint8)
    img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 3: Enhance the ridges using ridge filtering
    # Define the Sobel derivatives to get the x and y gradients
    sobelx = cv2.Sobel(img_clean, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_clean, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    
    # Scale the magnitude to range 0-255
    magnitude = np.clip((magnitude / magnitude.max()) * 255, 0, 255).astype(np.uint8)
    
    # Step 4: Binarize the result to get clear ridges
    _, img_final = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Create a color image with white background
    color_image = np.ones((img_final.shape[0], img_final.shape[1], 3), dtype=np.uint8) * 255
    
    # Set the ridges to blue (OpenCV uses BGR color format)
    color_image[img_final == 255] = [255, 0, 0]  # Blue
    
    # Display the result
    cv2.imshow('Enhanced Fingerprint in Blue', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Specify your image path
enhance_fingerprint('fingerprints/right thumb.jpeg')
