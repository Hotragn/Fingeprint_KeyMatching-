import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

def preprocess_fingerprint(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Binarization
    _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    return binary

def match_fingerprints(sample_img, test_img):
    # Preprocess both images
    sample_processed = preprocess_fingerprint(sample_img)
    test_processed = preprocess_fingerprint(test_img)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect and compute keypoints
    kp1, desc1 = sift.detectAndCompute(sample_processed, None)
    kp2, desc2 = sift.detectAndCompute(test_processed, None)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Calculate matching score
    score = len(good_matches) / min(len(kp1), len(kp2)) * 100
    
    return score, good_matches, kp1, kp2

def main():
    st.title("Fingerprint Matching System")
    
    # File upload
    sample = st.file_uploader("Upload Sample Fingerprint", type=['jpg', 'jpeg', 'png', 'tif'])
    test = st.file_uploader("Upload Test Fingerprint", type=['jpg', 'jpeg', 'png', 'tif'])
    
    if sample and test:
        # Convert uploaded files to opencv format
        sample_img = cv2.imdecode(np.fromstring(sample.read(), np.uint8), 1)
        test_img = cv2.imdecode(np.fromstring(test.read(), np.uint8), 1)
        
        # Match fingerprints
        score, matches, kp1, kp2 = match_fingerprints(sample_img, test_img)
        
        # Display results
        st.write(f"Matching Score: {score:.2f}%")
        
        # Draw matches
        result = cv2.drawMatches(sample_img, kp1, test_img, kp2, matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Convert BGR to RGB for display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Matching Results", use_column_width=True)

if __name__ == "__main__":
    main()