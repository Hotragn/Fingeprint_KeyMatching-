# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import io

# def preprocess_fingerprint(image):
#     # Convert to grayscale if needed
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image
    
#     # Enhance contrast
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(gray)
    
#     # Denoise
#     denoised = cv2.fastNlMeansDenoising(enhanced)
    
#     return denoised  # Return denoised image without binarization for better feature detection

# def convert_uploaded_file_to_cv2(uploaded_file):
#     # Read file as bytes
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     # Decode the bytes as an image
#     opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     return opencv_img

# def match_fingerprints(sample_img, test_img):
#     try:
#         # Preprocess images
#         sample_processed = preprocess_fingerprint(sample_img)
#         test_processed = preprocess_fingerprint(test_img)
        
#         # Create SIFT detector
#         sift = cv2.SIFT_create()
        
#         # Detect and compute keypoints
#         kp1, desc1 = sift.detectAndCompute(sample_processed, None)
#         kp2, desc2 = sift.detectAndCompute(test_processed, None)
        
#         if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
#             return 0, [], None, None, None, None
        
#         # Create BFMatcher
#         bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
#         # Perform knn matching
#         matches = bf.knnMatch(desc1, desc2, k=2)
        
#         # Apply ratio test
#         good_matches = []
#         for m, n in matches:
#             if m.distance < 0.75 * n.distance:
#                 good_matches.append(m)
        
#         # Calculate score
#         score = (len(good_matches) / min(len(kp1), len(kp2))) * 100
        
#         return score, good_matches, kp1, kp2, sample_processed, test_processed
        
#     except Exception as e:
#         st.error(f"Error in matching: {str(e)}")
#         return 0, [], None, None, None, None

# def draw_matches(img1, kp1, img2, kp2, matches):
#     # Create a new output image that concatenates the two images
#     rows1, cols1 = img1.shape[:2]
#     rows2, cols2 = img2.shape[:2]
    
#     out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    
#     # Place the first image to the left
#     out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    
#     # Place the next image to the right of it
#     out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    
#     # For each pair of points we have between both images
#     for mat in matches:
#         # Get the matching keypoints for each of the images
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx

#         # x - columns, y - rows
#         (x1,y1) = kp1[img1_idx].pt
#         (x2,y2) = kp2[img2_idx].pt

#         # Draw a small circle at both co-ordinates
#         cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
#         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        
#         # Draw a line between the two points
#         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    
#     return out

# def main():
#     st.title("Fingerprint Matching System")
    
#     # File upload
#     col1, col2 = st.columns(2)
    
#     with col1:
#         sample = st.file_uploader("Upload First Fingerprint", type=['jpg', 'jpeg', 'png', 'tif'])
        
#     with col2:
#         test = st.file_uploader("Upload Second Fingerprint", type=['jpg', 'jpeg', 'png', 'tif'])
    
#     if sample is not None and test is not None:
#         try:
#             # Convert uploaded files to opencv format
#             sample_img = convert_uploaded_file_to_cv2(sample)
#             test_img = convert_uploaded_file_to_cv2(test)
            
#             if sample_img is None or test_img is None:
#                 st.error("Error loading images. Please try again.")
#                 return
            
#             # Display original images
#             st.subheader("Uploaded Images")
#             col3, col4 = st.columns(2)
#             with col3:
#                 st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB), caption="Sample Image")
#             with col4:
#                 st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Test Image")
            
#             # Process images and get matches
#             score, matches, kp1, kp2, proc_img1, proc_img2 = match_fingerprints(sample_img, test_img)
            
#             # Display score
#             st.subheader("Matching Results")
#             if score > 0:
#                 if score >= 70:
#                     st.success(f"Match Score: {score:.2f}%")
#                 elif score >= 40:
#                     st.warning(f"Match Score: {score:.2f}%")
#                 else:
#                     st.error(f"Match Score: {score:.2f}%")
                
#                 # Draw and display matches
#                 if len(matches) > 0:
#                     st.subheader("Matching Points Visualization")
#                     result_img = draw_matches(proc_img1, kp1, proc_img2, kp2, matches[:50])
#                     st.image(result_img, caption="Matching Points", use_column_width=True)
                    
#                     # Display additional information
#                     st.subheader("Match Details")
#                     st.write(f"Number of keypoints in sample: {len(kp1)}")
#                     st.write(f"Number of keypoints in test: {len(kp2)}")
#                     st.write(f"Number of good matches: {len(matches)}")
#             else:
#                 st.error("No matches found between the images.")
                
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import cv2
import numpy as np
from PIL import Image

def preprocess_fingerprint(image):
    if image is None:
        return None
    
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
    
    return denoised

def match_with_sift_only(sample_img, test_img):
    try:
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect and compute keypoints
        kp1, desc1 = sift.detectAndCompute(sample_img, None)
        kp2, desc2 = sift.detectAndCompute(test_img, None)
        
        if desc1 is None or desc2 is None:
            return 0, [], None, None
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Calculate score
        score = (len(good_matches) / len(kp1)) * 100 if len(kp1) > 0 else 0
        
        return score, good_matches, kp1, kp2
        
    except Exception as e:
        st.error(f"Error in matching: {str(e)}")
        return 0, [], None, None

def match_with_preprocessing(sample_img, test_img):
    try:
        # Preprocess images
        sample_processed = preprocess_fingerprint(sample_img)
        test_processed = preprocess_fingerprint(test_img)
        
        if sample_processed is None or test_processed is None:
            return 0, [], None, None, None, None
        
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect and compute keypoints
        kp1, desc1 = sift.detectAndCompute(sample_processed, None)
        kp2, desc2 = sift.detectAndCompute(test_processed, None)
        
        if desc1 is None or desc2 is None:
            return 0, [], None, None, None, None
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                good_matches.append(p)
        
        # Calculate score
        keypoints = min(len(kp1), len(kp2))
        score = (len(good_matches) / keypoints) * 100 if keypoints > 0 else 0
            
        return score, good_matches, kp1, kp2, sample_processed, test_processed
        
    except Exception as e:
        st.error(f"Error in matching: {str(e)}")
        return 0, [], None, None, None, None

def convert_uploaded_file_to_cv2(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return opencv_img

def main():
    st.title("Fingerprint Matching System")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        sample = st.file_uploader("Upload Sample Fingerprint", type=['jpg', 'jpeg', 'png', 'tif'])
        
    with col2:
        test = st.file_uploader("Upload Test Fingerprint", type=['jpg', 'jpeg', 'png', 'tif'])
    
    if sample is not None and test is not None:
        try:
            # Convert uploaded files to opencv format
            sample_img = convert_uploaded_file_to_cv2(sample)
            test_img = convert_uploaded_file_to_cv2(test)
            
            if sample_img is None or test_img is None:
                st.error("Error loading images. Please try again.")
                return
            
            # Display original images
            st.subheader("Uploaded Images")
            col3, col4 = st.columns(2)
            with col3:
                st.image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB), caption="Sample Image")
            with col4:
                st.image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), caption="Test Image")
            
            # Add buttons for different matching options
            col5, col6 = st.columns(2)
            
            with col5:
                if st.button("Match with Preprocessing"):
                    score, matches, kp1, kp2, proc_img1, proc_img2 = match_with_preprocessing(sample_img, test_img)
                    
                    if proc_img1 is not None and proc_img2 is not None:
                        st.subheader("Preprocessed Images")
                        col7, col8 = st.columns(2)
                        with col7:
                            st.image(proc_img1, caption="Preprocessed Sample")
                        with col8:
                            st.image(proc_img2, caption="Preprocessed Test")
                    
                    display_results(score, matches, kp1, kp2, sample_img, test_img, "Preprocessed")
            
            with col6:
                if st.button("Match Directly (SIFT only)"):
                    score, matches, kp1, kp2 = match_with_sift_only(sample_img, test_img)
                    display_results(score, matches, kp1, kp2, sample_img, test_img, "Direct")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def display_results(score, matches, kp1, kp2, img1, img2, method):
    st.subheader(f"Matching Results ({method})")
    if score > 0:
        if score >= 70:
            st.success(f"Match Score: {score:.2f}%")
        elif score >= 40:
            st.warning(f"Match Score: {score:.2f}%")
        else:
            st.error(f"Match Score: {score:.2f}%")
        
        if len(matches) > 0:
            result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), 
                    caption="Matching Points Visualization",
                    use_column_width=True)
            
            st.subheader("Match Details")
            st.write(f"Number of keypoints in sample: {len(kp1)}")
            st.write(f"Number of keypoints in test: {len(kp2)}")
            st.write(f"Number of good matches: {len(matches)}")
            
            confidence = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
            st.write(f"Match Confidence: {confidence}")
    else:
        st.error("No matches found between the images.")

if __name__ == "__main__":
    main()