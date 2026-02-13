import numpy as np
import cv2 as cv
import glob
import os
import shutil

# 1. SETUP DIRECTORIES
os.makedirs('images/success', exist_ok=True)
os.makedirs('images/fail', exist_ok=True)

# Termination criteria and target resolution
standard_size = (4284, 5712) 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D real world space)
# Each square is 20mm (2cm) [cite: 1]
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 20

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Only look for images in the main folder
images = glob.glob('images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    if img is None: continue
    
    h, w = img.shape[:2]
    # Standardize resolution for consistency across all runs 
    if (w, h) != standard_size:
        print(f"Standardizing resolution for: {fname}")
        img = cv.resize(img, standard_size, interpolation=cv.INTER_AREA)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    base_name = os.path.basename(fname)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners for visual verification
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100) # Shortened wait time to speed up processing
        
        dest = os.path.join('images/success', base_name)
        print(f"Success: {base_name}")
    else:
        dest = os.path.join('images/fail', base_name)
        print(f"Failed: {base_name}")

    shutil.move(fname, dest)

cv.destroyAllWindows()

# 2. SAVE RAW POINTS FOR THE CALIBRATION SCRIPT
if len(objpoints) > 0:
    # We save only the raw points here so they can be merged with manual points later 
    np.savez('B.npz', objpoints=objpoints, imgpoints=imgpoints)
    print(f"\nCaptured points for {len(objpoints)} images and saved to B_points.npz.")
else:
    print("\nNo corners were detected. Check your image folder.")