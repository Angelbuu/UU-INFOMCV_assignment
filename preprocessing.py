import numpy as np
import cv2 as cv
import glob
import os
import shutil

# 1. SETUP DIRECTORIES FIRST
# This prevents the FileNotFoundError you saw earlier
os.makedirs('images/success', exist_ok=True)
os.makedirs('images/fail', exist_ok=True)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# Note: (9,6) means 9 columns and 6 rows of internal corners
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Only look for images in the main folder (ignores subfolders)
images = glob.glob('images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    if img is None: continue
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    base_name = os.path.basename(fname)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        
        dest = os.path.join('images/success', base_name)
        print(f"Success: {base_name}")
    else:
        dest = os.path.join('images/fail', base_name)
        print(f"Failed: {base_name}")

    # Move the file to the appropriate subfolder
    shutil.move(fname, dest)

cv.destroyAllWindows()

# 2. CALIBRATION SAFETY CHECK
# Calibration will crash if objpoints is empty
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n--- Calibration Results ---")
    print("Camera matrix (Intrinsic Parameters):")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)

    # Save the new results
    np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nResults saved to B.npz")
else:
    print("\n No corners were detected in any images. B.npz was not created.")