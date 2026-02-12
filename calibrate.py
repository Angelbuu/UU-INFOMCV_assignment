import numpy as np
import cv2
import glob

# --- CONFIGURATION ---
CHESSBOARD_SIZE = (9, 6) # CHANGE THIS to match your specific board (inner corners!)
SQUARE_SIZE = 0.025      # CHANGE THIS to your square size in meters (e.g., 2.5cm)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
# This represents the "ideal" board in the real world (Z=0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Arrays to store object points and image points from all valid images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Load your images
images = glob.glob('calibration_images/*.jpg') # Make sure your images are in this folder

print(f"Found {len(images)} images. Processing...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # flags=None can be replaced with cv2.CALIB_CB_ADAPTIVE_THRESH for better results
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine corner accuracy (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Optional: Draw and display the corners to check
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100) # Show for 100ms
    else:
        print(f"Could not find corners in {fname} - Save this for Manual Step!")

cv2.destroyAllWindows()

# --- CALIBRATION ---
print("Calibrating camera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the data for the Online Phase
np.savez('camera_params.npz', mtx=mtx, dist=dist)

print("Calibration successful. Matrix saved to 'camera_params.npz'")
print("Camera Matrix:\n", mtx)
print("Distortion Coeffs:\n", dist)