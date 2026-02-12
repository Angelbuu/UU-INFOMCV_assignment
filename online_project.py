import numpy as np
import cv2

# --- CONFIGURATION ---
CHESSBOARD_SIZE = (9, 6) # Same as before
SQUARE_SIZE = 0.025

# Load previously saved calibration data
with np.load('camera_params.npz') as data:
    mtx, dist = data['mtx'], data['dist']

# Define the 3D axis coordinates you want to draw
# This draws 3 lines of length 3 * SQUARE_SIZE along X, Y, Z axes
axis = np.float32([[3*SQUARE_SIZE,0,0], [0,3*SQUARE_SIZE,0], [0,0,-3*SQUARE_SIZE]]).reshape(-1,3)

# Start Webcam (or load a video file)
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret == True:
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # --- ESSENTIAL STEP: SOLVE PnP ---
        # Calculate the rotation and translation vectors for this specific frame
        # objp is the same "ideal grid" you defined in Phase 1
        objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE
        
        # Find rotation (rvecs) and translation (tvecs)
        _, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # Project the 3D axis points onto the 2D image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # Draw the lines
        corner = tuple(corners2[0].ravel().astype(int))
        frame = cv2.line(frame, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5) # Blue: X
        frame = cv2.line(frame, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5) # Green: Y
        frame = cv2.line(frame, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5) # Red: Z

    cv2.imshow('AR Projection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()