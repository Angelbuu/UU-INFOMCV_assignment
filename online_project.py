import numpy as np
import cv2 as cv

# --- CONFIGURATION (Must match your Offline Phase) ---
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 20  # You used * 20 in preprocessing
SQUARE_SIZE_M = 0.02

# 1. LOAD YOUR CALIBRATION (Use Run 1)
with np.load('final_calibration_results.npz') as data:
    mtx, dist = data['mtx'], data['dist']
    if mtx.ndim == 3:  # npz has 3 runs stacked
        mtx, dist = mtx[0], dist[0]

# 2. DEFINE 3D OBJECTS (Axes and Cube)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * SQUARE_SIZE_MM
cube = np.float32([[0,0,0], [2,0,0], [2,2,0], [0,2,0],
                   [0,0,-2], [2,0,-2], [2,2,-2], [0,2,-2]]) * SQUARE_SIZE_MM

cap = cv.VideoCapture(0) # Use 0 for MacBook webcam

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Pose Estimation
        objp = np.zeros((9*6, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * SQUARE_SIZE_MM
        _, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        # Project Points
        imgpts_axis, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
        imgpts_cube, _ = cv.projectPoints(cube, rvec, tvec, mtx, dist)
        imgpts_cube = np.int32(imgpts_cube).reshape(-1,2)

        # --- DRAWING ---
        # 1. Axes
        origin = tuple(np.int32(corners2[0].ravel()))
        frame = cv.line(frame, origin, tuple(np.int32(imgpts_axis[0].ravel())), (0,0,255), 5)
        frame = cv.line(frame, origin, tuple(np.int32(imgpts_axis[1].ravel())), (0,255,0), 5)
        frame = cv.line(frame, origin, tuple(np.int32(imgpts_axis[2].ravel())), (255,0,0), 5)

        # 2. Distance & Orientation Math
        R, _ = cv.Rodrigues(rvec)
        # Center of top plane in camera coords
        center_cam = R @ np.array([[20, 20, -40]]).T + tvec
        dist_m = np.linalg.norm(center_cam) / 1000.0
        
        # Angle for Hue
        normal_cam = R @ np.array([0, 0, -1])
        angle = np.degrees(np.arccos(np.clip(np.abs(normal_cam[2]), 0, 1)))

        # 3. HSV Color Logic
        v = np.clip(255 * (1 - dist_m / 4.0), 0, 255)
        h = np.clip(179 * (1 - angle / 45.0), 0, 179)
        color_bgr = cv.cvtColor(np.uint8([[[h, 255, v]]]), cv.COLOR_HSV2BGR)[0][0]
        color_bgr = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        # 4. Draw Cube with Color
        cv.fillConvexPoly(frame, imgpts_cube[4:], color_bgr)
        cv.drawContours(frame, [imgpts_cube[:4]], -1, (255,255,255), 2) # Bottom
        for i, j in zip(range(4), range(4,8)): # Verticals
            cv.line(frame, tuple(imgpts_cube[i]), tuple(imgpts_cube[j]), (255,255,255), 2)
        
        # 5. Text Overlay
        center_px = tuple(imgpts_cube[4:].mean(axis=0).astype(int))
        cv.putText(frame, f"{dist_m:.2f}m", center_px, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv.imshow('INFOMCV Real-Time AR', frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()