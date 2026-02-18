import numpy as np
import cv2 as cv
import os

# --- 1. CONFIGURATION ---
# Must match your Offline Phase settings
SQUARE_SIZE_MM = 20  
STANDARD_RES = (4284, 5712)

def get_hsv_color(distance_m, angle_deg):
    """
    Assignment requirements:
    - Intensity (V): 255 at 0m, 0 at 4m+ (linear scale)
    - Hue (H): Max at parallel (0°), 0 at 45°+ (linear scale)
    - Saturation (S): Constant at 255
    """
    # V (Intensity): 255 at 0m, 0 at 4m+
    v = np.clip(255 * (1 - distance_m / 4.0), 0, 255)
    
    # H (Hue): Max (179 in OpenCV) at parallel, 0 at 45°+
    h = np.clip(179 * (1 - angle_deg / 45.0), 0, 179)
    
    # S (Saturation): Constant 255 per assignment
    s = 255
    
    hsv_pixel = np.uint8([[[h, s, v]]])
    bgr_pixel = cv.cvtColor(hsv_pixel, cv.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in bgr_pixel)

def process_and_draw(image_path, mtx, dist, run_index):
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    # 1. Standardize resolution to match calibration
    img = cv.resize(img, STANDARD_RES)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 2. Find Corners on Test Image 
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if not ret:
        print(f"Run {run_index+1}: Chessboard not found in test image.")
        return

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 3. Pose Estimation (Extrinsics) 
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE_SIZE_MM
    _, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

    # 4. Define 3D Objects at Origin (0,0,0) 
    # Axes lines
    axis_3d = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) * SQUARE_SIZE_MM
    # Cube (2x2 squares wide/high)
    cube_3d = np.float32([[0,0,0], [40,0,0], [40,40,0], [0,40,0],
                          [0,0,-40], [40,0,-40], [40,40,-40], [0,40,-40]])

    # Project to 2D
    imgpts_axes, _ = cv.projectPoints(axis_3d, rvec, tvec, mtx, dist)
    imgpts_cube, _ = cv.projectPoints(cube_3d, rvec, tvec, mtx, dist)
    imgpts_cube = np.int32(imgpts_cube).reshape(-1, 2)

    # 5. Calculation for Top Plane 
    # Center of top plane in camera coordinates
    center_3d_world = np.array([[20.0, 20.0, -40.0]], dtype=np.float32)
    R, _ = cv.Rodrigues(rvec)
    center_cam = R @ center_3d_world.T + tvec
    dist_m = np.linalg.norm(center_cam) / 1000.0 # Convert mm to m

    # Orientation Angle (Normal vs Camera Z-axis)
    normal_cam = R @ np.array([0, 0, -1])
    angle_deg = np.degrees(np.arccos(np.clip(np.abs(normal_cam[2]), 0, 1)))

    # 6. DRAWING
    # Axes: Red=X, Green=Y, Blue=Z 
    origin = tuple(np.int32(corners2[0].ravel()))
    img = cv.line(img, origin, tuple(np.int32(imgpts_axes[0].ravel())), (0,0,255), 15)
    img = cv.line(img, origin, tuple(np.int32(imgpts_axes[1].ravel())), (0,255,0), 15)
    img = cv.line(img, origin, tuple(np.int32(imgpts_axes[2].ravel())), (255,0,0), 15)

    # Cube Edges
    cv.drawContours(img, [imgpts_cube[:4]], -1, (255,255,255), 5)
    for i, j in zip(range(4), range(4,8)):
        cv.line(img, tuple(imgpts_cube[i]), tuple(imgpts_cube[j]), (255,255,255), 5)

    # Colored Top Plane 
    color_bgr = get_hsv_color(dist_m, angle_deg)
    cv.fillConvexPoly(img, imgpts_cube[4:], color_bgr)
    
    # Center Dot and Distance Text 
    center_2d, _ = cv.projectPoints(center_3d_world, rvec, tvec, mtx, dist)
    center_px = tuple(np.int32(center_2d.ravel()))
    cv.circle(img, center_px, 20, (255, 255, 255), -1)
    cv.putText(img, f"{dist_m:.2f}m", center_px, cv.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)

    # Save final result
    out_name = f"result_run_{run_index + 1}_ct3.jpg"
    cv.imwrite(out_name, img)
    print(f"Saved {out_name}: Distance={dist_m:.2f}m, Angle={angle_deg:.1f}deg")

def main():
    test_img_path = 'images/test_image.jpg' # Make sure your best tilted image is here!
    calib_file = 'final_calibration_results.npz'

    if not os.path.exists(calib_file):
        print(f"Error: {calib_file} not found.")
        return

    data = np.load(calib_file)
    
    # Process all three runs stored in your .npz
    for i in range(3):
        print(f"Processing Run {i+1}...")
        process_and_draw(test_img_path, data['mtx'][i], data['dist'][i], i)

if __name__ == "__main__":
    main()