import cv2 as cv
import numpy as np

DST_SIZE = 800 

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20


def get_warped_points(img_path):
    """
    Warp image via 4 user clicks, detect chessboard corners, return corners in original coords.
    Returns None if image fails to load or corners cannot be found after warping.
    """
    img = cv.imread(img_path)
    if img is None:
        return None
    
    display_img = img.copy()
    clicked_pts = []

    print(f"\nProcessing: {img_path}")
    
    def mouse_callback(event, x, y, flags, param):
        """On left-click: store (x,y) and draw green circle for feedback."""
        if event == cv.EVENT_LBUTTONDOWN:
            clicked_pts.append([x, y])
            cv.circle(display_img, (x, y), 20, (0, 255, 0), -1)
            cv.imshow("Choice Task 3: Manual Warp", display_img)

    cv.namedWindow("Choice Task 3: Manual Warp", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Choice Task 3: Manual Warp", mouse_callback)
    cv.imshow("Choice Task 3: Manual Warp", display_img)
    
    while len(clicked_pts) < 4:
        cv.waitKey(1)
    cv.destroyAllWindows()

    src = np.float32(clicked_pts)
    dst = np.float32([[0, 0], [DST_SIZE, 0], [DST_SIZE, DST_SIZE], [0, DST_SIZE]])
    
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, (DST_SIZE, DST_SIZE))
    
    gray_warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    ret, corners_warped = cv.findChessboardCorners(gray_warped, (9, 6), None)
    
    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_warped = cv.cornerSubPix(gray_warped, corners_warped, (5, 5), (-1, -1), criteria)
        
        M_inv = np.linalg.inv(M)
        original_corners = cv.perspectiveTransform(corners_warped.reshape(-1, 1, 2), M_inv)
        return original_corners
    
    print("Failed to detect corners even after warping.")
    return None


manual_files = [
    'images/fail/IMG_5940.jpg',
    'images/fail/IMG_5944.jpg',
    'images/fail/IMG_5947.jpg',
    'images/fail/IMG_5950.jpg',
    'images/fail/IMG_5958.jpg',
]
all_obj = []
all_img = []

for f in manual_files:
    res = get_warped_points(f)
    if res is not None:
        all_obj.append(objp)
        all_img.append(res)

np.savez('F_task3.npz', objpoints=all_obj, imgpoints=all_img)
