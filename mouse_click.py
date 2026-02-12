from re import S
import numpy as np
import cv2 as cv
import glob


def bilinear_interpolation(P00, P10, P01, P11, rows, cols):
    P00 = np.array(P00)
    P10 = np.array(P10)
    P01 = np.array(P01)
    P11 = np.array(P11)

    grid = []

    for j in range(rows):
        v = j / (rows - 1)
        row_points = []
        for i in range(cols):
            u = i / (cols - 1)

            P = (
                (1 - u) * (1 - v) * P00 +
                u * (1 - v) * P10 +
                (1 - u) * v * P01 +
                u * v * P11
            )

            row_points.append(tuple(P))
        grid.append(row_points)

    return grid


def click_event(event, x, y, flags, params):
    display_img, original_img, scale_x, scale_y, corners = params

    if event == cv.EVENT_LBUTTONDOWN and len(corners) < 4:
        # Convert back to original image coordinates
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        corners.append((orig_x, orig_y))

        cv.putText(display_img, f"{orig_x},{orig_y}",
                   (x, y),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 0, 0), 2)

        cv.imshow('image', display_img)


def find_corners_manually(img, pattern, max_width=600):
    h, w = img.shape[:2]

    # --- Resize for display if too large ---
    scale = 1.0
    if w > max_width:
        scale = max_width / w

    display_img = cv.resize(img, None, fx=scale, fy=scale)

    scale_x = w / display_img.shape[1]
    scale_y = h / display_img.shape[0]

    corners = []

    cv.imshow('image', display_img)
    cv.setMouseCallback(
        'image',
        click_event,
        param=(display_img, img, scale_x, scale_y, corners)
    )

    cv.waitKey(0)
    cv.destroyAllWindows()

    if len(corners) != 4:
        raise ValueError("You must click exactly 4 corner points.")

    rows = pattern[0]
    cols = pattern[1]
    grid_points = bilinear_interpolation(
        corners[0],
        corners[1],
        corners[2],
        corners[3],
        rows,
        cols
    )

    flat_points = [
        grid_points[j][i]
        for i in range(cols)
        for j in reversed(range(rows))
    ]

    # Convert to OpenCV format: (N,1,2) float32
    corners_array = np.array(flat_points, dtype=np.float32)
    corners_array = corners_array.reshape(-1, 1, 2)

    return corners_array


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('images/fail/*.jpg')

failed_images = []

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if not ret:
        print('Failed to detect corners automatically')
        corners = find_corners_manually(gray, (9, 6))
    else:
        print('Automatic detection successful')

    # print('Corners:', corners)
    # print(corners.shape)

    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (9, 6), corners2, ret)
    scale = 0.1
    display_img = cv.resize(img, None, fx=scale, fy=scale)
    cv.imshow('img', display_img)
    cv.waitKey(0)


cv.destroyAllWindows()

# 1. Perform the actual calibration
# This function calculates the Intrinsic Matrix, Distortion Coefficients, etc.
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 2. Print the results
print("Camera matrix (Intrinsic Parameters):")
print(mtx)

print("\nDistortion coefficients:")
print(dist)

# 3. Save the results for future use (optional but recommended)
np.savez('F.npz', objpoints=objpoints, imgpoints=imgpoints)
