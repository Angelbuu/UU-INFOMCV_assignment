import numpy as np
import cv2 as cv
import glob
from mouse_click import find_corners_manually

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

test_image = glob.glob('images/success/IMG_5954.jpg')

img = cv.imread(test_image[0])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

# ret, man_corners = find_corners_manually(gray, (9, 6))
#
print('Auto corners:', corners, corners.shape)
# print('Manual corners:', man_corners, man_corners.shape)
# print('Difference:', corners - man_corners)

corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
# man_corners2 = cv.cornerSubPix(gray, man_corners, (11, 11), (-1, -1), criteria)

run1 = np.load('final_calibration_results.npz')
print(run1.files)

cv.drawChessboardCorners(img, (9, 6), corners2, ret)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20

ret, rvec, tvec = cv.solvePnP(objp, corners2, run1['mtx'], run1['dist'])

cv.drawFrameAxes(img, run1['mtx'], run1['dist'], rvec, tvec, 90, 20)

cube_edge_length = 40
cube_color = 128, 0, 128
cube_points = np.float32([
    [0, 0, 0],
    [cube_edge_length, 0, 0],
    [0, cube_edge_length, 0],
    [cube_edge_length, cube_edge_length, 0],
    [0, 0, -cube_edge_length],
    [cube_edge_length, 0, -cube_edge_length],
    [0, cube_edge_length, -cube_edge_length],
    [cube_edge_length, cube_edge_length, -cube_edge_length]
])

projected_cube_points, _ = cv.projectPoints(cube_points, rvec, tvec, run1['mtx'], run1['dist'])
projected_cube_points = np.int32(projected_cube_points).reshape(-1, 2)

cv.line(img, projected_cube_points[0], projected_cube_points[1], cube_color, 10)
cv.line(img, projected_cube_points[0], projected_cube_points[2], cube_color, 10)
cv.line(img, projected_cube_points[0], projected_cube_points[4], cube_color, 10)

cv.line(img, projected_cube_points[1], projected_cube_points[3], cube_color, 10)
cv.line(img, projected_cube_points[1], projected_cube_points[5], cube_color, 10)

cv.line(img, projected_cube_points[2], projected_cube_points[3], cube_color, 10)
cv.line(img, projected_cube_points[2], projected_cube_points[6], cube_color, 10)

cv.line(img, projected_cube_points[3], projected_cube_points[7], cube_color, 10)

cv.line(img, projected_cube_points[4], projected_cube_points[5], cube_color, 10)
cv.line(img, projected_cube_points[4], projected_cube_points[6], cube_color, 10)
cv.line(img, projected_cube_points[5], projected_cube_points[7], cube_color, 10)
cv.line(img, projected_cube_points[6], projected_cube_points[7], cube_color, 10)

scale = 0.2
display_img = cv.resize(img, None, fx=scale, fy=scale)
cv.imshow('img', display_img)
cv.waitKey(0)

cv.destroyAllWindows()
