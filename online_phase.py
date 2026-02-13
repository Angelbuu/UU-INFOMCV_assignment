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


def project_points(points, rvec, tvec, mtx, dist):
    projected_points, _ = cv.projectPoints(points, rvec, tvec, mtx, dist)
    return np.int32(projected_points).reshape(-1, 2)


def draw_axes(img, mtx, dist, rvec, tvec, axis_length=90, line_thickness=20):
    axes_points = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]
    ])
    proj_axes_points = project_points(axes_points, rvec, tvec, mtx, dist)

    img = cv.line(img, proj_axes_points[0], proj_axes_points[1], (0, 0, 255), line_thickness)
    img = cv.line(img, proj_axes_points[0], proj_axes_points[2], (0, 255, 0), line_thickness)
    img = cv.line(img, proj_axes_points[0], proj_axes_points[3], (255, 0, 0), line_thickness)


def draw_cube(img, mtx, dist, rvec, tvec, edge_length=40, color=(120, 0, 120), line_thickness=10):
    cube_points = np.float32([
        [0, 0, 0],
        [edge_length, 0, 0],
        [edge_length, edge_length, 0],
        [0, edge_length, 0],
        [0, 0, -edge_length],
        [edge_length, 0, -edge_length],
        [edge_length, edge_length, -edge_length],
        [0, edge_length, -edge_length],
    ])
    proj_cube_points = project_points(cube_points, rvec, tvec, mtx, dist)

    cv.polylines(img, [proj_cube_points[:4]], True, color, line_thickness)
    cv.polylines(img, [proj_cube_points[4:]], True, color, line_thickness)
    for i in range(4):
        cv.line(img, proj_cube_points[i], proj_cube_points[i+4], color, line_thickness)


def draw_polygon(img, mtx, dist, rvec, tvec, edge_length=40):
    polygon_points = np.float32([
        [0, 0, -edge_length],
        [edge_length, 0, -edge_length],
        [edge_length, edge_length, -edge_length],
        [0, edge_length, -edge_length],
        [edge_length/2, edge_length/2, -edge_length]
    ])
    proj_polygon_points = project_points(polygon_points[:4], rvec, tvec, mtx, dist)

    color = (0, 0, 0)
    cv.fillConvexPoly(img, proj_polygon_points, color)


draw_axes(img, run1['mtx'], run1['dist'], rvec, tvec)
draw_cube(img, run1['mtx'], run1['dist'], rvec, tvec)
draw_polygon(img, run1['mtx'], run1['dist'], rvec, tvec)

scale = 0.2
display_img = cv.resize(img, None, fx=scale, fy=scale)
cv.imshow('img', display_img)
cv.waitKey(0)

cv.destroyAllWindows()
