import numpy as np
import cv2 as cv
import glob
from preprocessing import resize_image


def prepare_object_and_image_points(img):
    """Returns 3D world points of chessboard corners and corresponding 2D image points."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20

    return objp, corners2


def project_points(points, r_vec, t_vec, mtx, dist):
    """Project 3D world points to 2D image points, using camera parameters."""
    projected_points, _ = cv.projectPoints(points, r_vec, t_vec, mtx, dist)
    return np.int32(projected_points).reshape(-1, 2)


def draw_axes(img, mtx, dist, r_vec, t_vec, axis_length=90, line_thickness=20):
    """Draws 3D world axes in the image."""
    axes_points = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]
    ])
    proj_axes_points = project_points(axes_points, r_vec, t_vec, mtx, dist)

    cv.line(img, proj_axes_points[0], proj_axes_points[1], (0, 0, 255), line_thickness)
    cv.line(img, proj_axes_points[0], proj_axes_points[2], (0, 255, 0), line_thickness)
    cv.line(img, proj_axes_points[0], proj_axes_points[3], (255, 0, 0), line_thickness)


def draw_cube(img, mtx, dist, r_vec, t_vec, edge_length=40, color=(120, 0, 120), line_thickness=10):
    """Draws 3D world cube in the image."""
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
    proj_cube_points = project_points(cube_points, r_vec, t_vec, mtx, dist)

    cv.polylines(img, [proj_cube_points[:4]], True, color, line_thickness)
    cv.polylines(img, [proj_cube_points[4:]], True, color, line_thickness)
    for i in range(4):
        cv.line(img, proj_cube_points[i], proj_cube_points[i+4], color, line_thickness)


def draw_polygon(img, mtx, dist, r_vec, t_vec, edge_length=40):
    """
    Colors the top of the cube based on the distance and orientation of the camera,
    places a dot in the center and shows the distance.
    """
    polygon_points = np.float32([
        [0, 0, -edge_length],
        [edge_length, 0, -edge_length],
        [edge_length, edge_length, -edge_length],
        [0, edge_length, -edge_length],
        [edge_length/2, edge_length/2, -edge_length]
    ])
    proj_polygon_points = project_points(polygon_points, r_vec, t_vec, mtx, dist)

    r_matrix, _ = cv.Rodrigues(r_vec)

    center_point_w = polygon_points[4].reshape(3, 1)
    center_point_c = r_matrix @ center_point_w + t_vec
    distance_to_camera = np.linalg.norm(center_point_c) / 1000.0

    normal_world = np.array([0, 0, -1])
    np.reshape(normal_world, (3, 1))
    normal_cam = r_matrix @ normal_world
    normal_cam = normal_cam.flatten()
    cos_theta = abs(np.dot(normal_cam, np.array([0, 0, 1])))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    intensity = 0 if distance_to_camera >= 4 else (4 - distance_to_camera) / 4 * 255
    hue = 0 if theta >= 45 else (45 - theta) / 45 * 179
    saturation = 255

    hsv_polygon_color = np.uint8([[[hue, saturation, intensity]]])
    rgb_polygon_color = cv.cvtColor(hsv_polygon_color, cv.COLOR_HSV2BGR)[0][0]
    rgb_polygon_color = tuple(int(x) for x in rgb_polygon_color)

    cv.fillConvexPoly(img, proj_polygon_points[:4], rgb_polygon_color)
    cv.circle(img, proj_polygon_points[4], 5, (255, 255, 255), 20)
    cv.putText(img, f'{distance_to_camera:.3f}',
               (proj_polygon_points[4][0], proj_polygon_points[4][1]),
               cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)


def run(objp, corners, mtx, dist, img, scale=0.1):
    """
    Draws 3D world axes, cube with colored top based on distance and orientation of camera,
    shows a dot with the distance in the center of the colored top. Shows the image.
    """
    ret, r_vec, t_vec = cv.solvePnP(objp, corners, mtx, dist)

    draw_axes(img, mtx, dist, r_vec, t_vec)
    draw_cube(img, mtx, dist, r_vec, t_vec)
    draw_polygon(img, mtx, dist, r_vec, t_vec)

    display_img = cv.resize(img, None, fx=scale, fy=scale)
    cv.imshow('img', display_img)
    cv.waitKey(0)

    cv.destroyAllWindows()


def main(new_image=True):
    """Loads the test image and performs the online phase for 3 different calibration runs."""
    test_image = glob.glob('images/test_image.jpg')
    camera_params = np.load('final_calibration_results.npz')
    img = cv.imread(test_image[0])
    img = resize_image(img)
    objp, corners = prepare_object_and_image_points(img)
    for i in range(3):
        if new_image:
            img = cv.imread(test_image[0])
            img = resize_image(img)
            objp, corners = prepare_object_and_image_points(img)
        run(objp, corners, camera_params['mtx'][i], camera_params['dist'][i], img)


if __name__ == '__main__':
    main(True)
