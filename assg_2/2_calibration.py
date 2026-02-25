import cv2 as cv
import numpy as np
from assg_1.calibrate import run_calibration_experiment


def prepare_object_points():
    # SHAPE = (8 * 6) or 6 * 8 ???
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 115
    return objp


def calculate_intrinsics(video, video_num, skip_frames=10):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = prepare_object_points()

    objpoints = []
    imgpoints = []

    ret, frame = video.read()
    img_size = frame.shape[::-1][1:]
    print('Image size:', img_size)
    num_frames = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break

        num_frames += 1
        if num_frames % skip_frames != 0:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners_found, corners = cv.findChessboardCorners(gray, (8, 6), None)

        if corners_found:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    print('All frames:', num_frames)
    print('Frames to be used for calibration:', len(imgpoints))

    return run_calibration_experiment(objpoints, imgpoints, img_size, f'Video {video_num}', reject_bad_images=False)


def calculate_extrinsics(video, mtx, dist):
    objp = prepare_object_points()
    corners = None
    ret, r_vec, t_vec = cv.solvePnP(objp, corners, mtx, dist)
    return r_vec, t_vec


def main():
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    for camera in cameras:
        print(f'\nCalibrating {camera}')
        path = 'data/' + camera
        video = cv.VideoCapture(path + '/intrinsics.avi')
        mtx, dist = calculate_intrinsics(video, camera, skip_frames=50)

        video = cv.VideoCapture(path + '/checkerboard.avi')
        # r_vec, t_vec = calculate_extrinsics(video, mtx, dist)


if __name__ == '__main__':
    main()
