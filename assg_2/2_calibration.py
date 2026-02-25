import cv2 as cv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from assg_1.calibrate import run_calibration_experiment
from assg_1.manual_corner_selection import find_corners_manually
from assg_1.online_phase import draw_axes


SQUARE_SIZE = 115


def prepare_object_points():
    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def calculate_intrinsics(video, video_num, skip_frames=50):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = prepare_object_points()

    objpoints = []
    imgpoints = []

    ret, frame = video.read()
    if not ret:
        return None, None
    h, w = frame.shape[:2]
    img_size = (w, h)
    print('Image size:', img_size)
    num_frames = 0
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

    if len(imgpoints) > 0:
        return run_calibration_experiment(objpoints, imgpoints, img_size, f'Video {video_num}', reject_bad_images=False)
    
    return None, None


def calculate_extrinsics(video, mtx, dist):
    objp = prepare_object_points()

    _, frame = video.read()
    width = frame.shape[1]
    print(width)
    _, corners = find_corners_manually(frame, (8, 6), 3000)

    ret, r_vec, t_vec = cv.solvePnP(objp, corners, mtx, dist)
    print(r_vec, t_vec)
    draw_axes(frame, mtx, dist, r_vec, t_vec, 400, 2)
    cv.imshow('img', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return r_vec, t_vec


def save_calibration(camera_id, mtx, dist):
    file_path = f'data/{camera_id}/intrinsics.xml'
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', mtx)
    fs.write('distortion_coefficients', dist)
    fs.release()
    print(f"Calibration saved to {file_path}")


def main():
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    for camera in cameras:
        print(f'\nCalibrating {camera}')
        path = f'data/{camera}'
        video = cv.VideoCapture(path + '/intrinsics.avi')
        mtx, dist = calculate_intrinsics(video, camera, skip_frames=50)
        
        if mtx is not None:
            save_calibration(camera, mtx, dist)
            
        video.release()

        video = cv.VideoCapture(path + '/checkerboard.avi')
        if video.isOpened() and mtx is not None:
            r_vec, t_vec = calculate_extrinsics(video, mtx, dist)
        video.release()


if __name__ == '__main__':
    main()
