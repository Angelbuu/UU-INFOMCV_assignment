import cv2 as cv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from assg_1.calibrate import run_calibration_experiment


SQUARE_SIZE = 115


def calibrate_intrinsics(video, video_num, skip_frames=10):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * SQUARE_SIZE

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

    run_calibration_experiment(objpoints, imgpoints, img_size, f'Video {video_num}', reject_bad_images=False)
    if len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
        print("Camera Matrix:\n", mtx)
        print("Distortion Coeffs:\n", dist)
        return mtx, dist
    
    return None, None

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
        path = f'data/{camera}/intrinsics.avi'
        video = cv.VideoCapture(path)
        mtx, dist = calibrate_intrinsics(video, camera, skip_frames=50)
        
        # Save them if calibration was successful
        if mtx is not None:
            save_calibration(camera, mtx, dist)
            
        video.release()
        



if __name__ == '__main__':
    main()
