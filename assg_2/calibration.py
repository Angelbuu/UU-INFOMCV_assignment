import cv2 as cv
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from assg_1.calibrate import run_calibration_experiment
from assg_1.manual_corner_selection import find_corners_manually
from assg_1.online_phase import draw_axes


SQUARE_SIZE = 115
BOARD_SHAPE = (8, 6)


def is_frame_quality_ok(gray, corners, pattern):
    """
    Choice Task 7: Reject low-quality frames beyond findChessboardCorners.
    Checks: (1) image sharpness, (2) corner spacing uniformity.
    """
    # 1. Sharpness: Laplacian variance - blurry images have low variance
    lap = cv.Laplacian(gray, cv.CV_64F)
    sharpness = lap.var()
    if sharpness < 100:
        return False, "blurry"
    # 2. Corner spacing: adjacent corners should have similar distances
    pts = corners.reshape(-1, 2)
    nc, nr = pattern[0], pattern[1]
    dists = []
    for i in range(nr):
        for j in range(nc - 1):
            a, b = pts[i * nc + j], pts[i * nc + j + 1]
            dists.append(np.linalg.norm(a - b))
    for i in range(nr - 1):
        for j in range(nc):
            a, b = pts[i * nc + j], pts[(i + 1) * nc + j]
            dists.append(np.linalg.norm(a - b))
    dists = np.array(dists)
    # Reject if spacing too irregular (extreme angle or distortion)
    if np.std(dists) / (np.mean(dists) + 1e-6) > 0.35:
        return False, "bad_spacing"
    return True, "ok"


def prepare_object_points():
    """Prepares object points (chessboard corners) as coordinates in 3D world."""
    objp = np.zeros((BOARD_SHAPE[0] * BOARD_SHAPE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SHAPE[0], 0:BOARD_SHAPE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    return objp


def calculate_intrinsics(video, video_num, skip_frames=50, use_quality_filter=True):
    """Calculates camera intrinsics from a number of video frames."""
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = prepare_object_points()

    objpoints = []
    imgpoints = []

    ret, frame = video.read()
    if not ret:
        return None, None

    h, w = frame.shape[:2]
    img_size = (w, h)
    num_frames = 1

    while True:
        ret, frame = video.read()
        if not ret:
            break

        num_frames += 1
        if num_frames % skip_frames != 0:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners_found, corners = cv.findChessboardCorners(gray, (BOARD_SHAPE[0], BOARD_SHAPE[1]), None)

        if corners_found:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if use_quality_filter:
                ok, _ = is_frame_quality_ok(gray, corners2, (BOARD_SHAPE[0], BOARD_SHAPE[1]))
                if not ok:
                    continue
            objpoints.append(objp)
            imgpoints.append(corners2)

    print('All frames:', num_frames)
    print('Frames to be used for calibration:', len(imgpoints))
    if len(imgpoints) == 0:
        return None, None

    return run_calibration_experiment(objpoints, imgpoints, img_size, f'Video {video_num}', reject_bad_images=False)


def calculate_extrinsics(video, mtx, dist, batch=False, fallback_video=None):
    """
    Calculates camera extrinsics based on (frame of a) video and previously calculated intrinsics.
    Also draws 3D world axes in the frame.
    batch: skip interactive windows; uses cv.findChessboardCorners for non-interactive.
    """
    objp = prepare_object_points()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern = (BOARD_SHAPE[0], BOARD_SHAPE[1])
    found, corners, frame = False, None, None

    def try_detect(cap):
        for _ in range(100):
            ret, f = cap.read()
            if not ret:
                return False, None, None
            gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
            ok, c = cv.findChessboardCorners(gray, pattern, None)
            if ok:
                c = cv.cornerSubPix(gray, c, (11, 11), (-1, -1), criteria)
                return True, c, f
        return False, None, None

    if batch:
        found, corners, frame = try_detect(video)
        if not found and fallback_video:
            cap = cv.VideoCapture(fallback_video)
            found, corners, frame = try_detect(cap)
            cap.release()
        if not found:
            raise RuntimeError("No chessboard found; run without --batch for manual selection")
    else:
        video.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, frame = video.read()
        _, corners = find_corners_manually(frame, pattern, 3000)
    ret, r_vec, t_vec = cv.solvePnP(objp, corners, mtx, dist)

    if not batch:
        draw_axes(frame, mtx, dist, r_vec, t_vec, 400, 2)
        cv.imshow('img', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return r_vec, t_vec


def save_calibration(camera_id, mtx, dist, r_vec, t_vec):
    """Saves camera intrinsic and extrinsic parameters to an .xml file"""
    path = 'data/' + camera_id + '/config.xml'
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', mtx)
    fs.write('distortion_coefficients', dist)
    fs.write('rotation_vector', r_vec)
    fs.write('translation_vector', t_vec)
    fs.release()
    print(f"Calibration saved to {path}")


def main():
    """Calculates and saves intrinsic and extrinsic parameters for four cameras."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--batch', action='store_true', help='No windows; use intrinsics.avi if checkerboard fails')
    p.add_argument('--no-quality-filter', action='store_true', help='Disable Choice Task 7 frame quality filter')
    a = p.parse_args()

    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    for camera in cameras:
        print(f'\nCalibrating {camera}')
        cam_dir = 'data/' + camera
        video = cv.VideoCapture(cam_dir + '/intrinsics.avi')

        mtx, dist = calculate_intrinsics(video, camera, skip_frames=50, use_quality_filter=not a.no_quality_filter)
        video.release()
        if mtx is None:
            print(f'Skipping {camera}: no valid calibration frames')
            continue

        video = cv.VideoCapture(cam_dir + '/checkerboard.avi')
        fallback = cam_dir + '/intrinsics.avi'
        r_vec, t_vec = calculate_extrinsics(video, mtx, dist, batch=a.batch, fallback_video=fallback)
        video.release()

        save_calibration(camera, mtx, dist, r_vec, t_vec)


if __name__ == '__main__':
    main()
