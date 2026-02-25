import cv2 as cv
import numpy as np


SHAPE = (8 * 6)  # or  6 * 8 ???


def calibrate_intrinsics(video):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 115

    objpoints = []
    imgpoints = []

    num_frames = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        num_frames += 1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    print('All frames:', num_frames)
    print('Frames with automatically detected chessboard corners:', len(imgpoints))


def main():
    path = 'data/cam1/intrinsics.avi'
    video = cv.VideoCapture(path)
    calibrate_intrinsics(video)


if __name__ == '__main__':
    main()
