import numpy as np
import cv2 as cv
from online_phase import draw_axes, draw_cube, draw_polygon


def live_drawing(mtx, dist):
    """Performs the online phase live."""
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

        if ret:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objp = np.zeros((9 * 6, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20

            _, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

            draw_axes(frame, mtx, dist, rvec, tvec)
            draw_cube(frame, mtx, dist, rvec, tvec)
            draw_polygon(frame, mtx, dist, rvec, tvec)

        cv.imshow('INFOMCV Real-Time AR', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def main():
    """Loads calibration results and performs the online phase live."""
    with np.load('final_calibration_results.npz') as data:
        mtx, dist = data['mtx'][0], data['dist'][0]

    live_drawing(mtx, dist)


if __name__ == '__main__':
    main()
