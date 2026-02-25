import numpy as np
import cv2 as cv
import glob
import os
import shutil


def resize_image(img, standard_size=(4284, 5712)):
    """In case the image is different from the size of other images (standard_size), resize the image."""
    h, w = img.shape[:2]
    if (w, h) != standard_size:
        img = cv.resize(img, standard_size)
    return img


def main():
    """
    Separates the images to ones where findChessboardCorners works (success folder)
    and where it does not (fail folder). For the 'successful' images, saves 3D world corner points and
    the corresponding 2D corner image points found by findChessboardCorners.
    """
    os.makedirs('images/success', exist_ok=True)
    os.makedirs('images/fail', exist_ok=True)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20

    objpoints = []
    imgpoints = []

    images = glob.glob('images/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        img = resize_image(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

        base_name = os.path.basename(fname)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(100)

            dest = os.path.join('images/success', base_name)
            print(f"Success: {base_name}")
        else:
            dest = os.path.join('images/fail', base_name)
            print(f"Failed: {base_name}")

        shutil.move(fname, dest)

    cv.destroyAllWindows()

    if len(objpoints) > 0:
        np.savez('B.npz', objpoints=objpoints, imgpoints=imgpoints)


if __name__ == '__main__':
    main()
