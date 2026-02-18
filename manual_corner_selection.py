import numpy as np
import cv2 as cv
import glob


def bilinear_interpolation(top_left, top_right, bottom_left, bottom_right, rows, cols):
    """
    Linearly interpolates chessboard corners, given the 4 out-most corners, number of rows
    and number of columns.
    """
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)

    grid = []

    for i in range(rows):
        v = i / (rows - 1)
        row_points = []
        for j in range(cols):
            u = j / (cols - 1)

            point = (
                (1 - u) * (1 - v) * top_left +
                u * (1 - v) * top_right +
                (1 - u) * v * bottom_left +
                u * v * bottom_right
            )

            row_points.append(tuple(point))
        grid.append(row_points)

    return grid


def click_event(event, x, y, flags, params):
    """Displays 4 clicked coordinates and saves them to corners parameter."""
    display_img, original_img, scale_x, scale_y, corners = params

    if event == cv.EVENT_LBUTTONDOWN and len(corners) < 4:
        # Convert back to original image coordinates
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        corners.append((orig_x, orig_y))

        cv.putText(display_img, f"{orig_x},{orig_y}",
                   (x, y),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 0, 0), 2)

        cv.imshow('image', display_img)


def find_corners_manually(img, pattern, max_width=600):
    """
    Shows a resized image (to fit on the screen). Allows to select 4 out-most corner points.
    Linearly interpolates the inside points (given the number of rows and columns in the pattern).
    Transforms the output to be the same as given by the cv function findChessboardCorners.
    """
    h, w = img.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w

    display_img = cv.resize(img, None, fx=scale, fy=scale)

    scale_x = w / display_img.shape[1]
    scale_y = h / display_img.shape[0]

    corners = []

    cv.imshow('image', display_img)
    cv.setMouseCallback(
        'image',
        click_event,
        param=(display_img, img, scale_x, scale_y, corners)
    )

    cv.waitKey(0)
    cv.destroyAllWindows()

    rows = pattern[0]
    cols = pattern[1]
    grid_points = bilinear_interpolation(
        corners[0],
        corners[1],
        corners[2],
        corners[3],
        rows,
        cols
    )

    flat_points = [
        grid_points[j][i]
        for i in range(cols)
        for j in reversed(range(rows))
    ]

    corners_array = np.array(flat_points, dtype=np.float32)
    corners_array = corners_array.reshape(-1, 1, 2)

    return True, corners_array


def main():
    """
    Traverses the 'fail' image folder. For each image, checks if corners are selected automatically.
    If yes, raises an exception. If not prompts manual corner selection and saves the 3D world points
    of the corners with corresponding 2D image points as given by the interface and interpolation.
    """
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 20

    objpoints = []
    imgpoints = []

    images = glob.glob('images/fail/*.jpg')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

        if not ret:
            print('Failed to detect corners automatically')
            ret, corners = find_corners_manually(gray, (9, 6))
        else:
            raise Exception('Automatic detection successful')

        objpoints.append(objp)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, (9, 6), corners2, ret)
        scale = 0.1
        display_img = cv.resize(img, None, fx=scale, fy=scale)
        cv.imshow('img', display_img)
        cv.waitKey(0)

    cv.destroyAllWindows()

    np.savez('F.npz', objpoints=objpoints, imgpoints=imgpoints)


if __name__ == '__main__':
    main()
