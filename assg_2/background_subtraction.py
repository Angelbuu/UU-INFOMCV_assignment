import argparse
import cv2 as cv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

THRESH_H, THRESH_S, THRESH_V = 10, 40, 40
KERNEL = (5, 5)


def build_bg_model(path, method='median', step=3):
    """Average or median over frames from background.avi yp build a background model."""
    cap = cv.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(cv.cvtColor(f, cv.COLOR_BGR2HSV).astype(np.float32))
        i += 1
    cap.release()
    stack = np.stack(frames, axis=0)
    if method == 'median':
        return np.median(stack, axis=0).astype(np.uint8)
    return np.mean(stack, axis=0).astype(np.uint8)


def h_diff(a, b):
    """Calculates difference between hues. Hue wraps at 180."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.minimum(d, 180 - d)


def get_fg_mask(frame_hsv, bg_hsv, th, ts, tv, combine='or'):
    """Threshold HSV differences. OR = fg if any channel differs."""
    hd = h_diff(frame_hsv[:, :, 0], bg_hsv[:, :, 0])
    sd = np.abs(frame_hsv[:, :, 1].astype(np.int16) - bg_hsv[:, :, 1].astype(np.int16))
    vd = np.abs(frame_hsv[:, :, 2].astype(np.int16) - bg_hsv[:, :, 2].astype(np.int16))
    fh = (hd > th).astype(np.uint8)
    fs = (sd > ts).astype(np.uint8)
    fv = (vd > tv).astype(np.uint8)
    if combine == 'or':
        return np.clip(fh + fs + fv, 0, 1).astype(np.uint8) * 255
    return (fh & fs & fv).astype(np.uint8) * 255


def cleanup_mask(mask):
    """Performs erosion and dilation."""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, KERNEL)
    mask = cv.erode(mask, k, iterations=1)
    return cv.dilate(mask, k, iterations=2)


def count_noise(mask):
    """Small blobs = noise. Used for auto threshold search."""
    _, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if len(stats) <= 1:
        return 0
    areas = stats[1:, cv.CC_STAT_AREA]
    return np.sum(areas < 50) + 0.5 * np.sum(areas < 100)


def find_thresh_noise(bg_hsv, frame_hsv):
    """Choice task: pick thresholds that minimize small blobs."""
    best, best_t = np.inf, (THRESH_H, THRESH_S, THRESH_V)
    for th in range(5, 25, 5):
        for ts in range(20, 60, 10):
            for tv in range(20, 60, 10):
                m = cleanup_mask(get_fg_mask(frame_hsv, bg_hsv, th, ts, tv))
                s = count_noise(m)
                if s < best:
                    best, best_t = s, (th, ts, tv)
    return best_t


def run_camera(cam_id, auto_thresh=False, show=True, save=True):
    """
    Runs background subtraction for one camera: builds a background model, extracts foreground,
    and saves to foreground_output/.
    """
    cam_dir = os.path.join(DATA_DIR, cam_id)
    bg_path = os.path.join(cam_dir, 'background.avi')
    vid_path = os.path.join(cam_dir, 'video.avi')

    print(f"\n{cam_id}")
    bg = build_bg_model(bg_path, method='median', step=3)
    # get thresholds
    if auto_thresh:
        cap = cv.VideoCapture(vid_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            fh = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            th, ts, tv = find_thresh_noise(bg, fh)
            print("thresh:", th, ts, tv)
        else:
            th, ts, tv = THRESH_H, THRESH_S, THRESH_V
    else:
        th, ts, tv = THRESH_H, THRESH_S, THRESH_V
    # process video
    cap = cv.VideoCapture(vid_path)
    fps = cap.get(cv.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = None
    if save:
        out_dir = os.path.join(cam_dir, 'foreground_output')
        os.makedirs(out_dir, exist_ok=True)
        out = cv.VideoWriter(os.path.join(out_dir, 'foreground.avi'),
                             cv.VideoWriter_fourcc(*'XVID'), fps, (w, h), False)
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fh = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cleanup_mask(get_fg_mask(fh, bg, th, ts, tv))
        if out:
            out.write(mask)
        if show:
            cv.imshow(f'{cam_id} fg', mask)
            cv.imshow(f'{cam_id} overlay', cv.addWeighted(frame, 0.7, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), 0.3, 0))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        n += 1
    cap.release()
    if out:
        out.release()
        print(f"  saved {n} frames")
    if show:
        cv.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--auto-threshold', action='store_true')
    p.add_argument('--cameras', nargs='+', default=['cam1', 'cam2', 'cam3', 'cam4'])
    a = p.parse_args()
    for c in a.cameras:
        run_camera(c, auto_thresh=a.auto_threshold, show=not a.no_display, save=True)
