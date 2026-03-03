import cv2 as cv
import numpy as np

from assg_1.online_phase import draw_axes
from assg_2.create_lookup_table import load_camera_params


def find_visible_voxels(lookup_table, views):
    height, width = views[0].shape[:2]
    visible_voxels = []

    for voxel, projections in lookup_table.items():

        visible_from_all = True

        for camera, projected_point in projections.items():

            u, v = projected_point[0]

            # Convert to integer pixel coords
            # u = int(round(u))
            # v = int(round(v))

            if not (0 <= u < width and 0 <= v < height):
                visible_from_all = False
                break

            if views[camera][v, u].all() == 0:
                visible_from_all = False
                break

        if visible_from_all:
            visible_voxels.append(voxel)

    return visible_voxels


def voxel_reconstruction(skip_frames=4, base_dir='data/'):
    file = '/foreground_output/foreground.avi'
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    videos = []
    lookup_table = np.load(base_dir + 'lookup_table.npz', allow_pickle=True)['lookup_table'].item()

    for camera in cameras:
        video = cv.VideoCapture(base_dir + camera + file)
        videos.append(video)

    visible_voxels = None
    frames = 0
    while True:
        ret = True
        views = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                break
            views.append(frame)
        frames += 1

        if not ret:
            break

        if frames % skip_frames == 0:
            visible_voxels = find_visible_voxels(lookup_table, views)

    for video in videos:
        video.release()

    print(frames)
    return visible_voxels


def test_world_origin():
    video = cv.VideoCapture('data/cam1/video.avi')
    ret, frame = video.read()
    mtx, dist, r_vec, t_vec = load_camera_params(cv.FileStorage('data/cam1/config.xml', cv.FILE_STORAGE_READ))
    draw_axes(frame, mtx, dist, r_vec, t_vec, 1000, 2)
    cv.imshow('img', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    test_world_origin()
    voxel_reconstruction()
