import cv2 as cv
import numpy as np


def voxel_reconstruction(lookup_table, views):
    height, width = views[0].shape[:2]
    visible_voxels = []

    for voxel, projections in lookup_table.items():

        visible_from_all = True

        for camera, projected_point in projections.items():

            u, v = projected_point[0]
            # print('2d image point:', u, v)

            # Convert to integer pixel coords
            u = int(round(u))
            v = int(round(v))

            # Check if inside image bounds
            if not (0 <= u < width and 0 <= v < height):
                visible_from_all = False
                break

            # Check if pixel is foreground
            # print(frames[camera][v, u])
            if views[camera][v, u].all() == 0:
                visible_from_all = False
                break
            else:
                pass
                # print('Foreground')

        if visible_from_all:
            output_voxel = [int(x) // 100 for x in voxel]
            visible_voxels.append(output_voxel)

    return visible_voxels


def main(skip_frames=100):
    base_dir, file = 'data/', '/foreground_output/foreground.avi'
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    videos = []
    lookup_table = np.load('data/lookup_table.npz', allow_pickle=True)['lookup_table'].item()

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
            visible_voxels = voxel_reconstruction(lookup_table, views)
            break
        # print(visible_voxels)
        # visualize 3d model

    for video in videos:
        video.release()

    print(frames)
    return visible_voxels


if __name__ == '__main__':
    visible_voxels = main()
    print(visible_voxels)
