import cv2 as cv
import numpy as np

from assg_1.online_phase import draw_axes
from assg_2.create_lookup_table import load_camera_params


def find_visible_voxels(lookup_table, views):
    """Finds a list of voxels that are visible from all camera views."""
    height, width = views[0].shape[:2]
    visible_voxels = []

    for voxel, projections in lookup_table.items():
        visible_from_all = True

        for camera, projected_point in projections.items():
            x, y = projected_point[0]

            if not (0 <= x < width and 0 <= y < height):
                visible_from_all = False
                break

            if views[camera][y, x].all() == 0:
                visible_from_all = False
                break

        if visible_from_all:
            visible_voxels.append(voxel)

    return visible_voxels


def compute_depth_buffers(lookup_table, views):
    """Computes per-pixel depth buffers and visible voxel sets per camera using z-buffering."""
    cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
    cameras = []

    for camera in cam_names:
        cam_params_file = '../data/' + camera + '/config.xml'
        cameras.append(load_camera_params(
            cv.FileStorage(cam_params_file, cv.FILE_STORAGE_READ)
        ))

    height, width = views[0].shape[:2]

    depth_buffers = []
    owner_buffers = []

    for _ in cameras:
        depth_buffers.append(np.full((height, width), np.inf))
        owner_buffers.append(np.full((height, width), None, dtype=object))

    for voxel, projections in lookup_table.items():
        X = np.array(voxel)

        for cam_idx, params in enumerate(cameras):
            mtx, dist, r_vec, t_vec = params

            R, _ = cv.Rodrigues(r_vec)
            X_cam = R @ X.reshape(3, 1) + t_vec
            depth = float(X_cam[2])

            if depth <= 0:
                continue

            x, y = projections[cam_idx][0]
            x = int(round(x))
            y = int(round(y))

            if 0 <= x < width and 0 <= y < height:
                if depth < depth_buffers[cam_idx][y, x]:
                    depth_buffers[cam_idx][y, x] = depth
                    owner_buffers[cam_idx][y, x] = voxel

    # Now build final visible sets properly
    visible_per_camera = []

    for cam_idx in range(len(cameras)):
        visible_voxels = set()

        for y in range(height):
            for x in range(width):
                voxel = owner_buffers[cam_idx][y, x]
                if voxel is not None:
                    visible_voxels.add(voxel)

        visible_per_camera.append(visible_voxels)

    return depth_buffers, visible_per_camera


def color_visible_voxels(visible_voxels, lookup_table, views, visible_per_camera):
    """
    Assigns RGB float colors in range [0,1]
    suitable for OpenGL-style visualization.
    Returns: voxel -> (R, G, B)
    """
    voxel_colors = {}

    height, width = views[0].shape[:2]

    for voxel in visible_voxels:
        projections = lookup_table[voxel]
        colors = []

        for camera, projected_point in projections.items():
            if voxel not in visible_per_camera[camera]:
                continue

            x, y = projected_point[0]

            x = int(round(x))
            y = int(round(y))

            if 0 <= x < width and 0 <= y < height:
                # OpenCV gives BGR uint8
                bgr = views[camera][y, x].astype(np.float32)

                # Convert BGR -> RGB
                rgb = bgr[::-1]

                colors.append(rgb)

        if len(colors) > 0:
            colors = np.array(colors)
            final_rgb = np.median(colors, axis=0) / 255.0
        else:
            final_rgb = (0.0, 0.0, 0.0)  # default for occluded voxels

        voxel_colors[voxel] = tuple(final_rgb)

    return voxel_colors


def voxel_reconstruction(skip_frames=4, base_dir='data/', color_voxels=False):
    """Finds a list of voxels that are visible from all camera views, for a number of frames in a video."""
    file = '/foreground_output/foreground.avi'
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    videos = []
    videos_color = []
    lookup_table = np.load(base_dir + 'lookup_table.npz', allow_pickle=True)['lookup_table'].item()

    for camera in cameras:
        video = cv.VideoCapture(base_dir + camera + file)
        videos.append(video)
        video_color = cv.VideoCapture(base_dir + camera + '/video.avi')
        videos_color.append(video_color)

    visible_voxels = None
    frames = 0
    while True:
        ret = True
        views = []
        views_color = []

        for video in videos:
            ret, frame = video.read()
            if not ret:
                break
            views.append(frame)

        for video in videos_color:
            ret, frame = video.read()
            if not ret:
                break
            views_color.append(frame)

        frames += 1
        if not ret:
            break

        if frames % skip_frames == 0:
            visible_voxels = find_visible_voxels(lookup_table, views)
            if color_voxels:
                _, visible_per_camera = compute_depth_buffers(lookup_table, views)
                voxel_colors = color_visible_voxels(visible_voxels, lookup_table, views_color, visible_per_camera)
                return visible_voxels, voxel_colors

    for video in videos:
        video.release()
    for video in videos_color:
        video.release()

    return visible_voxels


def test_world_origin():
    """Shows the world origin and axes. Used for figuring out which points in 3D world to put in a lookup table."""
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
