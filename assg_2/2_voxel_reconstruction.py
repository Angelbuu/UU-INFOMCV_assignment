import cv2 as cv


def voxel_reconstruction(lookup_table, silhouettes, img_shape):
    height, width = img_shape
    visible_voxels = []

    for voxel, views in lookup_table.items():

        visible_from_all = True

        for camera, projected_point in views.items():

            u, v = projected_point
            print('2d image point:', u, v)

            # Convert to integer pixel coords
            u = int(round(u))
            v = int(round(v))

            # Check if inside image bounds
            if not (0 <= u < width and 0 <= v < height):
                visible_from_all = False
                break

            # Check if pixel is foreground
            if silhouettes[camera][v, u] == 0:
                visible_from_all = False
                break

        if visible_from_all:
            visible_voxels.append(list(voxel))

    return visible_voxels


def main():
    base_dir, file = 'data/', '/video.avi'
    cameras = ['cam1', 'cam2', 'cam3', 'cam4']
    videos = []
    for camera in cameras:
        video = cv.VideoCapture(base_dir + camera + file)
        videos.append(video)

    while True:
        ret = True
        for video in videos:
            ret, frame = video.read()
            if not ret:
                break
            # BACKGROUND SUBTRACTION

        if not ret:
            break
        # visible_voxels = voxel_reconstruction()
        # visualize 3d model


if __name__ == '__main__':
    main()
