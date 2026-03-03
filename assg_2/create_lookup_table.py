import cv2 as cv
import numpy as np
from assg_1.online_phase import project_points


def load_camera_params(file):
    mtx = file.getNode("camera_matrix").mat()
    dist = file.getNode("distortion_coefficients").mat()
    r_vec = file.getNode("rotation_vector").mat()
    t_vec = file.getNode("translation_vector").mat()

    file.release()

    return mtx, dist, r_vec, t_vec


def create_lookup_table(cameras, space_size=2000.0, voxel_size=100.0):
    voxels_x = np.arange(-space_size, space_size, voxel_size)
    print(voxels_x)
    voxels_y = np.arange(-space_size, space_size, voxel_size)
    voxels_z = np.arange(-space_size, space_size, voxel_size)

    lookup_table = {}

    for x in voxels_x:
        for y in voxels_y:
            for z in voxels_z:
                voxel = (round(x, 3), round(y, 3), round(z, 3))
                lookup_table[voxel] = {}

                obj_point = np.array([[x, y, z], ])

                for camera, params in enumerate(cameras):
                    mtx, dist, r_vec, t_vec = params
                    projected_point = project_points(obj_point, r_vec, t_vec, mtx, dist)

                    lookup_table[voxel][camera] = projected_point

    print("Number of voxels:", len(lookup_table))
    print("Example entry:")
    first_key = next(iter(lookup_table))
    print(first_key, "->", lookup_table[first_key])
    np.savez('data/lookup_table.npz', lookup_table=lookup_table)


def main():
    cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
    cameras = []
    for camera in cam_names:
        cam_params_file = 'data/' + camera + '/config.xml'
        cam_params = load_camera_params(cv.FileStorage(cam_params_file, cv.FILE_STORAGE_READ))
        cameras.append(cam_params)
    create_lookup_table(cameras)


if __name__ == '__main__':
    main()
