import glm
import random
import numpy as np
import cv2 as cv
from assg_2.voxel_reconstruction import voxel_reconstruction
from assg_2.create_lookup_table import load_camera_params

block_size = 1.0
SCALE = 20


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    """
    Calls voxel reconstruction to get a list of visible voxels. Scales and converts the voxels to correct
    axis coordinates for the visualization world. Returns the corrected voxels and a color scheme.
    """
    data, colors = [], []
    visible_voxels, voxel_colors = voxel_reconstruction(base_dir='../data/', skip_frames=400, color_voxels=True)

    for voxel in visible_voxels:
        x, y, z = voxel

        world_x = int(x) // SCALE * block_size
        world_y = int(-z) // SCALE * block_size
        world_z = int(y) // SCALE * block_size

        data.append([world_x, world_y, world_z])
        colors.append(voxel_colors[voxel])
    return data, colors


def get_cam_positions():
    """Calculates and scales camera positions for the visualization world."""
    cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
    camera_coords = []

    for camera in cam_names:
        cam_params_file = '../data/' + camera + '/config.xml'
        _, _, r_vec, t_vec = load_camera_params(cv.FileStorage(cam_params_file, cv.FILE_STORAGE_READ))

        r_matrix, _ = cv.Rodrigues(r_vec)
        camera_pos = -r_matrix.T @ t_vec
        camera_pos = camera_pos.flatten().tolist()
        camera_pos = [coord // SCALE for coord in camera_pos]
        camera_coords.append(camera_pos)

    return [[camera_coords[0][0], -camera_coords[0][2], camera_coords[0][1]],
            [camera_coords[1][0], -camera_coords[1][2], camera_coords[1][1]],
            [camera_coords[2][0], -camera_coords[2][2], camera_coords[2][1]],
            [camera_coords[3][0], -camera_coords[3][2], camera_coords[3][1]]], \
        [[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    """Calculates camera orientations for the visualization world."""
    cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
    cam_rotations = []

    for camera in cam_names:
        cam_params_file = '../data/' + camera + '/config.xml'
        _, _, r_vec, t_vec = load_camera_params(cv.FileStorage(cam_params_file, cv.FILE_STORAGE_READ))

        r_matrix, _ = cv.Rodrigues(r_vec)
        r_matrix_world = r_matrix.T

        rotation = glm.mat4(1.0)
        for i in range(3):
            for j in range(3):
                rotation[i][j] = r_matrix_world[i, j]

        cam_rotations.append(rotation)

    return cam_rotations


if __name__ == '__main__':
    print(get_cam_positions()[0])
    print(get_cam_rotation_matrices())
