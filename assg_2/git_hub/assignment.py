import glm
import random
import numpy as np
import cv2 as cv
from assg_2.voxel_reconstruction import main
from assg_2.create_lookup_table import load_camera_params

block_size = 1.0


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
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    visible_voxels = main()
    for voxel in visible_voxels:
        x, y, z = voxel

        # Convert grid coordinates to world coordinates
        world_x = x * block_size - width / 2
        world_y = -z * block_size
        world_z = y * block_size - depth / 2

        data.append([world_x, world_y, world_z])

        # Simple coloring scheme (same as before)
        colors.append([
            x / width,
            z / depth,
            y / height
        ])
    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])
    print('Voxels displayed')
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
    camera_coords = []
    for camera in cam_names:
        cam_params_file = '../data/' + camera + '/config.xml'
        _, _, r_vec, t_vec = load_camera_params(cv.FileStorage(cam_params_file, cv.FILE_STORAGE_READ))
        r_matrix, _ = cv.Rodrigues(r_vec)
        camera_pos = -r_matrix.T @ t_vec
        camera_coords.append(camera_pos.flatten().tolist())

    return [camera_coords[0],
            camera_coords[1],
            camera_coords[2],
            camera_coords[3]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]], \
    #     [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations


if __name__ == '__main__':
    print(get_cam_positions()[0])
