import numpy as np
from skimage.measure import marching_cubes
import open3d as o3d
from assg_2.voxel_reconstruction import voxel_reconstruction


def voxels_to_volume(visible_voxels, voxel_size):
    """
    Converts list of voxel world coordinates into a dense 3D binary volume.
    """
    voxels = np.array(visible_voxels)

    # Shift to positive grid indices
    min_coords = voxels.min(axis=0)
    shifted = voxels - min_coords

    # Convert world coords -> grid indices
    indices = (shifted / voxel_size).astype(int)

    # Determine grid size
    max_indices = indices.max(axis=0) + 1

    volume = np.zeros(max_indices, dtype=np.uint8)

    for idx in indices:
        volume[tuple(idx)] = 1

    return volume, min_coords


def extract_mesh_from_volume(volume, voxel_size, min_coords):
    """
    Runs marching cubes and converts vertices back to world coordinates.
    """
    verts, faces, normals, _ = marching_cubes(volume, level=0.5)

    # Convert grid coords back to world coords
    verts = verts * voxel_size + min_coords

    return verts, faces


def visualize_mesh(verts, faces):
    """Builds an Open3D triangle mesh from vertices and faces, then displays it."""
    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    volume, min_coords = voxels_to_volume(voxel_reconstruction(skip_frames=400), voxel_size=20.0)
    verts, faces = extract_mesh_from_volume(volume, voxel_size=20.0, min_coords=min_coords)
    visualize_mesh(verts, faces)
