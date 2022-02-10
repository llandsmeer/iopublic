# Build the mesh for tube around a piecewise-linear path
#   Adapted from
#   https://github.com/vispy/vispy/blob/main/vispy/visuals/tube.py
# The code over there directly builds the mesh, but we don't want
# to end up with 100000 meshes, 1000 at most.
# So we need to combine them, which is impossible with the old code
# Therefore the adaption

import collections
import numpy as np
from numpy.linalg import norm
from vispy.util.transforms import rotate

def mesh_tube(points, radius=1.0, closed=False, tube_points=8):
    # make sure we are working with floats
    points = np.array(points).astype(float)

    tangents, normals, binormals = _frenet_frames(points, closed)

    segments = len(points) - 1

    # if single radius, convert to list of radii
    if not isinstance(radius, collections.Iterable):
        radius = [radius] * len(points)
    elif len(radius) != len(points):
        raise ValueError('Length of radii list must match points.')

    # get the positions of each vertex
    grid = np.zeros((len(points), tube_points, 3))
    for i in range(len(points)):
        pos = points[i]
        normal = normals[i]
        binormal = binormals[i]
        r = radius[i]

        # Add a vertex for each point on the circle
        v = np.arange(tube_points,
                      dtype=np.float) / tube_points * 2 * np.pi
        cx = -1. * r * np.cos(v)
        cy = r * np.sin(v)
        grid[i] = (pos + cx[:, np.newaxis]*normal +
                   cy[:, np.newaxis]*binormal)

    # construct the mesh
    indices = []
    for i in range(segments):
        for j in range(tube_points):
            ip = (i+1) % segments if closed else i+1
            jp = (j+1) % tube_points

            index_a = i*tube_points + j
            index_b = ip*tube_points + j
            index_c = ip*tube_points + jp
            index_d = i*tube_points + jp

            indices.append([index_a, index_b, index_d])
            indices.append([index_b, index_c, index_d])

    vertices = grid.reshape(grid.shape[0]*grid.shape[1], 3)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def _frenet_frames(points, closed):
    """Calculates and returns the tangents, normals and binormals for
    the tube.
    """
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    if not closed:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    mags = np.sqrt(np.sum(tangents * tangents, axis=1))
    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)

    normals[0] = np.cross(tangents[0], vec)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i-1]

        vec = np.cross(tangents[i-1], tangents[i])
        if norm(vec) > epsilon:
            vec /= norm(vec)
            theta = np.arccos(np.clip(tangents[i-1].dot(tangents[i]), -1, 1))
            normals[i] = rotate(-np.degrees(theta),
                                vec)[:3, :3].dot(normals[i])

    if closed:
        theta = np.arccos(np.clip(normals[0].dot(normals[-1]), -1, 1))
        theta /= len(points) - 1

        if tangents[0].dot(np.cross(normals[0], normals[-1])) > 0:
            theta *= -1.

        for i in range(1, len(points)):
            normals[i] = rotate(-np.degrees(theta*i),
                                tangents[i])[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals
