"""Camera geometry for vision grounding: converting between pixel
coordinates in a rendered frame and 3D world points.

Used to let Gemini locate objects from the camera image alone (open-
vocabulary grounding) instead of being handed oracle simulator coordinates.
Assumes objects rest on a known, flat horizontal plane (the tabletop) --
the standard simplification single-camera tabletop grounding relies on,
since a single 2D point alone is otherwise depth-ambiguous.
"""

from __future__ import annotations

import numpy as np


def _intrinsics(model, cam_id: int, width: int, height: int):
    fovy_deg = model.cam_fovy[cam_id]
    fy = height / (2.0 * np.tan(np.radians(fovy_deg) / 2.0))
    fx = fy  # square pixels
    return fx, fy, width / 2.0, height / 2.0


def pixel_to_world(model, data, cam_id: int, u: float, v: float,
                    width: int, height: int, plane_z: float) -> np.ndarray:
    """Back-project pixel (u, v) to the 3D point where its camera ray
    intersects the horizontal plane z = plane_z."""
    fx, fy, cx, cy = _intrinsics(model, cam_id, width, height)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    x = (u - cx) / fx
    y = -(v - cy) / fy
    dir_cam = np.array([x, y, -1.0])
    dir_cam /= np.linalg.norm(dir_cam)
    dir_world = cam_mat @ dir_cam

    if abs(dir_world[2]) < 1e-8:
        raise ValueError("camera ray is parallel to the target plane")
    t = (plane_z - cam_pos[2]) / dir_world[2]
    if t < 0:
        raise ValueError("target plane is behind the camera")
    return cam_pos + t * dir_world


def world_to_pixel(model, data, cam_id: int, point: np.ndarray,
                    width: int, height: int) -> tuple[float, float]:
    """Project a 3D world point to pixel coordinates. Mainly useful for
    testing/visualizing pixel_to_world, not part of the grounding pipeline."""
    fx, fy, cx, cy = _intrinsics(model, cam_id, width, height)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    local = cam_mat.T @ (np.asarray(point, dtype=float) - cam_pos)
    x, y, z = local
    if z >= 0:
        raise ValueError("point is behind the camera")
    u = fx * (x / -z) + cx
    v = cy - fy * (y / -z)
    return u, v
