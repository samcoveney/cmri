"""Creation of coordinate systems"""

import numpy as np


def LV_coords(image_shape, LVcent):
    """Uses LV center to define simple coordinate system."""

    [Y, X] = np.meshgrid(range(image_shape[1]), range(image_shape[0]))

    # radial direction
    rad = np.zeros([image_shape[0], image_shape[1], 3])
    rad[:, :, 0] = X - LVcent[0]
    rad[:, :, 1] = Y - LVcent[1]
    rad[:, :, 2] = 0.0
    rad = rad / np.linalg.norm(rad, axis=-1)[..., None]

    lon = np.zeros_like(rad)
    lon[:, :, 2] = +1.0 

    lon = lon / np.linalg.norm(lon, axis=-1)[..., None]
    #lon *= -1  # FIXME: debugging still

    # circular direction - should be clockwise looking from base to apex
    cir = np.cross(rad, lon)
    cir = cir / np.linalg.norm(cir, axis=-1)[..., None]
    #cir *= -1  # FIXME: checking sense of circle

    return rad, lon, cir

