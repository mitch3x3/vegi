import numpy as np


UINT8 = float(2**8 - 1)  # 255.0
UINT14 = float(2**14 - 1)  # 16383.0
UINT16 = float(2**16 - 1)  # 65535.0


def ndvi(red, nir, low=0.0, high=1.0):
    """
    Calculates NDVI between input RED and NIR 2dim ndarray objects.

    Parameters
    ----------
    red : ndarray
        Input grayscale image of RED channel (625–740 nm)
    nir : ndarray
        Input grayscale image of NIR channel (800-900 nm)
    low : float, optional
        Lower bound for scaling NDVI values to.
    high : float, optional
        Upper bound for scaling NDVI values to.

    Returns
    -------
    ndvi : ndarray
        Output ndarray with range of values from -1.0 -> 1.0

    Notes
    -----
    -1.0 -> -0.1: Water
    -0.1 ->  0.1: Barren areas of rock, sand, or snow
     0.2 ->  0.4: Shrub and grassland
     0.4 ->  1.0: Temperate and tropical rainforests

    Examples
    --------

    """

    red = np.asarray(red, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)

    ndvi = (nir - red) / (nir + red)
    ndvi[np.isnan(ndvi)] = 0
    ndvi[np.isinf(ndvi)] = 0

    del red
    del nir

    # Ignore calculation if values will not change result
    if low != 0.0 or high != 1.0:
        ndvi = (ndvi - low) / (high - low)

    ndvi[np.isnan(ndvi)] = 0
    ndvi[np.isinf(ndvi)] = 0

    return ndvi


def pansharpen(r, g, b, pan, method='browley', W=0.1):
    """
    Produces pansharpened image using an input 3 channel image and a high
    resolution 1 channel image used for sharpening

    Parameters
    ----------
    r : ndarray
        Input grayscale image of red channel (625–740 nm)
    g : ndarray
        Input grayscale image of green channel (800-900 nm)
    b : ndarray
        Input grayscale image of blue channel (800-900 nm)
    low : float, optional
        Lower bound for scaling NDVI values to.
    high : float, optional
        Upper bound for scaling NDVI values to.

    Returns
    -------
    ndvi : ndarray
        Output ndarray with range of values from -1.0 -> 1.0

    Notes
    -----
    -1.0 -> -0.1: Water
    -0.1 ->  0.1: Barren areas of rock, sand, or snow
     0.2 ->  0.4: Shrub and grassland
     0.4 ->  1.0: Temperate and tropical rainforests

    Examples
    --------

    """

    if method == 'simple_browley':
        all_in = r + g + b
        # prod = np.multiply(all_in, pan)

        r = np.multiply(r, pan / all_in)
        g = np.multiply(g, pan / all_in)
        b = np.multiply(b, pan / all_in)

    if method == 'sample_mean':
        r = 0.5 * (r + pan)
        g = 0.5 * (g + pan)
        b = 0.5 * (b + pan)

    if method == 'esri':
        rgb = np.empty((r.shape[0], r.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        ADJ = pan - rgb.mean(axis=2)

        r = (r + ADJ)
        g = (g + ADJ)
        b = (b + ADJ)

    if method == 'browley':
        pan /= (W * r + W * g + W * b)

        # Multiply by DNF
        r *= pan
        g *= pan
        b *= pan

    return r, g, b