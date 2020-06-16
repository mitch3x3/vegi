import functools
import numpy as np


UINT8 = float(2**8 - 1)  # 255.0
UINT16 = float(2**16 - 1)  # 65535.0


def validate(array, value=0):
    """Sets Infs and NaNs in array to input value [default: 0]"""
    array[np.isnan(array)] = value
    array[np.isinf(array)] = value
    return array


def rescale(array, low, high, bounds=(None, None)):
    """Rescales data to low & high input values, and crops to bound limits"""
    array = validate(array)

    bmin, bmax = bounds

    # Skip rescale if low & high values equal limiting boundaries
    if low != bmin or high != bmax:
        array = (array - low) / (high - low)
        array = validate(array)

    # Crop extreme/outlier values to limiting boundaries
    if bmin is not None and bmax is not None:
        array = crop_to_bounds(array, bmin, bmax)

    return array


def crop_to_bounds(array, low, high):
    """Sets values in array under low and over high to their respective values"""
    array[array < low] = low
    array[array > high] = high
    return array


def inputs_to_float(func):
    """Converts all positional and optional arguments to float32"""
    @functools.wraps(func)
    def inner(*args, **kwargs):
        args = tuple(np.asarray(x, dtype=np.float32) for x in args)
        kwargs = dict((k, np.float32(v)) for (k, v) in kwargs.items())
        return func(*args, **kwargs)

    return inner


def convert_8bit_to_16bit(array):
    """Converts 8bit array to 16bit"""
    array = np.clip(array, 0, UINT8).astype(np.uint8)
    array = np.asarray(array, dtype=np.float)
    array = np.divide(array, UINT8)
    array = np.multiply(array, UINT16)
    array = np.asarray(array, dtype=np.uint16)
    array = np.clip(array, 0, UINT16).astype(np.uint16)
    return array


def convert_16bit_to_8bit(array):
    """Converts 16bit array to 8bit"""
    array = np.clip(array, 0, UINT16).astype(np.uint16)
    array = np.asarray(array, dtype=np.float)
    array = np.divide(array, UINT16)
    array = np.multiply(array, UINT8)
    array = np.asarray(array, dtype=np.uint8)
    array = np.clip(array, 0, UINT8).astype(np.uint8)
    return array
