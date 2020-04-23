import numpy as np
from .utils import rescale, inputs_to_float


@inputs_to_float
def vndvi(green, red, low=0.0, high=1.0):
    """
    Visible Normalized Difference Vegetation Index

             GREEN - RED
    VNDVI = -------------
             GREEN + RED
    """

    num = green - red
    den = green + red

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def gli(blue, green, red, low=0.0, high=1.0):
    """
    Green Leaf Index

           2 * GREEN - RED - BLUE
    GLI = ------------------------
           2 * GREEN + RED + BLUE
    """

    num = 2 * green - red - blue
    den = 2 * green + red + blue

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def vari(blue, green, red, low=0.0, high=1.0):
    """
    Visible Atmospherically Resistant Index

               GREEN - RED
    VARI = --------------------
            GREEN + RED - BLUE

    Based on ARVI
    """

    num = green - red
    den = green + red - blue

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def vdvi(blue, green, red, low=0.0, high=1.0):
    """
    Visible Difference Vegetation Index

            2 * GREEN - RED - BLUE
    VDVI = ------------------------
            2 * GREEN + RED + BLUE

    """

    num = 2 * green - red - blue
    den = 2 * green + red + blue

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def grvi(green, red, low=0.0, high=1.0):
    """
    Green Red Vegetation Index

            GREEN - RED
    GRVI = -------------
            GREEN + RED

    """

    num = green - red
    den = green + red

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def tgi(blue, green, red, low=0.0, high=1.0):
    """
    Triangular Greenness Index

    TGI = GREEN - 0.39 * RED - 0.61 * BLUE

    """

    vi = green - 0.39 * red - 0.61 * blue

    return rescale(vi, low, high)