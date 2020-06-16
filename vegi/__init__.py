"""
Vegetation Index Library

References:

[1] https://www.harrisgeospatial.com/docs/BroadbandGreenness.html

[2] https://www.indexdatabase.de/db/i.php

[3] https://blog.aerobotics.co/ndre-vs-ndvi-whats-the-difference-4f507662823

[4] https://sentera.com/ndvi-vs-ndre-whats-difference/

[5] https://support.micasense.com/hc/en-us/articles/227837307-An-overview-of-the-available-layers-and-indices-in-ATLAS

[6] https://www.hindawi.com/journals/tswj/2014/142939/

[7] http://www.gisresources.com/ndvi-ndbi-ndwi-ranges-1-1/

"""

import numpy as np
from .utils import rescale, inputs_to_float


def vi(*args, **kwargs):
    """VI wrapper function where first arg is the VI name"""
    vi_name = args[0]
    args = args[1:]
    return globals()[vi_name.lower()](*args, **kwargs)


@inputs_to_float
def ndvi(red, nir, low=-1.0, high=1.0):
    """
    Normalized Difference Vegetation Index

    Calculates NDVI between input RED and NIR 2D ndarray objects.

            NIR - RED
    NDVI = -----------
            NIR + RED

    Parameters
    ----------
    red : ndarray
        2D pixel array (625–740 nm)

    nir : ndarray
        2D pixel array (800-900 nm)

    low : float, optional
        Lower bound for scaling output values (default: -1.0)

    high : float, optional
        Upper bound for scaling output values (default: 1.0)

    Returns
    -------
    ndvi : ndarray
        Output ndarray with range of values from low -> high

    Notes
    -----
    -1.0 -> -0.1: Water
    -0.1 ->  0.1: Barren areas of rock, sand, or snow
     0.2 ->  0.4: Shrub and grassland
     0.4 ->  1.0: Temperate and tropical rainforests

    References: [3]

    TREES / ORCHARDS:

    The per tree NDVI index is an estimation of the greenness and density of
    biomass of the tree. Because the NDVI uses visual red light as one of the
    bands in the calculation, the light can’t penetrate very deep into the
    canopy.

    NDVI allows for a generalized indication of stress levels in your orchard
    and works well for determining water stress, leaf nutrient levels, and
    variation in tree canopy especially in the early to mid-development stages.

    The NDVI index provides insights into the biomass levels of trees and is
    more indicative of health in smaller trees.

    References: [4]

    NDVI correlates with chlorophyll

    The visual-band red content that is used in NDVI is absorbed strongly by
    the top of the plant canopy. Lower levels of crop dont contribute to NDVI.
    This impairs correlation of NDVI to things like leaf area index (LAI).
    This effect increases in plants with more layers of leaves, like tree
    canopies or later-stage corn (or wheat).

    In addition, in some permanent crops, grasses and cereal crops, or during
    later growth stages of certain row crops, chlorophyll content reaches a
    point at which NDVI “saturates” at the maximum NDVI value of 1.0. In these
    scenarios, variability in the crop is hard to detect with NDVI until an
    issue becomes severe enough to drop the NDVI value below saturation, which
    may be at a point at which damage has already occurred.

    References: [5]

    Use cases:
    - plant vigor
    - differences in soil water availability
    - foliar nutrient content (when water is not limiting)
    - yield potential

    As plants become healthier, the intensity of reflectance increases in the
    NIR and decreases in the Red, which is the physical basis for most
    vegetation indices. NDVI values can be a maximum value of 1, with lower
    values indicating lower plant vigor. Therefore, 0.5 typically indicates
    low vigor whereas 0.9 indicates very high vigor. NDVI is also effective for
    distinguishing vegetation from soil. NDVI is recommended when looking for
    differences in above-ground biomass in time or across space. NDVI is most
    effective at portraying variation in canopy density during early and mid
    development stages but tends to lose sensitivity at high levels of canopy
    density.

    Examples
    --------

    """

    num = nir - red
    den = nir + red

    vi = np.divide(num, den)

    return rescale(vi, low, high)


def gndvi(green, nir, low=0.0, high=1.0):
    """
    Green Normalized Difference Vegetation Index

    Identical to ndvi(), uses GREEN channel in place of RED

             NIR - GREEN
    GNDVI = -------------
             NIR + GREEN

    References: [1]

    This index is similar to NDVI except that it measures the green spectrum
    from 540 to 570 nm instead of the red spectrum. This index is more
    sensitive to chlorophyll concentration than NDVI.

    """

    num = nir - green
    den = nir + green

    vi = np.divide(num, den)

    return rescale(vi, low, high)


def ndwi(green, nir, low=0.0, high=1.0):
    """
    Normalized Water Index

    Identical to ndvi(), uses GREEN channel in place of RED

            GREEN - NIR
    NDWI = -------------
            GREEN + NIR

    References: [7]

    Used to extract water bodies.

    The NDWI maximizes reflectance of water by using green band wavelengths and
    minimizes low reflectance of NIR by absorbing maximum of wavelength. As a
    result, water features are enhanced owing to having positive values and
    vegetation and soil are suppressed due to having zero or negative values.

    """

    num = nir - green
    den = nir + green

    vi = np.divide(num, den)

    return rescale(vi, low, high)


def ndre(re, nir, low=0.0, high=1.0):
    """
    Normalized Difference Red Edge Index

    Identical to ndvi(), uses RED EDGE channel in place of RED

            NIR - REDEDGE
    NDRE = ---------------
            NIR + REDEDGE

    re = 715 nm

    References: [3]

    NDRE is a better indicator of vegetation health/vigor than NDVI for mid to
    late season crops that have accumulated high levels of chlorophyll in their
    leaves because red-edge light is more translucent to leaves than red light
    and so it is less likely to be completely absorbed by a canopy.

    TREES / ORCHARDS:

    The RedEdge light of the electromagnetic spectrum isn’t as strongly
    absorbed by chlorophyll pigments as Visual Red light in the leaf, therefore
    the light penetrates deeper into the tree canopy.

    The NDRE index provides a more accurate value on the chlorophyll content of
    a greater area of the tree above and below the surface and therefore works
    well with larger trees.

    References: [4]

    Provides a measurement that is not as strongly absorbed by just the topmost
    layers of leaves. NDRE can give better insight into permanent or later
    stage crops because it’s able to measure further down into the canopy.

    NDRE is also less prone to saturation in the presence of dense vegetation

    References: [5]

    Use cases:
    - leaf chlorophyll content
    - plant vigor
    - stress detection
    - fertilizer demand
    - Nitrogen uptake

    """

    num = nir - re
    den = nir + re

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def evi(blue, red, nir, low=0.0, high=1.0):
    """
    Enhanced Vegetation Index

                  G * (NIR - RED)
    EVI = --------------------------------
           NIR + C1 * RED - C2 * BLUE + L
    """

    G = 2.5  # Gain factor
    C1 = 6.0  # Aerosol resistance coefficient
    C2 = 7.5  # Aerosol resistance coefficient
    L = 1.0  # Canopy background adjustment factor

    num = G * (nir - red)
    den = nir + C1 * red - C2 * blue + L

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def evi2(red, nir, low=0.0, high=1.0):
    """
    Enhanced Vegetation Index 2

              G * (NIR - RED)
    EVI2 = --------------------
            NIR + C1 * RED + L
    """

    G = 2.5  # Gain factor
    C1 = 2.4  # Aerosol resistance coefficient
    L = 1.0  # Canopy background adjustment factor

    num = G * (nir - red)
    den = nir + C1 * red + L

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def endvi(blue, green, nir, low=0.0, high=1.0):
    """
    Enchanced Normalized Difference Vegetation Index

             NIR + GREEN - 2 * BLUE
    ENDVI = ------------------------
             NIR + GREEN + 2 * BLUE
    """

    num = nir + green - 2 * blue
    den = nir + green + 2 * blue

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def gci(green, nir, low=0.0, high=1.0):
    """
    Green Chlorophyll Index

            NIR            NIR - GREEN
    GCI = ------- - 1  =  -------------
           GREEN              GREEN

    References: [1]

    This index is used to estimate leaf chlorophyll content across a wide range
    of plant species.

    Having broad NIR and green wavelengths provides a better prediction of
    chlorophyll content while allowing for more sensitivity and a higher
    signal-to-noise ratio.


    """

    vi = np.divide(nir - green, green)

    return rescale(vi, low, high)


@inputs_to_float
def arvi(blue, red, nir, low=0.0, high=1.0):
    """
    Atmospherically Resistant Vegetation Index

            NIR - 2 * RED + BLUE
    ARVI = ----------------------
            NIR + 2 * RED + BLUE
    """

    num = nir - 2 * red + blue
    den = nir + 2 * red + blue

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def savi(red, nir, L=0.5, low=0.0, high=1.0):
    """
    Soil Adjusted Vegetation Index

            (1 + L) * (NIR - RED)
    SAVI = -----------------------
                NIR + RED + L
    """

    # L = Canopy background adjustment factor

    num = (1 + L) * (nir - red)
    den = nir + red + L

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def osavi(red, nir, low=0.0, high=1.0):
    """
    Optimized Soil Adjusted Vegetation Index

             (1 + 0.16) * (NIR - RED)
    OSAVI = --------------------------
                 NIR + RED + 0.16

    Identical to savi(), where L = 0.16

    References: [5]

    - differentiate soil pixels
    - related to LAI at some levels where NDVI saturates
    - accounts for non-linear interactions of light between soil and vegetation
    - used as a structural index for some combined indices designed for
      chlorophyll detection

    OSAVI maps variability in canopy density. In addition, it is not sensitive
    to soil brightness (when different soil types are present). It is robust to
    variability in soil brightness and has enhanced sensitivity to vegetation
    cover greater than 50%. This index is best used in areas with relatively
    sparse vegetation where soil is visible through the canopy and where NDVI
    saturates (high plant density).

    OSAVI is a special case of the Soil Adjusted Vegetation Index (SAVI). OSAVI
    was developed by Rondeaux et al. in 1996 using the reflectance in the
    near-infrared (nir) and red (r) bands with an optimized soil adjustment
    coefficient. The soil adjustment coefficient (0.16) was selected as the
    optimal value to minimize NDVI's sensitivity to variation in soil
    background under a wide range of environmental conditions. OSAVI is a
    hybrid between ratio-based indices such as NDVI and orthogonal indices such
    as PVI. SAVI has a default soil-adjustment factor of 0.5; however, it is
    recommended to use 0.16 as implemented in OSAVI. Like any normalized
    difference index, OSAVI values can range from -1 to 1. High OSAVI values
    indicate denser, healthier vegetation whereas lower values indicate less
    vigor.

    Outputs similar to NDVI -1.0 -> 1.0
    """

    return savi(red, nir, L=0.16)


@inputs_to_float
def gosavi(green, nir, L=0.16, low=0.0, high=1.0):
    """
    Green Optimized Soil Adjusted Vegetation Index

                  NIR - GREEN
    GOSAVI = --------------------
              NIR + GREEN + 0.16

    """

    num = nir - green
    den = nir + green + L

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def msavi2(red, nir, low=0.0, high=1.0):
    """
    Modified Soil Adjusted Vegetation Index 2

              (2 * NIR + 1) - sqrt[ (2 * NIR + 1)**2 - 8 * (NIR - RED) ]
    MSAVI2 = ------------------------------------------------------------
                                        2
    """

    S1 = 2 * nir + 1.0
    S2 = S1 ** 2
    S3 = 8.0 * (nir - red)

    num = np.subtract(S1, np.sqrt(S2 - S3))
    den = 2.0

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def tsavi(red, nir, m=0.5, b=0.0, low=0.0, high=1.0, **kwargs):
    """
    Transformed Soil Adjusted Vegetation Index

                     m * (NIR - m * RED - b)
    TSAVI = -----------------------------------------
             RED + m * (NIR - b) + 0.08 * (1 + m**2)

    m = slope
    b = intercept
    """

    S1 = nir - m * red - b
    S2 = m * (nir - b)
    S3 = 0.08 * (1.0 + m ** 2)

    num = m * S1
    den = red + S2 + S3

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def gari(blue, green, red, nir, y=1.7, low=0.0, high=1.0):
    """
    Green Atmospherically Resistant Vegetation Index

            NIR - [GREEN - y * (BLUE - RED)]
    GARI = ----------------------------------
            NIR + [GREEN - y * (BLUE - RED)]

    y = gamma [default: 1.7]

    The gamma constant is a weighting function that depends on aerosol
    conditions in the atmosphere. ENVI uses a value of 1.7, which is the
    recommended value from Gitelson, Kaufman, and Merzylak (1996, page 296)

    This index is more sensitive to a wide range of chlorophyll concentrations
    and less sensitive to atmospheric effects than NDVI.
    """

    S1 = y * (blue - red)
    S2 = green - S1

    num = nir - S2
    den = nir + S2

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def tdvi(red, nir, low=0.0, high=1.0):
    """
    Transformed Difference Vegetation Index

                1.5 * (NIR - RED)
    TDVI = ----------------------------
            sqrt( NIR**2 + RED + 0.5 )
    """

    S1 = nir - red
    S2 = nir**2 + red + 0.5

    num = 1.5 * S1
    den = np.sqrt(S2)

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def cvi(green, red, nir, low=0.0, high=1.0):
    """
    Chlorophyll Vegetation Index

           NIR * RED
    CVI = -----------
           GREEN**2

    References: [5]

    - detect chlorotic crops
    - stress detection
    - identify vigorous, healthy crops
    - estimate chlorophyll content
    - estimate N content if you know that N is limiting

    """

    num = nir * red
    den = green ** 2

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def mtvi2(green, red, nir, low=0.0, high=1.0):
    """
    Modified Triangular Vegetation Index 2

                1.5 * (2.5 * (NIR - GREEN) - 2.5 * (RED - GREEN))
    MTVI2 = ----------------------------------------------------------
             sqrt( (2 * NIR + 1)**2 - 6 * NIR - 5 * sqrt(RED) - 0.5 )
    """

    S1 = 2.5 * (nir - green)
    S2 = 2.5 * (red - green)
    S3 = np.square(2.0 * nir + 1.0)
    S4 = 6.0 * nir
    S5 = 5.0 * np.sqrt(red)

    num = 1.5 * (S1 - S2)
    den = np.sqrt(S3 - S4 - S5 - 0.5)

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def sipi(blue, red, nir, low=0.0, high=1.0):
    """
    Structure Insensitive Pigment Index

            NIR - BLUE
    SIPI = ------------
            NIR - RED
    """

    num = nir - blue
    den = nir - red

    vi = np.divide(num, den)

    return rescale(vi, low, high)


@inputs_to_float
def sr(red, nir, low=0.0, high=1.0):
    """
    Simple Ratio (SR) or Ratio Vegetation Index (RVI)

          NIR
    SR = -----
          RED
    """

    vi = np.divide(nir, red)

    return rescale(vi, low, high)


@inputs_to_float
def tvx(red, nir, t, low=0.0, high=1.0):
    """
    Temperature Vegetation Index

           LST
    TVX = ------
           NDVI

    LST = Land Surface Temperature

    References: [6]

    """
    # TODO: Enable ability to scale thermal values and convert si_units
    # t = rescale(t, low, high)

    vi = np.divide(t, ndvi(red, nir))

    return rescale(vi, low, high)


@inputs_to_float
def ipvi(red, nir, low=0.0, high=1.0):
    """
    Infrared Percentage Ratio Index

               NIR
    IPVI = -----------
            NIR + RED
    """

    vi = np.divide(nir, red)

    return rescale(vi, low, high)


@inputs_to_float
def dvi(red, nir, low=0.0, high=1.0):
    """
    Difference Vegetation Index

    DVI = NIR - RED
    """

    vi = nir - red

    return rescale(vi, low, high)


@inputs_to_float
def grndvi(green, red, nir, low=0.0, high=1.0):
    """
    Green Red Normalized Vegetation Index

              NIR - (GREEN + RED)
    GRNDVI = ---------------------
              NIR + (GREEN + RED)
    """

    num = nir - (green + red)
    den = nir + (green + red)

    vi = np.divide(num, den)

    return rescale(vi, low, high)


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


def pansharpen(r, g, b, pan, method='browley', W=0.1):
    """
    Produces pansharpened image using an input 3 channel image and a high
    resolution 1 channel image used for sharpening

    Parameters
    ----------
    r : ndarray
        2D image of red channel (625–740 nm)
    g : ndarray
        2D image of green channel (495-570 nm)
    b : ndarray
        2D image of blue channel (400-480 nm)
    pan : ndarray
        2D image of pan channel (Ideally: 400-740 nm)
    method : str, optional
        Pansharpening method [default: 'browley']
    W : float, optional
        Upper bound for scaling NDVI values to.

    Returns
    -------
    r : ndarray
        Pansharpened 2D image of red channel (625–740 nm)
    g : ndarray
        Pansharpened 2D image of green channel (495-570 nm)
    b : ndarray
        Pansharpened 2D image of blue channel (400-480 nm)

    Notes
    -----
    Available methods: ['browley', 'simple_browley', 'sample_mean', 'esri']

    Examples
    --------

    """

    if method == 'simple_browley':
        den = pan / (r + g + b)

        r = np.multiply(r, den)
        g = np.multiply(g, den)
        b = np.multiply(b, den)

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
