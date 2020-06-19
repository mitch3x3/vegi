<p align="center">
  <img src="static/images/vegi_raleway_transparent.png" alt="vegi" height="160px">
</p>

---

Python Library for deriving vegetation indexes from satellite or aerial imagery

[Full Index List](docs/index_list.md)

---

## Installation

``` bash
pip install vegi
```

###  Dependencies

- `numpy`
- `rasterio`

## Usage

Input arguments follow a convention of increasing wavelength using their full lowercase color as the variable name.

``` python
import vegi

# Normalized Difference Vegetation Index
array = vegi.ndvi(red, nir)

# Green Atmospherically Resistant Vegetation Index
array = vegi.gari(blue, green, red, nir)

# Soil Adjusted Vegetation Index
array = vegi.savi(red, nir, L=0.5)
```
