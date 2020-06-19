<p align="center">
  <img src="static/images/vegi_raleway_transparent.png" alt="vegi" height="160px">
</p>

---

Python Library for deriving vegetation indexes from satellite or aerial imagery

---

[Full Index List](docs/index_list.md)

---

## Usage

Input arguments follow a convention of increasing wavelength using their full lowercase color as the variable name.

``` python
import vegi as vi

# Normalized Difference Vegetation Index
array = vi.ndvi(red, nir)

# Chlorophyll Vegetation Index
array = vi.cvi(green, red, nir)

# Green Atmospherically Resistant Vegetation Index
array = vi.gari(blue, green, red, nir)

# Soil Adjusted Vegetation Index
array = vi.savi(red, nir, L=0.5)
```

## Dependencies

- `numpy`
- `rasterio`
