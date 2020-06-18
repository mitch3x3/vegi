<p align="center">
  <img src="static/images/vegi_raleway_transparent.png" alt="vegi" height="200px">
</p>

# Vegetation Index Library

Python Library for deriving vegetation indexes from satellite or aerial imagery

[Full Index List](docs/index_list.md)

## Information

Input arguments follow a convention of increasing wavelength using their full lowercase color as the variable name.

Examples:

``` python
import vegi as vi

array = vi.ndvi(red, nir)

array = vi.grvi(green, red)

array = vi.gari(blue, green, red, nir)
```
