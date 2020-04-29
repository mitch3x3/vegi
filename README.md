# Vegetation Index Library

Python Library for Vegetation Indexes

[Full Index List](docs/index_list.md)

## Information

Input arguments follow a convention of increasing wavelength using their full lowercase color as the variable name.

Examples:

``` python
import vilib as vi

array = vi.ndvi(red, nir)

array = vi.grvi(green, red)

array = vi.gari(blue, green, red, nir)
```
