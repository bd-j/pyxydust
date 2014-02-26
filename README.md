pyxydust
===

my first python (and OO) code, for fitting astrophysical dust emission
models from Draine & Li 2007 to image pixels

with inspiration from
- k_correct
- EZGAL

Requirements
- [sedpy](https://github.com/bd-j/sedpy)
- [Draine & Li 2007 models](https://www.astro.princeton.edu/~draine/dust/irem.html)

The Draine & Li models should be downloaded, and turned into a large
FITS binary table file using the provided `dl_struct.pro` IDL code.

The MCMC version is slow and out of date.  Example usage of the grid
based version is given in`demos/example_grid_obj.py`. There is no warranty
