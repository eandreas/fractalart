"""An implementation of the Julia set."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/fractals/03_julia.ipynb.

# %% auto 0
__all__ = ['smooth_coloring', 'julia_step']

# %% ../../nbs/fractals/03_julia.ipynb 4
#from abc import abstractmethod
import numpy as np
#from fractalart.core import Image
from numba import njit, prange
#from numba import njit, prange, jit
import math
#import matplotlib.pyplot as plt
from .abstract_fractal import Fractal

# %% ../../nbs/fractals/03_julia.ipynb 5
_inv_log2 = 1.0 / math.log(2.0)

@njit
def smooth_coloring(zr, zi, iteration):
    mag_sq = zr * zr + zi * zi

    # Ensure mag_sq > 1 to avoid log(≤0)
    if mag_sq > 1e-8:
        log_zn = 0.5 * math.log(mag_sq)
        if log_zn > 1e-8:
            nu = math.log(log_zn * _inv_log2) * _inv_log2
            return iteration + 1 - nu

    return float(iteration)
    
@njit
def julia_step(zr, zi, cr, ci):
    zr2 = zr * zr
    zi2 = zi * zi
    zr_new = zr2 - zi2 + cr
    zi_new = 2.0 * zr * zi + ci
    return zr_new, zi_new

# %% ../../nbs/fractals/03_julia.ipynb 6
@njit(parallel=True, fastmath=True)
def _compute_julia(x_min: float, x_max: float, y_min: float, y_max: float, cr: float, ci: float, resolution: tuple[int, int],
                     max_iter: int, fractal_fn, order: int = 1, smooth: bool = True) -> np.ndarray:
    
    width, height = resolution
    result = np.zeros((height, width), dtype=np.float64)
    r2_cut = max(abs(x_max), abs(x_min)) * max(abs(x_max), abs(x_min)) + max(abs(y_max), abs(y_min)) * max(abs(y_max), abs(y_min))

    dx = (x_max - x_min) / (width - 1)
    dy = (y_max - y_min) / (height - 1)
    inv_log2 = 1.0 / math.log(2.0)

    for j in prange(height):
        zy = y_min + j * dy
        for i in range(width):
            zx = x_min + i * dx
            zr = zx
            zi = zy
            cr = cr
            ci = ci
            iteration = 0

            while zr * zr + zi * zi <= r2_cut and iteration < max_iter:
                zr, zi = fractal_fn(zr, zi, cr, ci)
                iteration += 1

            if smooth and iteration < max_iter:
                result[j, i] = smooth_coloring(zr, zi, iteration)
            else:
                result[j, i] = iteration

    return result
