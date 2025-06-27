# mandelbrot_app.py

import streamlit as st
import matplotlib.pyplot as plt
from enum import Enum

from fractalart.core import *
from fractalart.fractal import *

# --- Enum Definition ---
class FractalType(Enum):
    MANDELBROT = "Mandelbrot Set"
    MANDELBROT_CROSS_TRAP = "Mandelbrot Set (Cross Trap)"
    JULIA = "Julia Set"
    JULIA_CROSS_TRAP = "Julia Set (Cross Trap)"

# --- Factory Function ---
#def create_fractal(fractal_type: FractalType) -> Fractal:
#    fractal_classes = {
#        FractalType.MANDELBROT: Mandelbrot,
#        FractalType.MANDELBROT_CROSS_TRAP: MandelbrotCrossTrap,
#        FractalType.JULIA: Julia,
#        FractalType.JULIA_CROSS_TRAP: JuliaCrossTrap
#    }
#    return fractal_classes[fractal_type]()

def create_fractal(
    fractal_type: FractalType,
    x_min: float, x_max: float, y_min: float, y_max: float, c_re: float, c_im: float
) -> Fractal:
    if fractal_type == FractalType.JULIA:
        return Julia(cr = c_re, ci = c_im, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
    elif fractal_type == FractalType.JULIA_CROSS_TRAP:
        return JuliaCrossTrap(cr = c_re, ci = c_im, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
    elif fractal_type == FractalType.MANDELBROT:
        return Mandelbrot(x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
    elif fractal_type == FractalType.MANDELBROT_CROSS_TRAP:
        return MandelbrotCrossTrap(x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)
    else:
        raise ValueError(f"Unsupported fractal type: {fractal_type}")

c_re = 0.0
c_im = 0.0

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Parameters")

    selected_type = st.selectbox(
        "Fractal type",
        options=list(FractalType),
        format_func=lambda ft: ft.value
    )

    match selected_type:
        case FractalType.JULIA | FractalType.JULIA_CROSS_TRAP:
            c_re = st.slider("Re(c)", -1.0, 1.0, 0.0, step=0.001, format="%0.3f")
            c_im = st.slider("Im(c)", -1.0, 1.0, 0.0, step=0.001, format="%0.3f")
            x_min, x_max = -1.5, 1.5
            y_min, y_max = -1.5, 1.5
        case FractalType.MANDELBROT | FractalType.MANDELBROT_CROSS_TRAP:
            x_min, x_max = -2.0, 1
            y_min, y_max = -1.5, 1.5
            
    resolution = st.slider("Resolution", 100, 4000, 600, step=100)
    # max_iterations = st.slider("Max iterations", 100, 3000, 1000, step=100)
    max_iterations = st.select_slider("Max iterations", options=[10, 100, 200, 500, 1000, 3000, 5000, 10000], value=200)

    st.text("Center position")
    center_x = x_min + (x_max - x_min) / 2
    center_y = y_min + (y_max - y_min) / 2

    zoom = st.select_slider("Zoom factor", options=[1, 5, 25, 125, 625, 3125, 15625, 78125], value=1)
    hist_eq_enabled = st.toggle("Histogram equalization", value=True)

# --- Fractal Configuration ---
fractal = create_fractal(selected_type, x_min, x_max, y_min, y_max, c_re, c_im)
fractal.resolution = (resolution, resolution)
fractal.max_iter = max_iterations
fractal.set_zoom(zoom, (center_x, center_y))
fractal.render()

# --- Histogram Equalization ---
if hist_eq_enabled:
    fractal.equalize_histogram()

fig, ax = plt.subplots()
ax.imshow(fractal._image, extent=(x_min, x_max, y_min, y_max), cmap='turbo')
ax.axis("off")
st.pyplot(fig)
