# mandelbrot_app.py

import streamlit as st
import matplotlib.pyplot as plt
from enum import Enum

from fractalart.core import *
from fractalart.fractal import *

# --- Enum Definition ---
class FractalType(Enum):
    MANDELBROT = "Mandelbrot Set"
    JULIA = "Julia Set"

# --- Factory Function ---
def create_fractal(fractal_type: FractalType) -> Fractal:
    fractal_classes = {
        FractalType.MANDELBROT: Mandelbrot,
        FractalType.JULIA: Julia,
    }
    return fractal_classes[fractal_type]()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Parameters")

    selected_type = st.selectbox(
        "Fractal type",
        options=list(FractalType),
        format_func=lambda ft: ft.value
    )

    resolution = st.slider("Resolution", 100, 4000, 600, step=100)
    # max_iterations = st.slider("Max iterations", 100, 3000, 1000, step=100)
    max_iterations = st.select_slider("Max iterations", options=[10, 100, 200, 500, 1000, 3000, 5000, 10000], value=200)

    st.text("Center position")
    center_x = st.number_input("X-coordinate", format="%.10f", value=-0.5)
    center_y = st.number_input("Y-coordinate", format="%.10f", value=0.0)

    zoom = st.select_slider("Zoom factor", options=[1, 5, 25, 125, 625, 3125, 15625, 78125], value=1)
    hist_eq_enabled = st.toggle("Histogram equalization", value=True)

# --- Fractal Configuration ---
fractal = create_fractal(selected_type)
fractal.resolution = (resolution, resolution)
fractal.max_iter = max_iterations
fractal.set_zoom(zoom, (center_x, center_y))
fractal.render()

# --- Histogram Equalization ---
if hist_eq_enabled:
    fractal.equalize_histogram()

# --- Plotting ---
delta = 2 / zoom
x_min, x_max = center_x - delta, center_x + delta
y_min, y_max = center_y - delta, center_y + delta

fig, ax = plt.subplots()
ax.imshow(fractal._image, extent=(x_min, x_max, y_min, y_max), cmap='turbo')
ax.axis("off")
st.pyplot(fig)
