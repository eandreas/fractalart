# mandelbrot_app.py

import streamlit as st
from fractalart.core import *
from fractalart.fractal import *
import matplotlib.pyplot as plt

# Streamlit interface
st.title("Mandelbrot-Explorer")
st.sidebar.header("Parameters")

res = st.sidebar.slider("resolutiomn", 100, 8000, 800, step=100)
max_iter = st.sidebar.slider("max. iterations", 100, 3000, 1000, step=100)
st.sidebar.text("center position")
c_x = st.sidebar.number_input("x-coordinate", format="%0.10f", value = -0.5)
c_y = st.sidebar.number_input("y-coordinate", format="%0.10f", value = 0.0)
zoom = st.sidebar.select_slider("Zoom factor", options = [1, 5, 25, 125, 625, 3125, 15625, 78125], value = 1.0)
hist_eq_on = st.sidebar.toggle("Histogram eualization", value=True)


# Parameters
width, height = res, res
delta = 2 / zoom
center = c_x, c_y
x_min, y_min = center[0] - delta, center[1] - delta
x_max, y_max = center[0] + delta, center[1] + delta

# Render the fractal
m = Mandelbrot()
m.set_zoom(zoom, (c_x, c_y))
m.render()

# Equalize smooth values
if hist_eq_on:
    m.equalize_histogram()

# Plotting
plt.figure()
plt.imshow(m._image, extent=(x_min, x_max, y_min, y_max), cmap='turbo')
plt.axis('off')
st.pyplot(plt)


