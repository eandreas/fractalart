import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def newtons_method_numba(z, max_iter, tol=1e-6):
    for i in range(max_iter):
        dz = (z**3 - 1) / (3 * z**2)
        z -= dz
        if abs(dz) < tol:
            return z, i
    return z, max_iter - 1

@njit
def generate_fractal_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    fractal = np.zeros((height, width), dtype=np.int32)
    iteration_counts = np.zeros((height, width), dtype=np.float64)

    # Known roots of z^3=1
    roots = np.array([1+0j, 
                      -0.5 + 0.86602540378j,  # cos(120°)+i*sin(120°)
                      -0.5 - 0.86602540378j]) # cos(240°)+i*sin(240°)

    for i in range(height):
        for j in range(width):
            z = xs[j] + 1j*ys[i]
            z_final, iter_count = newtons_method_numba(z, max_iter)
            iteration_counts[i, j] = iter_count

            # Assign root index based on closest root
            distances = np.abs(roots - z_final)
            root_index = 0
            min_dist = distances[0]
            for k in range(1, 3):
                if distances[k] < min_dist:
                    min_dist = distances[k]
                    root_index = k
            fractal[i, j] = root_index + 1  # +1 to avoid zero (background)
    return fractal, iteration_counts

def equalize_histogram(image):
    flat = image.flatten()
    hist, bins = np.histogram(flat, bins=512, density=True)
    cdf = hist.cumsum() / hist.sum()  # Normalize to [0,1]
    equalized = np.interp(flat, bins[:-1], cdf).reshape(image.shape)
    return equalized

def main():
    st.title("Newton's Fractal Visualizer with Numba")

    # Sidebar controls
    st.sidebar.header("Parameters")
    max_iter = st.sidebar.number_input(
        "Max Iterations", min_value=1, max_value=10000, value=100, step=1
    )
    resolution = st.sidebar.slider("Resolution (pixels)", 100, 1000, 500, step=50)
    cmap = st.sidebar.selectbox("Colormap", plt.colormaps(), index=plt.colormaps().index("twilight"))
    apply_equalization = st.sidebar.checkbox("Apply Histogram Equalization", value=True)

    fractal, iteration_counts = generate_fractal_numba(-2, 2, -2, 2, resolution, resolution, max_iter)

    if apply_equalization:
        image = equalize_histogram(iteration_counts)
    else:
        image = iteration_counts

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, extent=(-2, 2, -2, 2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Newton's Fractal", fontsize=16)
    st.pyplot(fig)

if __name__ == "__main__":
    main()

