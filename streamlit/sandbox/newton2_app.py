import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def newtons_method(z, max_iter, tol=1e-6):
    """Applies Newton's method to find roots of z^3 - 1 = 0."""
    for i in range(max_iter):
        dz = (z**3 - 1) / (3 * z**2)
        z -= dz
        if np.abs(dz) < tol:
            break
    return z, i

def generate_fractal(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Generate Newton's fractal and iteration counts."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    fractal = np.zeros(Z.shape, dtype=int)
    iteration_counts = np.zeros(Z.shape, dtype=float)
    root_colors = {}

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]
            z_final, iter_count = newtons_method(z, max_iter)
            root = np.round(z_final, decimals=6)

            if root not in root_colors:
                root_colors[root] = len(root_colors) + 1

            fractal[i, j] = root_colors[root]
            iteration_counts[i, j] = iter_count

    return fractal, iteration_counts

def equalize_histogram(image):
    """Apply histogram equalization to a 2D image array."""
    flat = image.flatten()
    hist, bins = np.histogram(flat, bins=512, density=True)
    cdf = hist.cumsum() / hist.sum()  # Normalize to [0,1]
    equalized = np.interp(flat, bins[:-1], cdf).reshape(image.shape)
    return equalized

def main():
    st.title("Newton's Fractal Visualizer")

    # Sidebar controls
    st.sidebar.header("Parameters")
    max_iter = st.sidebar.number_input(
        "Max Iterations", min_value=1, max_value=10000, value=100, step=1
    )
    resolution = st.sidebar.slider("Resolution (pixels)", 100, 1000, 500, step=50)
    cmap = st.sidebar.selectbox("Colormap", plt.colormaps(), index=plt.colormaps().index("twilight"))
    apply_equalization = st.sidebar.checkbox("Apply Histogram Equalization", value=True)

    # Compute fractal and iteration counts
    fractal, iteration_counts = generate_fractal(-2, 2, -2, 2, resolution, resolution, max_iter)

    # Apply equalization if selected
    if apply_equalization:
        image = equalize_histogram(iteration_counts)
    else:
        image = iteration_counts

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, extent=(-2, 2, -2, 2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Newton's Fractal", fontsize=16)
    st.pyplot(fig)

if __name__ == "__main__":
    main()

