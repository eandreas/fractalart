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
    """Generate Newton's fractal."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    fractal = np.zeros(Z.shape, dtype=int)
    root_colors = {}

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z = Z[i, j]
            z_final, iter_count = newtons_method(z, max_iter)
            root = np.round(z_final, decimals=6)

            if root not in root_colors:
                root_colors[root] = len(root_colors) + 1

            fractal[i, j] = root_colors[root]

    return fractal

def main():
    st.title("Newton's Fractal Visualizer")
    st.markdown("Visualize the fractal for Newton's method applied to the function $f(z) = z^3 - 1$.")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    max_iter = st.sidebar.slider("Max Iterations", 1, 100, 30)
    resolution = st.sidebar.slider("Resolution (pixels)", 100, 1000, 500, step=50)
    cmap = st.sidebar.selectbox("Colormap", plt.colormaps(), index=plt.colormaps().index("twilight"))

    # Generate fractal
    fractal = generate_fractal(-2, 2, -2, 2, resolution, resolution, max_iter)

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(fractal, cmap=cmap, extent=(-2, 2, -2, 2))
    ax.set_title("Newton's Fractal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    st.pyplot(fig)

if __name__ == "__main__":
    main()

