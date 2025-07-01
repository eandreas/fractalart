import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numba import njit
from matplotlib.colors import ListedColormap


@njit
def assign_roots(Z, delta, tol, max_roots):
    img = np.zeros(Z.shape, dtype=np.int32)
    roots = np.zeros(max_roots, dtype=np.complex128)
    num_roots = 0
    h, w = Z.shape

    for i in range(h):
        for j in range(w):
            if delta[i, j] < tol and img[i, j] == 0:
                z_val = Z[i, j]
                assigned = False
                for k in range(num_roots):
                    if np.abs(z_val - roots[k]) < tol:
                        img[i, j] = k + 1
                        assigned = True
                        break
                if not assigned and num_roots < max_roots:
                    roots[num_roots] = z_val
                    img[i, j] = num_roots + 1
                    num_roots += 1
    return img, roots[:num_roots]


def newton_fractal(user_func_str, bounds=(-2, 2, -2, 2), resolution=2400, max_iter=1000, tol=1e-6):
    # Step 1: Parse symbolic function and derivative
    z = sp.symbols('z')
    f_sym = sp.sympify(user_func_str)
    f_prime_sym = sp.diff(f_sym, z)

    f = sp.lambdify(z, f_sym, 'numpy')
    f_prime = sp.lambdify(z, f_prime_sym, 'numpy')

    # Step 2: Create complex grid
    x_min, x_max, y_min, y_max = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    img = np.zeros(Z.shape, dtype=np.int32)
    max_roots = 50

    for _ in range(max_iter):
        Z_new = Z - f(Z) / f_prime(Z)
        delta = np.abs(Z_new - Z)
        Z = Z_new

        # Use Numba to assign root indices (this is the hotspot)
        img_update, roots = assign_roots(Z, delta, tol, max_roots)
        img[img == 0] = img_update[img == 0]

        if np.all(delta < tol):
            break

    # Step 4: Plot
    plt.figure(figsize=(8, 8))
    cmap = ListedColormap(plt.cm.hsv(np.linspace(0, 1, len(roots) + 1)))
    plt.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap=cmap, origin='lower')
    plt.title(f"Newton Fractal for $f(z) = {sp.latex(f_sym)}$")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.tight_layout()
    plt.show()


# === Example usage ===
if __name__ == "__main__":
    user_input = input("Enter a function f(z): ")  # e.g., "z**3 - 1"
    newton_fractal(user_input)

