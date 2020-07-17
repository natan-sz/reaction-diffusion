#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set Constants
N = 10
DA = 1.0
DB = 0.5
feed = 0.055
k = 0.062
dt = 1

# Initialise Chemical A & B
A = np.ones((N, N))
B = np.zeros((N, N))

# Now let's add a disturbance in the center
N2 = N // 2
radius = r = int(N / 10.0)

A[N2 - r : N2 + r, N2 - r : N2 + r] = 0.50
B[N2 - r : N2 + r, N2 - r : N2 + r] = 0.25

# Find Laplacian
def laplacian(M, i, j):
    L = 0
    L += M[i, j] * -1
    L += M[i - 1, j] * 0.2
    L += M[i + 1, j] * 0.2
    L += M[i, j - 1] * 0.2
    L += M[i, j + 1] * 0.2
    L += M[i - 1, j - 1] * 0.05
    L += M[i + 1, j + 1] * 0.05
    L += M[i + 1, j - 1] * 0.05
    L += M[i - 1, j + 1] * 0.05
    return L


# Update A & B
def update(A, B, DA, DB, feed, k, dt):
    for i in range(A.ndim):
        for j in range(A.ndim):
            diff_A = (
                (DA * laplacian(A, i, j))
                - (A[i, j] * B[i, j] ** 2)
                + (feed * (1 - A[i, j]))
            ) * dt
            A[i, j] += diff_A

            diff_B = (
                (DB * laplacian(B, i, j))
                + (A[i, j] * B[i, j] ** 2)
                - ((k + feed) * B[i, j])
            ) * dt
            B[i, j] += diff_B


while True:
    update(A, B, DA, DB, feed, k, dt)
    plt.imshow(A)
    plt.show()
