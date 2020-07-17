#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# Set Constants
N = 100
DA = 1.0
DB = 0.5
feed = 0.055
k = 0.062

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# Initialise Chemical A & B
A = np.ones((N, N))
B = np.zeros((N, N))

B = 0.2 * np.random.random((N, N))

# Now let's add a disturbance in the center
N2 = N // 2
radius = r = int(N / 10.0)

A[N2 - r : N2 + r, N2 - r : N2 + r] = 0.50
B[N2 - r : N2 + r, N2 - r : N2 + r] = 0.25

# Find Laplacian
def laplacian(M, N):
    L = np.zeros((N, N))
    for i in range(N)[1:-1]:
        for j in range(N)[1:-1]:
            L[i, j] += M[i, j] * -1  # Center Pixel
            L[i - 1, j] += M[i - 1, j] * 0.2  # Left Pixel
            L[i + 1, j] += M[i + 1, j] * 0.2  # Right Pixel
            L[i, j - 1] += M[i, j - 1] * 0.2  # Bottom Pixel
            L[i, j + 1] += M[i, j + 1] * 0.2  # Top Pixel
            L[i - 1, j - 1] += M[i - 1, j - 1] * 0.05  # Bottom Left
            L[i + 1, j + 1] += M[i + 1, j + 1] * 0.05  # Top Right
            L[i + 1, j - 1] += M[i + 1, j - 1] * 0.05  # Bottom Right
            L[i - 1, j + 1] += M[i - 1, j + 1] * 0.05  # Top Left
    return L


# Update A & B
def update(A, B, DA, DB, feed, k, N):

    LA = laplacian(A, N)
    LB = laplacian(B, N)

    diff_A = ((DA * LA) - (A * B ** 2) + (feed * (1 - A))) * 0.1
    diff_B = ((DB * LB) + (A * B ** 2) - ((k + feed) * B)) * 0.1

    A += diff_A
    B += diff_B

    return A, B


def animate(i):
    update(A, B, DA, DB, feed, k, N)
    ax1.clear()
    ax1.imshow(A)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show(
