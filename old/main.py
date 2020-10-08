#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
from scipy import signal

# Set Constants
N = 200
DA = 1.0
DB = 0.5
feed = 0.034
k = 0.060

# Initialise Chemicals A & B
A = np.ones((N, N))
B = np.zeros((N, N))

# Add Square to the middle
N2 = N // 2
r = int(N / 10.0)

A[N2 - r : N2 + r, N2 - r : N2 + r] = 0.50
B[N2 - r : N2 + r, N2 - r : N2 + r] = 0.25

# add scanning kernel
cen = -1
adj = 0.2
cor = 0.05

kernel = np.array([[cor, adj, cor], [adj, cen, adj], [cor, adj, cor]])

# DO A CONVOLUTION USING FFT (outdated)
#
# offset = len(kernel) // 2  # Only Compatible with 1:1 matrices

# Laplacian Function (TO BE FIXED)
# def convolution(M, kernel, offset):
#
#    L = np.zeros((M.shape[0], M.shape[1]))
#
#    for i in range(offset, M.shape[0] - offset):
#        for j in range(offset, M.shape[1] - offset):
#            sum = 0
#            for a in range(len(kernel)):
#                for b in range(len(kernel)):
#                    sum += kernel[a][b] * M[i - offset + a][j - offset + b]
#            L[i][j] = sum
#    return L

# Update A & B (based on previous iteration)


def update(A, B, DA, DB, feed, k, N, kernel):
    diff_A = (
        (DA * signal.convolve2d(A, kernel, mode="same", boundary="symm"))
        - (A * B ** 2)
        + (feed * (1 - A))
    )
    diff_B = (
        (DB * signal.convolve2d(B, kernel, mode="same", boundary="symm"))
        + (A * B ** 2)
        - ((k + feed) * B)
    )

    A += diff_A
    B += diff_B

    return A, B


# Live Preview (Errors at larger simulations)
fig, axes = plt.subplots()
im = plt.imshow(A, animated=True, vmin=0, vmax=1)
plt.set_cmap("rainbow")
plt.axis("off")
# fig.subplots_adjust(0, 0, 1, 1)
fig.tight_layout(pad=0)


def animate(i):
    for n in range(10):
        update(A, B, DA, DB, feed, k, N, kernel)

    im.set_array(A)
    return [im]


ani = animation.FuncAnimation(fig, animate, blit=True, interval=250)
plt.show()

# No live preview (only last frame)
# iter = 5000
#
# for i in range(iter):
#    update(A, B, DA, DB, feed, k, N, kernel)
#    print(str(float(float(i) * 100.0 // iter)) + "%", end="\r", flush=True)
#
# print("Done!")
# plt.imshow(A)
# plt.savefig("reaction.png", dpi=300)
# plt.show()
