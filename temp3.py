#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

# Set Constants
N = 100
DA = 1.0
DB = 0.5
feed = 0.055
k = 0.062

# Initialise Chemicals A & B
A = np.ones((N, N))
B = np.zeros((N, N))

# Add Square to the middle
N2 = N // 2
r = int(N / 10.0)

A[N2 - r : N2 + r, N2 - r : N2 + r] = 0.50
B[N2 - r : N2 + r, N2 - r : N2 + r] = 0.25

cen = -1
adj = 0.2
cor = 0.05

kernel = np.array([[cor, adj, cor], [adj, cen, adj], [cor, adj, cor]])

offset = len(kernel) // 2  # Only Compatible with 1:1 matrices

# Laplacian Function (TO BE FIXED)
def convolution(M, kernel, offset):

    L = np.zeros((M.shape[0], M.shape[1]))

    for i in range(offset, M.shape[0] - offset):
        for j in range(offset, M.shape[1] - offset):
            sum = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    sum += kernel[a][b] * M[i - offset + a][j - offset + b]
            L[i][j] = sum
    return L


# Update A & B (based on previous iteration)
def update(A, B, DA, DB, feed, k, N, kernel, offset):

    diff_A = (DA * convolution(A, kernel, offset)) - (A * B ** 2) + (feed * (1 - A))
    diff_B = (DB * convolution(B, kernel, offset)) + (A * B ** 2) - ((k + feed) * B)

    A += diff_A
    B += diff_B

    return A, B


iter = 100

for i in range(iter):
    update(A, B, DA, DB, feed, k, N, kernel, offset)
    print(str(i * 100 // iter) + ".0%", end="\r", flush=True)

print("Done!")
plt.imshow(A)
plt.show()
