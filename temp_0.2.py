#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import cv2

# Set Constants
N = 100
DA = 1.0
DB = 0.5
feed = 0.055
k = 0.062

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)

# Initialise Chemicals A & B
A = np.ones((N, N))
B = np.zeros((N, N))

# Now let's add a disturbance in the center
N2 = N // 2
radius = r = int(N / 50.0)

A[N2 - r : N2 + r, N2 - r : N2 + r] = 0.50
B[N2 - r : N2 + r, N2 - r : N2 + r] = 0.25

# Initialise Kernel
def convolution(M):
    cen = -1
    adj = 0.2
    cor = 0.05

    kernel = np.array([[cor, adj, cor], [adj, cen, adj], [cor, adj, cor]])

    offset = len(kernel) // 2  # Only Compatible with 1:1 matrices

    L = np.zeros((M.shape[0], M.shape[1]))

    for i in range(offset, M.shape[0] - offset):
        for j in range(offset, M.shape[1] - offset):
            sum = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    sum += kernel[a][b] * M[i - offset + a][j - offset + b]
            L[i][j] = sum
    return L


# Update A & B
def update(A, B, DA, DB, feed, k, N):

    LA = convolution(A)
    LB = convolution(B)

    diff_A = ((DA * LA) - (A * B ** 2) + (feed * (1 - A))) * 0.1
    diff_B = ((DB * LB) + (A * B ** 2) - ((k + feed) * B)) * 0.1

    A += diff_A
    B += diff_B

    return A, B


for i in range(100):
    A, B = update(A, B, DA, DB, feed, k, N)

AS = cv2.resize(A, (800, 800))
cv2.imshow("reaction", AS)
cv2.waitKey(0)
cv2.destroyAllWindows()
# def animate(i):
#    update(A, B, DA, DB, feed, k, N)
#    ax1.clear()
#    ax1.imshow(A)
#
#
# ani = animation.FuncAnimation(fig, animate, interval=100)
# plt.show()
