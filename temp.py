#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set Resolution of grid
RES = 100

# Create an empty grid
grid = np.empty((RES, RES))

# Fill with random integers
for i in range(RES):
    for j in range(RES):
        grid[i, j] = np.array([1, 2])

# Plot and Show
plt.imshow(grid, cmap=cm.Greys_r)
plt.show()
