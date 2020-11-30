
import numpy as np
from itertools import product

# fixed-point tensor
Astar = np.array([[1/2, 1/2], [1/2, 1/2]])
# identity matrix
idM = np.eye(2)
# initialize the response matrix R
R = np.zeros([4, 4])
# assemble matrix R
for a, b, c, d in product(range(2), range(2), range(2), range(2)):
    R[2 * a + b, 2 * c + d] = idM[a, c] * Astar[d, b] + Astar[a, c] * idM[d, b]
print("The response matrix at high temperature phase fixed point is")
print(R)
