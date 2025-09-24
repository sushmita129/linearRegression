import numpy as np
from scipy.linalg import lu_factor, lu_solve

#  User Input 
x = list(map(float, input("Enter x values separated by space: ").split()))
y = list(map(float, input("Enter y values separated by space: ").split()))

X = np.array(x)
Y = np.array(y)


