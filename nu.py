import numpy as np
from scipy.linalg import lu_factor, lu_solve

#  User Input 
x = list(map(float, input("Enter x values separated by space: ").split()))
y = list(map(float, input("Enter y values separated by space: ").split()))

X = np.array(x)
Y = np.array(y)
# Add column of 1s for intercept
X_design = np.vstack([np.ones(len(X)), X]).T   # 2D Matrix

# Normal Equation (A θ = b)
A = X_design.T @ X_design
b = X_design.T @ Y

#  Solve using LU Decomposition 
lu, piv = lu_factor(A)
theta_lu = lu_solve((lu, piv), b)

