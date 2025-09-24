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

#  Solve using Least Squares  
theta_ls, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)

#  Show Results 
print("\nUsing LU Decomposition:")
print(f"  Intercept = {theta_lu[0]:.4f}")
print(f"  Slope     = {theta_lu[1]:.4f}")
print(f"  Equation  : y = {theta_lu[0]:.2f} + {theta_lu[1]:.2f}x")

print("\nUsing Standard Least Squares:")
print(f"  Intercept = {theta_ls[0]:.4f}")
print(f"  Slope     = {theta_ls[1]:.4f}")
print(f"  Equation  : y = {theta_ls[0]:.2f} + {theta_ls[1]:.2f}x")

#  Prediction 
x_new = float(input("\nEnter a new x value for prediction: "))
y_pred = theta_lu[0] + theta_lu[1] * x_new
print(f"Prediction: For x = {x_new}, predicted y = {y_pred:.4f}")


