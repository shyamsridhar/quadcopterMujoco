import numpy as np

Y = np.array([50, 55, 60, 65, 75, 85, 100], ndmin=2).T

A = np.array([[629.88, 1],[693.24, 1],[744.87, 1],[791.99, 1],[894.33, 1],[988.76, 1],[1095.78, 1]])

inverse = np.linalg.inv(np.matmul(A.T,A))

X = np.matmul(np.matmul(inverse,A.T),Y)
print(X)