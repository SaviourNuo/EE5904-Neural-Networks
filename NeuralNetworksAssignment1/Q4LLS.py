# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt

# Define given input-output pairs
X = np.array([0, 0.8, 1.6, 3, 4.0, 5.0])
Y = np.array([0.5, 1, 4, 5, 6, 8])

# Define necessary intermediate variables
sum_x = 0
sum_y = 0
sum_xy = 0
sum_x_square = 0
n = len(X)

for i in range(n):
    sum_x +=  X[i]
    sum_y += Y[i]
    sum_x_square += X[i] ** 2
    sum_xy += X[i] * Y[i]

avg_x = sum_x / n
avg_y = sum_y / n
w = (sum_xy - n * avg_x * avg_y) / (sum_x_square - n * avg_x ** 2)
b = avg_y - w * avg_x
X_fit = np.linspace(min(X), max(X), 100)
Y_fit = w * X_fit + b

# Plot scatter plots and the fit line
plt.figure()
plt.scatter(X, Y, color="red", label="Input-output Pairs", edgecolors="black")
plt.plot(X_fit, Y_fit, color="blue", label=f"LLS Fit:  $y={w:.3f}x + {b:.3f}$")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Least-squares Method")
plt.legend()
plt.grid(True)
plt.show()