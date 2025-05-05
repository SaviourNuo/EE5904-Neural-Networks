# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import random

# Starting point randomly initialized in the open interval (-1,1)
while True:
    x_init = random.uniform(-1, 1)
    y_init = random.uniform(-1, 1)
    if (x_init != -1 and x_init != 1) or (y_init != -1 and y_init != 1):
        break

# Define Rosenbrock function
def rosenbrock(x, y):
    return (1-x) ** 2 + 100 * (y - x ** 2) **2

# Define the derivative function of Rosenbrock
def rosenbrock_gradient(x, y):
    dfx = 400 * x ** 3 - 400 * x * y + 2 * x - 2
    dfy = 200 * y - 200 * x ** 2
    return np.array([dfx, dfy])

# Define hyperparameters
xy_trajectory = []
function_values = []
learning_rate = 0.001
epochs = 100000
threshold = 0.001
x_update = x_init
y_update = y_init

# Find the minimum value of the rosenbrock function by steepest gradient descent method
for i in range(epochs):
    xy_trajectory.append((x_update, y_update))
    function_values.append(rosenbrock(x_update,y_update))
    grad = rosenbrock_gradient(x_update, y_update)
    x_update -= learning_rate * grad[0]
    y_update -= learning_rate * grad[1]

    # If the L2 norm of the gradient is less thant the predetermined threshold
    # The function is considered to have basically converged
    if np.linalg.norm(grad) < threshold:
        print(f"Converged at epoch {i+1}")
        break

print(f"Final (x, y): ({x_update:.6f}, {y_update:.6f})")
print(f"Function value at final point: {rosenbrock(x_update, y_update):.6f}")

# Extract x and y values of each interation separately
x_vals, y_vals = zip(*xy_trajectory)

# Plot the trajectory of (x,y)
plt.figure()
plt.plot(x_vals, y_vals, marker = "o", linestyle = "-", color = "b", markersize = 2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Optimization Trajectory")
plt.show(block = False)

# Plot the function value
plt.figure()
plt.plot(range(len(function_values)), function_values, color = "r")
plt.xlabel("Iterations")
plt.ylabel("Function Value")
plt.title("Function Value Over Iterations")
plt.show()

