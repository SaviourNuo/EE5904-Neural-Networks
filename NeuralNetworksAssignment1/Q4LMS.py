# Import necessary packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sympy import false

# Define given input-output pairs
# Float32 to meet the requirements of torch.nn.Linear and gradient calculation
X = torch.tensor([[0], [0.8], [1.6], [3], [4.0], [5.0]], dtype = torch.float32)
Y = torch.tensor([[0.5], [1], [4], [5], [6], [8]], dtype = torch.float32)

# Randomly initialize weight and bias and record their change
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
weight_trajectories = []
bias_trajectories = []

# Define hyperparameters
learning_rate = 0.15
epochs = 100
loss_function = nn.MSELoss() # Mean-Square Error function as loss function

# Model training
for epoch in range(epochs):
    y_predict = w * X + b
    loss = loss_function(y_predict, Y) # Loss calculation
    loss.backward() # Back propagation
    with torch.no_grad():
        w -= learning_rate * w.grad # Gradient calculation
        b -= learning_rate * b.grad

    w.grad.zero_() # Clear gradient
    b.grad.zero_()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Add the updated weights and biases to the list for every epoch
    weight_trajectories.append(w.detach().numpy().flatten().copy())
    bias_trajectories.append(b.detach().numpy().copy())

weight_trajectories = np.array(weight_trajectories)
bias_trajectories = np.array(bias_trajectories)
w_value = w.detach().numpy()
b_value = b.detach().numpy()

X_fit = np.linspace(min(X), max(X), 100)
Y_fit = w_value * X_fit + b_value

# Plot the trajectories of weights and biases
plt.figure()
plt.plot(weight_trajectories, label = "Weight (w)")
plt.plot(bias_trajectories, label = "Bias (b)", linestyle = "dashed")
plt.title("Trajectories of Weights and Bias (Learning Rate = 0.15)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show(block = false)

# Plot scatter plots and the fit line
plt.figure()
plt.scatter(X, Y, color="red", label="Input-output Pairs", edgecolors="black")
plt.plot(X_fit, Y_fit, color="blue", label=f"LMS Fit:  $y={w_value[0]:.3f}x + {b_value[0]:.3f}$")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Least-mean-square Method (Learning Rate = 0.15)")
plt.legend()
plt.grid(True)
plt.show()