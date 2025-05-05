# Import necessary packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sympy import false

# Define a Multi-Layer Perceptron class
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.out = nn.Linear(2,1)
        ''' Define a full connected linear layer
        with 2 input features (x1,x2) and 1 output feature (y)'''
        self.sigmoid = nn.Sigmoid() # Sigmoid as activation function

        nn.init.normal_(self.out.weight, mean = 0.0, std = 0.1) # Randomly initialize weights
        nn.init.zeros_(self.out.bias) # Set initial biases to be 0

    def forward(self,x):
        x = self.sigmoid(self.out(x))
        ''' Pass input through the Linear layer
        then apply the activation function'''
        return x

# Create an instance of MLP class
mlp=MLP()

# Define input features and expected output
# Float32 to meet the requirements of torch.nn.Linear and gradient calculation
X=torch.tensor([(0,0),(0,1),(1,0),(1,1)], dtype = torch.float32)
Y=torch.tensor([[1],[0],[0],[1]], dtype = torch.float32)

# Record weights and biases change
weight_trajectories = []
bias_trajectories = []

# Define hyperparameters
learning_rate = 1.0
epochs = 100
loss_function = nn.MSELoss() # Mean-Square Error function as loss function
optimizer = torch.optim.SGD(mlp.parameters(), lr = learning_rate) # Stochastic gradient descend as optimizer

# Model training
for epoch in range(epochs):
    optimizer.zero_grad() # Clear gradient
    y_predict = mlp(X) # Forward propagation
    loss = loss_function(y_predict,Y) # Loss calculation
    loss.backward() # Back propagation
    optimizer.step() # Weight update
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}") # Print epoch and loss respectively

    # Add the updated weights and biases to the list
    weight_trajectories.append(mlp.out.weight.detach().numpy().flatten().copy())
    bias_trajectories.append(mlp.out.bias.detach().numpy().copy())

weight_trajectories = np.array(weight_trajectories)
bias_trajectories = np.array(bias_trajectories)

# Plot the trajectories of weights and biases
plt.figure()
plt.plot(weight_trajectories[:, 0], label = "Weight 1 (w1)")
plt.plot(weight_trajectories[:, 1], label = "Weight 2 (w2)")
plt.plot(bias_trajectories, label = "Bias (b)", linestyle = "dashed")
plt.title("Trajectories of Weights and Bias (XOR Task)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show(block = false)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
res = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
''' After flattening the two-dimensional matrix into a one-dimensional array,
it is then concatenated into an array of two-dimensional coordinates'''
grid_tensor = torch.tensor(grid_points) # Numpy array to Pytorch tensor

with torch.no_grad(): # Disable gradient calculation
    zz = mlp(grid_tensor).numpy() # Put grid tensor into trained model for prediction
    zz = (zz > 0.5).astype(np.float32).reshape(xx.shape)
    ''' Binarization: zz greater than 0.5 predicted as Class 1 (y = 1),
    other wise Class 0 (y = 0)'''

plt.figure()
plt.contourf(xx, yy, zz, alpha = 0.5, cmap = "coolwarm")
plt.scatter(X[:, 0], X[:, 1], c = Y.numpy(), edgecolors = "k", marker = "o", cmap = "coolwarm")
plt.title("Decision Boundary of XOR Task (NOT Linearly Separable)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()