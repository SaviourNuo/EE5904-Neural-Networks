# Import necessary packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sympy import false

# Define a Multi-Layer Perceptron class for COMPLEMENT task
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.out = nn.Linear(1,1)
        ''' Define a full connected linear layer
        with 1 input feature (x) and 1 output feature (y)'''
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
X=torch.tensor([[0],[1]], dtype = torch.float32)
Y=torch.tensor([[1],[0]], dtype = torch.float32)

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
plt.plot(weight_trajectories, label = "Weight (w)")
plt.plot(bias_trajectories, label = "Bias (b)", linestyle = "dashed")
plt.title("Trajectories of Weights and Bias (COMPLEMENT Task)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show(block = false)

# Plot the decision boundary
x_range = np.linspace(-0.5, 1.5, 200).reshape(-1, 1) # Input range from -0.5 to 1.5, reshape to (100,1) dimension
tensor_range = torch.tensor(x_range, dtype=torch.float32) # Numpy array to Torch tensor

with torch.no_grad():
    y_predict = mlp(tensor_range).numpy()  # 计算 x_range 上的模型预测值

plt.figure()
plt.scatter(X.numpy(), np.zeros_like(X.numpy()), c=Y.numpy(), edgecolors="k", marker="o", cmap="coolwarm", s=100)
decision_boundary = tensor_range[np.abs(y_predict - 0.5).argmin()].item()
plt.axvspan(-0.5, decision_boundary, color="red", alpha=0.3)
plt.axvspan(decision_boundary, 1.5, color="blue", alpha=0.3)
plt.axvline(decision_boundary, color="black", label=f"x = {decision_boundary:.3f})")
plt.title("Decision Boundary of COMPLEMENT Task")
plt.xlabel("X")
plt.xticks([0, 0.5, 1])
plt.yticks([])
plt.legend()
plt.show()

