import torch
import torch.nn as nn

# Linear regression
# f = w * x
# here : f = 2 * x

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Design Model, the model has to implement the forward pass!

# Here we could simply use a built-in model from PyTorch
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):# base class for all neural network models
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__() #initializes the nn.Module
        #Required in all PyTorch models to make things like .parameters() and .to(device) work.
        self.lin = nn.Linear(input_dim, output_dim) # This defines a linear layer: output=w⋅x+b
    def forward(self, x):
        return self.lin(x)

input_size, output_size = n_features, n_features # both 1
model = LinearRegression(input_size, output_size)
print(f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#model.parameters() tells the optimizer what to update (i.e., the weights and biases of your mode

# 3) Training loop
for epoch in range(n_epochs):
    # predict = forward pass with our model
    y_predicted = model(X)

    l = loss(Y, y_predicted)
   # calculate gradients
    l.backward()
    # update weights
    optimizer.step()
   # zero the gradients after updating
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        w, b = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l.item())

print(f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')