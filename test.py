import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the GPU
model = SimpleNN().to(device)

# Create some dummy data
dummy_input = torch.randn(64, 10).to(device)  # Batch size of 64, input size of 10
dummy_target = torch.randn(64, 1).to(device)  # Corresponding targets

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Perform a forward pass
outputs = model(dummy_input)
loss = criterion(outputs, dummy_target)

# Backpropagation and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
