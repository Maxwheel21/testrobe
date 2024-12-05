import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt

# Dataset class with handling for invalid values
class TimeSeriesDataset(Dataset):
    '''Custom Dataset for bivariate time-series regression (without pandas).'''
    def __init__(self, csv_file):
        self.times = []
        self.x = []
        self.y = []

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                try:
                    # Convert strings to floats, skip rows with invalid values
                    time = float(row[0])
                    x = float(row[1]) if row[1] != '-' else None
                    y = float(row[2]) if row[2] != '-' else None
                    
                    if x is not None and y is not None:
                        self.times.append(time)
                        self.x.append(x)
                        self.y.append(y)
                    else:
                        print(f"Skipping invalid row: {row}")
                except ValueError:
                    print(f"Skipping invalid row: {row}")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        time = self.times[idx]
        target = torch.tensor([self.x[idx], self.y[idx]], dtype=torch.float32)
        return torch.tensor(time, dtype=torch.float32), target

# Define the MLP Model
class Net(nn.Module):
    '''Model to regress 2D time series values given scalar input time.'''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # Input is scalar time
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Output is 2D (x, y)

    def forward(self, x):
        x = x.view(-1, 1)  # Ensure x is treated as a 2D input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 300

# Load the dataset
dataset = TimeSeriesDataset('data.csv')
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Initialize model, loss function, and optimizer
net = Net()
loss_fn = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss for logging
        running_loss += loss.item()
        if i % 20 == 19:  # Print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished Training')

# Visualization
def visualize_predictions_2_1(model, dataset):
    times = torch.tensor(dataset.times, dtype=torch.float32)
    ground_truth = torch.tensor(list(zip(dataset.x, dataset.y)), dtype=torch.float32)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        predicted = model(times)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, ground_truth[:, 0], label="True x(t)", color="blue")
    plt.plot(times, predicted[:, 0], label="Predicted x(t)", linestyle="dashed", color="red")
    plt.title("x(t) Ground Truth vs Predictions")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, ground_truth[:, 1], label="True y(t)", color="blue")
    plt.plot(times, predicted[:, 1], label="Predicted y(t)", linestyle="dashed", color="red")
    plt.title("y(t) Ground Truth vs Predictions")
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.legend()

    plt.tight_layout()
    plt.show()

visualize_predictions_2_1(net, dataset)

