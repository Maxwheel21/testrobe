import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt

# Dataset class with replacement of '-' with 0
class TimeSeriesDataset(Dataset):
    '''Custom Dataset for bivariate time-series regression (replace '-' with 0).'''
    def __init__(self, csv_file):
        self.times = []
        self.x = []
        self.y = []

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                try:
                    # Convert strings to floats, replace '-' with 0
                    time = float(row[0])
                    x = float(row[1]) if row[1] != '-' else 0.0
                    y = float(row[2]) if row[2] != '-' else 0.0
                    
                    self.times.append(time)
                    self.x.append(x)
                    self.y.append(y)
                except ValueError:
                    print(f"Skipping invalid row: {row}")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        time = self.times[idx]
        target = torch.tensor([self.x[idx], self.y[idx]], dtype=torch.float32)
        return torch.tensor(time, dtype=torch.float32), target

# Extended MLP Model with sinusoidal and polynomial features for extrapolation
class ExtrapolationNet(nn.Module):
    '''Extended MLP with sinusoidal and polynomial time features for extrapolation.'''
    def __init__(self):
        super(ExtrapolationNet, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # Input is [t, sin(t), cos(t), t^2, t^3, t^4]
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)  # Output is 2D (x, y)

    def forward(self, t):
        t = t.view(-1, 1)
        features = torch.cat(
            [t, torch.sin(t), torch.cos(t), t**2, t**3, t**4], dim=1
        )
        x = torch.relu(self.fc1(features))
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
extrapolation_net = ExtrapolationNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(extrapolation_net.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = extrapolation_net(inputs)

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

# Visualization for extrapolation
def visualize_predictions_2_2(model, dataset, t_range=(0, 100)):
    times = torch.linspace(*t_range, steps=100)  # Extrapolation range
    ground_truth_times = torch.tensor(dataset.times, dtype=torch.float32)
    ground_truth_values = torch.tensor(list(zip(dataset.x, dataset.y)), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted = model(times)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ground_truth_times, ground_truth_values[:, 0], label="True x(t)", color="blue")
    plt.plot(times, predicted[:, 0], label="Predicted x(t) (Extrapolated)", linestyle="dashed", color="red")
    plt.title("x(t) Ground Truth vs Predictions (Extrapolation)")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ground_truth_times, ground_truth_values[:, 1], label="True y(t)", color="blue")
    plt.plot(times, predicted[:, 1], label="Predicted y(t) (Extrapolated)", linestyle="dashed", color="red")
    plt.title("y(t) Ground Truth vs Predictions (Extrapolation)")
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.legend()

    plt.tight_layout()
    plt.show()

visualize_predictions_2_2(extrapolation_net, dataset)

