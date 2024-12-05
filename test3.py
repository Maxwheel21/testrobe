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

extrapolation_net = ExtrapolationNet()
optimizer = optim.Adam(extrapolation_net.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = extrapolation_net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 20 == 19:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished Training')

# Visualization
def visualize_predictions_2_2(model, dataset, t_range=(0, 100)):
    times = torch.linspace(*t_range, steps=100)
    ground_truth_times = dataset.times
    ground_truth_values = torch.tensor(list(zip(dataset.x, dataset.y)), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted = model(times)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ground_truth_times, ground_truth_values[:, 0], label="True x(t)", color="blue")
    plt.plot(times, predicted[:, 0], label="Predicted x(t)", linestyle="dashed", color="red")
    plt.title("x(t) Ground Truth vs Predictions (Extrapolation)")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ground_truth_times, ground_truth_values[:, 1], label="True y(t)", color="blue")
    plt.plot(times, predicted[:, 1], label="Predicted y(t)", linestyle="dashed", color="red")
    plt.title("y(t) Ground Truth vs Predictions (Extrapolation)")
    plt.xlabel("Time")
    plt.ylabel("y(t)")
    plt.legend()

    plt.tight_layout()
    plt.show()

visualize_predictions_2_2(extrapolation_net, dataset)

