import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Generazione dei dati (range esteso)
np.random.seed(42)
torch.manual_seed(42)

x_train = np.linspace(-15, 15, 1500).reshape(-1, 1)
y_train = np.sin(x_train) + 0.1 * np.random.normal(0, 1, size=x_train.shape)

x_tensor = torch.tensor(x_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

# Modello con regolarizzazione
class RegularizedNN(nn.Module):
    def __init__(self):
        super(RegularizedNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Dropout(0.2),  # Aggiunta di Dropout
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(0.2),  # Aggiunta di Dropout
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Modello, loss, ottimizzatore
model = RegularizedNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 1000
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

# Predizione su [-20, 20]
x_full = np.linspace(-20, 20, 2000).reshape(-1, 1)
x_full_tensor = torch.tensor(x_full, dtype=torch.float32)
y_pred_full = model(x_full_tensor).detach().numpy()

# Grafico
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label="Dati di Addestramento", s=5)
plt.plot(x_full, y_pred_full, label="Predizione del Modello (-20, 20)", color="red", linewidth=2)
plt.legend()
plt.title("Predizione del modello: continuazione oltre il range di addestramento")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
