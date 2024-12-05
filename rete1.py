import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definizione della funzione non lineare (esempio: seno)
def non_linear_function(x):
    return np.sin(x)

# Creazione dei dati di addestramento (0-20) e di test (20-40)
x_train = np.linspace(0, 20, 100)  # punti da 0 a 20
y_train = non_linear_function(x_train)

x_test = np.linspace(20, 40, 100)  # punti da 20 a 40
y_test = non_linear_function(x_test)

# Normalizzazione degli input
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

# Convertire in tensori PyTorch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Modello di rete neurale aggiornato
class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(1, 128)  # Primo livello
        self.fc2 = nn.Linear(128, 128)  # Secondo livello
        self.fc3 = nn.Linear(128, 1)  # Output
        self.tanh = nn.Tanh()  # Funzione di attivazione non lineare
    
    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Inizializzazione del modello, funzione di perdita e ottimizzatore
model = EnhancedNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Addestramento del modello
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Test del modello
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test_tensor)

# Visualizzazione dei risultati
plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label="Train Data (Ground Truth)", color="blue")
plt.plot(x_test, y_test, label="Test Data (Ground Truth)", color="green")
plt.plot(x_test, y_test_pred.numpy(), label="Test Data (Prediction)", color="red", linestyle="dashed")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Non-linear Function Prediction with Enhanced PyTorch Model")
plt.show()
