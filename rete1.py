import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definiamo una funzione non lineare (ad esempio, il seno)
def non_linear_function(x):
    return np.sin(x)

# Creiamo i dati di addestramento
x_train = np.linspace(0, 20, 100)  # punti da 0 a 20
y_train = non_linear_function(x_train)

# Creiamo i dati di test
x_test = np.linspace(20, 40, 100)  # punti da 20 a 40
y_test = non_linear_function(x_test)

# Convertiamo i dati in tensori PyTorch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Definiamo un semplice modello di rete neurale
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Inizializziamo il modello, la funzione di perdita e l'ottimizzatore
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Addestramento del modello
epochs = 500
for epoch in range(epochs):
    # Azzeriamo i gradienti
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x_train_tensor)
    
    # Calcoliamo la perdita
    loss = criterion(y_pred, y_train_tensor)
    
    # Backward pass
    loss.backward()
    
    # Aggiorniamo i pesi
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
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
plt.title("Non-linear Function Prediction with PyTorch")
plt.show()
