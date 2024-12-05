import torch
import matplotlib.pyplot as plt

# Generate data
def generate_data():
    data = torch.rand(1000, 2)
    label = ((data[:, 0] + 0.3 * data[:, 1]) > 0.5).to(torch.int)
    return data[:, 0], label

# Initialize data and parameters
input, label = generate_data()
inputs = torch.split(input, 32)
labels = torch.split(label, 32)

# Define the parameters to optimize
b1 = torch.autograd.Variable(torch.tensor([0.01], dtype=torch.float32), requires_grad=True)
b2 = torch.autograd.Variable(torch.tensor([0.01], dtype=torch.float32), requires_grad=True)

# Learning rate
alpha = 0.1

# Training loop
for epoch in range(15):
    for x_batch, y_batch in zip(inputs, labels):
        # Logistic regression model
        p_x = 1 / (1 + torch.exp(-(b1 + b2 * x_batch)))
        
        # Negative log-likelihood loss
        loss = -torch.sum(y_batch * torch.log(p_x) + (1 - y_batch) * torch.log(1 - p_x))
        
        # Backpropagation: Compute gradients
        loss.backward()
        
        # Update parameters
        with torch.no_grad():
            b1 -= alpha * b1.grad
            b2 -= alpha * b2.grad
            
            # Zero the gradients after updating
            b1.grad.zero_()
            b2.grad.zero_()
    
    # Print loss per epoch
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Plot the result
x = torch.linspace(0, 1, 100)
y_pred = 1 / (1 + torch.exp(-(b1.detach() + b2.detach() * x)))

plt.scatter(input, label, color="red", alpha=0.5, label="True Data")
plt.plot(x, y_pred, color="blue", label="Learned Logistic Curve")
plt.xlabel("x")
plt.ylabel("Probability")
plt.legend()
plt.title("Logistic Regression Decision Boundary")
plt.show()
