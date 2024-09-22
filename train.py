import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(X, y, batch_size=32):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, criterion, optimizer, device, epochs=1000):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')
