import torch
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
from model import NeuralNetwork
from train import create_dataloader, save_model
from evaluate import evaluate_model

def main(file_path='train.csv', batch_size=32, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    X, y, vectorizer, mlb = preprocess_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_loader = create_dataloader(X_train, y_train, batch_size)
    test_loader = create_dataloader(X_test, y_test, batch_size)

    # Initialize the model, loss function, and optimizer
    model = NeuralNetwork(input_size=X.shape[1], output_size=y.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, epochs)

    # Save the trained model
    save_model(model, 'trained_model_1000e.pth')

    # Make predictions
    with torch.no_grad():
        model.eval()  
        y_test_pred_logits = model(torch.FloatTensor(X_test).to(device))
        y_test_pred = (y_test_pred_logits.cpu().numpy() > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Evaluate the model
    evaluate_model(y_test, y_test_pred)

if __name__ == "__main__":
    main()
