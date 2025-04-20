import os  # For file and path handling
import torch  # For tensors, generators, etc.
from torch.utils.data import DataLoader, random_split  # For data loading/splitting
from torchvision import datasets, transforms  # For image datasets and transformations
import torch.nn as nn
import torch.optim as optim
from LogisticModel import LogisticModel
from train_and_evalutate import train_model, evaluate_model

class LR:
    def __init__(self, input_dim, train_loader, val_loader, test_loader):
        self.input_dim = input_dim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def evaluate_LR(self, save_path='model_LR.pth'):
        # Device configuration
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_LR = LogisticModel(self.input_dim).to(device) # Initialize the model architecture with the same input dimension

        # Load the saved model parameters
        model_LR.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded model from {save_path}")

        # Define the loss function for evaluation (must be consistent with training)
        criterion = nn.BCEWithLogitsLoss()

        # Evaluate the model using your evaluation function
        avg_loss, accuracy = evaluate_model(model_LR, self.test_loader, criterion, device)
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def train_LR(self, num_epochs, save_path = 'model_LR.pth'):
        # Device configuration
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model_LR = LogisticModel(self.input_dim).to(device)

        # Loss function (BCEWithLogitsLoss automatically includes Sigmoid activation)
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer: Adam with learning rate 1e-3 and L2 regularization (weight_decay)
        optimizer = optim.Adam(model_LR.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_loss = float('inf')
        final_train_accuracy = None
        final_val_accuracy = None

        try:
            for epoch in range(num_epochs):
                model_LR.train()
                train_loss, train_accuracy = train_model(model_LR, self.train_loader, criterion, optimizer, device)

                # Validation Phase
                model_LR.eval()
                val_loss, val_accuracy = evaluate_model(model_LR, self.val_loader, criterion, device)

                print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, " +
                      f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                final_train_accuracy = train_accuracy
                final_val_accuracy = val_accuracy


                # Save the model if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model_LR.state_dict(), save_path) # Save the trained model's parameters
                    print(f"Model saved with validation loss: {best_val_loss:.4f}")

        except Exception as e:
            print("Exception occured: ", e)
        return final_train_accuracy, final_val_accuracy
        # Training loop