import torch

def train_model(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    Args:
      - model: The neural network model.
      - dataloader: DataLoader providing the training data.
      - criterion: Loss function (BCEWithLogitsLoss) which handles both sigmoid and BCE.
      - optimizer: Adam optimizer with L2 regularization (weight_decay).
      - device: 'cpu' or 'cuda' (if GPU is available).

    Returns:
      - avg_loss: The average training loss over the epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    # [batch gradient decent]
    for inputs, targets in dataloader: # Each iteration retrieves a batch of images (inputs) and their labels (targets).
        # Move data to the designated device
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters


        running_loss += loss.item() * inputs.size(0)
        # Convert logits to probability using Sigmoid, then classify with threshold 0.5
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct_predictions += (preds == targets).sum().item()
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    print("Returning from evaluate_model:", avg_loss, accuracy)
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the given data loader.

    Args:
      - model: The trained model.
      - dataloader: DataLoader providing the evaluation data.
      - criterion: Loss function.
      - device: 'cpu' or 'cuda'.

    Returns:
      - avg_loss: The average evaluation loss.
      - accuracy: Proportion of correct predictions.
    """
    model.eval()  # Set model to evaluation mo     de
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad(): #for memory conservation . it disables GD
        for inputs, targets in dataloader:

            # Move to the appropriate device
            inputs = inputs.to(device)
            # For binary labels, ensure targets shape is consistent: [batch_size, 1]
            targets = targets.to(device).float().unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            # Convert logits to probability using Sigmoid, then classify with threshold 0.5
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_predictions += (preds == targets).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    print("Returning from evaluate_model, average_loss", avg_loss, "accracy : ", accuracy)
    return avg_loss, accuracy