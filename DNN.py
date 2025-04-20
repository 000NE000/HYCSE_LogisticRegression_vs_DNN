import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class DNN(nn.Module):
    def __init__(self, layer_dims, learning_rate=0.001, momentum=0.9, reg_coeff=0.01, dropout_rate=0.1):
        super(DNN, self).__init__()

        #Store hyperparameter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reg_coeff = reg_coeff
        self.dropout_rate = dropout_rate


        #Network Architecture : 3 layer + BN!!
        self.layer1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.bn1 = nn.BatchNorm1d(layer_dims[1])
        self.layer2 = nn.Linear(layer_dims[1], layer_dims[2])
        self.bn2 = nn.BatchNorm1d(layer_dims[2])
        self.layer3 = nn.Linear(layer_dims[2], 1)

        #dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        #Parameter initialization
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')

        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image [batch_size, channels, height, width] -> [batch_size, input_dim]
        # Pass through the 1st linear layer, apply Batch Normalization, then ReLU and dropout
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Pass through the 2nd linear layer, apply Batch Normalization, then ReLU and dropout
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Pass through the 3rd linear layer, apply Batch Normalization, then ReLU and dropout
        x = self.layer3(x)

        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() # Set model to training mode (enables dropout, BN)
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        targets = labels.to(device).float().unsqueeze(1) # For BCEWithLogitsLoss, ensure targets are float tensors with shape [batch_size, 1]

        optimizer.zero_grad() # Reset gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Compute loss

        loss.backward() #Backpropagation to compute gradients
        optimizer.step() # Update model parameters

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float() # Convert logits to probabilities using sigmoid and classify with threshold 0.5
        correct_predictions += (preds == targets).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)

    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, device):
    model.eval() # Set model to evaluation mode (disables dropout, BN updates)
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad(): # Disable gradient computations for efficiency
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            targets = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_predictions += (preds == targets).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy

def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, patience=5):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    best_train_acc, best_val_acc = 0, 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        # If validation loss improves, save model state and reset the early stop counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stopping_counter = 0
            best_train_acc = train_acc
            best_val_acc = val_acc
            print("Validation loss improved. Saving best model.")
        else:
            early_stopping_counter += 1
            print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")
            if (early_stopping_counter >= patience):
                print("Early stopping activated. Stopping training.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_train_acc, best_val_acc