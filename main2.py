import os  # For file and path handling

import optuna
import torch  # For tensors, generators, etc.
from torch.utils.data import DataLoader, random_split  # For data loading/splitting
from torchvision import datasets, transforms  # For image datasets and transformations
from LR import LR
from DNN import DNN, train_one_epoch, train_model_with_early_stopping, evaluate_model
import torch.nn as nn
import torch.optim as optim
from train_and_evalutate import train_model, evaluate_model

root_dir = os.path.expanduser("../chest_xray")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), #Converts RGB images to 1-channel grayscale
    transforms.Resize ((128, 128)), #Resizes all images to 128x128 pixels
    transforms.ToTensor(), #Converts the PIL image (0–255) to a PyTorch tensor (0–1) and adds a channel dimension (C×H×W).
    # transforms.Normalize(mean=[0.5], std=[0.5]),
])

input_dim = 128 * 128 #Calculates the flattened input dimension for the model

trainval_ds = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=transform)
test_ds = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=transform)
train_size = int(0.8 * len(trainval_ds))
val_size = len(trainval_ds) - train_size
train_ds, val_ds = random_split(trainval_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader (train_ds, batch_size=32, shuffle=True) #Creates a DataLoader for the training set B = 32
val_loader = DataLoader(val_ds, batch_size=32) #Validation DataLoader (no shuffling, deterministic evaluation)
test_loader = DataLoader (test_ds, batch_size=32) #Test DataLoader (used during final model evaluation)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")






def objective(trial):
    # Sample hyperparameters
    hidden_dim1 = trial.suggest_int("hidden_dim1", 256, 1024)
    hidden_dim2 = trial.suggest_int("hidden_dim2", 128, hidden_dim1)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # Set layer dimensions: [input_dim, hidden_dim1, hidden_dim2]
    layer_dims = [input_dim, hidden_dim1, hidden_dim2]

    # Create the model and move it to the device
    model = DNN(layer_dims, dropout_rate=dropout_rate).to(device)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Set up the optimizer with the sampled hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model for a few epochs for quick tuning (e.g., 5 epochs)
    num_epochs = 27
    for epoch in range(num_epochs):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate on the validation set
    val_loss, _ = evaluate_model(model, val_loader, criterion, device)
    return val_loss

# Run the optimization (number of trials can be adjusted)
# study = optuna.create_study(direction="minimize")
# print("Starting hyperparameter optimization with Optuna...")
# study.optimize(objective, n_trials=50)
#
# best_params = study.best_trial.params
# print("============================================================")
# print("Best hyperparameters:", best_params)


best_params = {
    'hidden_dim1': 939,
    'hidden_dim2': 799,
    'dropout_rate': 0.2963617429089822,
    'learning_rate': 0.002844883832313113,
    'weight_decay': 1.2973725923247895e-05
}


# Build the final model using the best hyperparameters from Optuna
layer_dims_final = [input_dim, best_params["hidden_dim1"], best_params["hidden_dim2"]]
final_model = DNN(layer_dims_final, dropout_rate=best_params["dropout_rate"]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])

# Train the final model for the full experiment epochs using early stopping
experiment_epoch = [2, 10, 20]

for num_epochs in experiment_epoch:
    ######################################
    #                   LR               #
    ######################################
    lr_instance = LR(input_dim, train_loader, val_loader, test_loader)
    acc_LR_train, acc_LR_val = lr_instance.train_LR(num_epochs, save_path='model_LR.pth')
    _, acc_LR_test = lr_instance.evaluate_LR(save_path='model_LR.pth')

    ######################################
    #                DNN                 #
    ######################################
    best_model_final, DNN_train_acc, DNN_val_acc = train_model_with_early_stopping(
        final_model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device, patience=4
    )
    _, DNN_test_acc = evaluate_model(best_model_final, test_loader, criterion, device)

    ######################################
    #         PLOTTING RESULTS           #
    ######################################

    import matplotlib.pyplot as plt

    LR_results = ['LR_train', 'LR_val', 'LR_test']
    LR_accuracies = [acc_LR_train, acc_LR_val, acc_LR_test]

    DNN_results = ['DNN train', 'DNN val', 'DNN test']
    DNN_accuracies = [DNN_train_acc, DNN_val_acc, DNN_test_acc]

    plt.bar(LR_results, LR_accuracies)
    plt.bar(DNN_results, DNN_accuracies)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()

