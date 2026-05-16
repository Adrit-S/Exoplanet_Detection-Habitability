import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from LightCurveDataset import LightCurveDataset, DATA_DIRECTORY
from sklearn.model_selection import KFold

# Utilize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Network Architecture
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)  # Convolutional Layer 1
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)  # Convolutional Layer 2
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)  # Convolutional Layer 3
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # Convolutional Layer 4
        self.bn4 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Compress each channel to a single value
        self.fc1 = nn.Linear(128, 64)  # Adjusted fully connected layer size
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 2)  # Binomial output 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # Apply new conv4

        x = self.global_pool(x)  # Pooling
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer

        return x

def train_model():
    # Load dataset with data augmentation enabled
    dataset = LightCurveDataset(DATA_DIRECTORY)

    # Initialize CNN model and move to appropriate device (GPU/CPU)
    model = CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Implement 2-fold Cross Validation (as an example)
    kfold = KFold(n_splits=2, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{2}")

        # Create training and validation datasets for the current fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)  # Reduced batch size
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)

        # Training loop
        for epoch in range(10):
            model.train()
            train_loss = 0
            all_preds, all_labels = [], []

            # Training loop
            for flux, label in train_loader:
                flux, label = flux.to(device), label.long().to(device)

                optimizer.zero_grad()  # Reset gradients
                outputs = model(flux)  # Forward pass
                loss = criterion(outputs, label)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters

                train_loss += loss.item()

                # Compute predictions and store results for metrics
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = label.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

            # Compute training metrics
            train_loss /= len(train_loader)
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            # Print epoch results
            print(f"Epoch [{epoch+1}/10], Loss: {train_loss:.4f}, "
                  f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            # Step scheduler after each epoch
            scheduler.step()

        # Save the trained model after each fold
        torch.save(model.state_dict(), f"lightcurve_cnn_fold{fold+1}.pth")
        print(f"Model for Fold {fold+1} saved!")

if __name__ == "__main__":
    train_model()
