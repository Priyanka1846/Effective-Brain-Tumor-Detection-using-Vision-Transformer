import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


def train_model(model, train_loader, test_loader, device, epochs=10,
                save_path='best_model.pth', lr=1e-4, weight_decay=1e-5,
                early_stopping=True, patience=5, metrics_dir="metrics"):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    epochs_no_improve = 0

    train_losses = []
    val_accuracies = []
    train_accuracies = []
    val_losses = []
    
    os.makedirs(metrics_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = 100 * correct_val / total_val

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% - "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # Save best model if improved
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with accuracy: {best_acc:.2f}%")
        else:
            epochs_no_improve += 1

        # Early stopping
        if early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no improvement.")
            break

    print(f"Training complete. Best Accuracy: {best_acc:.2f}%")

    # Load best model weights before returning
    model.load_state_dict(torch.load(save_path))

    # Return model and metrics for plotting/saving
    return model, train_losses, train_accuracies, val_losses, val_accuracies