import torch
import torch.nn as nn
from torchvision import models, transforms
from data_loader import get_data_loaders
from train import train_model
from evaluate import evaluate_model

def create_resnet18_model(num_classes, device):
    # Load ImageNet-pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Adapt final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model

def main():
    train_dir = r'C:\Users\admin\Desktop\Brain Tumour\dataset\Training'
    test_dir = r'C:\Users\admin\Desktop\Brain Tumour\dataset\Testing'
    batch_size = 16
    image_size = 224
    epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_loader, test_loader, class_names = get_data_loaders(
        train_dir, test_dir, batch_size, image_size,
        train_transform=train_transform, test_transform=test_transform
    )
    print(f"Classes: {class_names}")

    model = create_resnet18_model(num_classes=len(class_names), device=device)
    model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model, train_loader, test_loader, device, epochs=epochs,
    save_path='best_resnet18_model.pth',
    metrics_dir="metrics_cnn"
)

    evaluate_model(model, test_loader, class_names, device)

if __name__ == '__main__':
    main()