import matplotlib.pyplot as plt

epochs = range(1, 11)

# Resnet metrics 
train_acc_resnet = [85.77, 93.87, 96.5, 96.91, 97.32, 97.39, 97.65, 97.9, 97.97, 97.6]
val_acc_resnet   = [92.14, 93.59, 95.8, 96.26, 97.18, 97.48, 98.25, 97.51, 97.48, 98.47]
train_loss_resnet= [0.363, 0.175, 0.1057, 0.0924, 0.079, 0.0734, 0.0676, 0.0617, 0.0666, 0.0709]
val_loss_resnet  = [0.2288, 0.1914, 0.1048, 0.1055, 0.0777, 0.0777, 0.0476, 0.0579, 0.0683, 0.0456]

# AE+ViT metrics
train_acc_ae_vit = [89.2, 95.5, 96.6, 97.64, 98.25, 98.16, 98.6, 98.76, 99.0, 98.93]
val_acc_ae_vit   = [88.41, 95.42, 97.86, 98.25, 97.94, 98.02, 98.4, 98.25, 99.24, 99.31]
train_loss_ae_vit= [0.3064, 0.1301, 0.1015, 0.0685, 0.06, 0.0512, 0.0505, 0.0375, 0.0303, 0.0318]
val_loss_ae_vit  = [0.3444, 0.1189, 0.0657, 0.0526, 0.0549, 0.0532, 0.0484, 0.0551, 0.0174, 0.0295]

# ViT metrics
train_acc_vit = [87.49, 94.69, 96.55, 97.28, 97.79, 97.78, 98.13, 98.33, 98.49, 98.27]
val_acc_vit   = [90.28, 94.51, 96.83, 97.26, 97.56, 97.75, 98.33, 97.88, 98.36, 98.89]
train_loss_vit= [0.335, 0.153, 0.104, 0.080, 0.070, 0.062, 0.059, 0.050, 0.048, 0.051]
val_loss_vit  = [0.287, 0.155, 0.085, 0.079, 0.066, 0.065, 0.048, 0.057, 0.043, 0.038]

# Graph 1: Training Accuracy
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc_resnet, label='Resnet Train Accuracy', marker='o')
plt.plot(epochs, train_acc_ae_vit, label='AE+Vit Train Accuracy', marker='x')
plt.plot(epochs, train_acc_vit, label='ViT Train Accuracy', marker='s')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Graph 2: Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(epochs, val_acc_resnet, label='Resnet Validation Accuracy', marker='o')
plt.plot(epochs, val_acc_ae_vit, label='AE+ViT Validation Accuracy', marker='x')
plt.plot(epochs, val_acc_vit, label='ViT Validation Accuracy', marker='s')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Graph 3: Training Loss
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss_resnet, label='Resnet Train Loss', marker='o')
plt.plot(epochs, train_loss_ae_vit, label='AE+Vit Train Loss', marker='x')
plt.plot(epochs, train_loss_vit, label='ViT Train Loss', marker='s')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Graph 4: Validation Loss
plt.figure(figsize=(8,5))
plt.plot(epochs, val_loss_resnet, label='Resnet Validation Loss', marker='o')
plt.plot(epochs, val_loss_ae_vit, label='AE+Vit Validation Loss', marker='x')
plt.plot(epochs, val_loss_vit, label='ViT Validation Loss', marker='s')
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()