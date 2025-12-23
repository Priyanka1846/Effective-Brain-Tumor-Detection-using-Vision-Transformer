import matplotlib.pyplot as plt

epochs = range(1, 11)

train_loss_ae_vit= [0.3064, 0.1301, 0.1015, 0.0685, 0.06, 0.0512, 0.0505, 0.0375, 0.0303, 0.0318]
val_loss_ae_vit  = [0.3444, 0.1189, 0.0657, 0.0526, 0.0549, 0.0532, 0.0484, 0.0551, 0.0174, 0.0295]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss_ae_vit, label='AE+Vit Training Loss', marker='o',color='red')
plt.plot(epochs, val_loss_ae_vit, label='AE+Vit Validation Loss', marker='x',color='green')
plt.title('AE+Vit Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()