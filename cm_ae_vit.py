import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
cm = np.array([
    [296, 4, 0, 0],
    [3, 299, 1, 3],
    [0, 1, 299, 0],
    [0, 1, 0, 404]
])

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("AE+ViT Confusion Matrix")
plt.show()