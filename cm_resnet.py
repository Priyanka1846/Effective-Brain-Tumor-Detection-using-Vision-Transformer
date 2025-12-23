import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
cm = np.array([
    [290, 10, 0, 0],
    [0, 294, 12, 0],
    [0, 0, 405, 0],
    [1, 3, 0, 296]
])

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Resnet Transformer - Confusion Matrix")
plt.show()