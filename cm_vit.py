import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
cm = np.array([
    [293, 7, 0, 0],
    [2, 297, 7, 1],
    [0, 0, 352, 0],
    [0, 2, 0, 350]
])

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("ViT Confusion Matrix")
plt.show()