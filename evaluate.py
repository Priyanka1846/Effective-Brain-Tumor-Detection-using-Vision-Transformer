import torch
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    correct_examples, incorrect_examples = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            images = images.cpu()
            labels = labels.cpu()
            preds = preds.cpu()

            for img, true, pred in zip(images, labels, preds):
                if true == pred:
                    correct_examples.append((img, true, pred))
                else:
                    incorrect_examples.append((img, true, pred))

    # Print Confusion Matrix and Classification Report
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Sample examples to visualize, safely up to 4
    correct_samples = random.sample(correct_examples, min(4, len(correct_examples)))
    incorrect_samples = random.sample(incorrect_examples, min(4, len(incorrect_examples)))

    def plot_examples(samples, class_names, title):
        plt.figure(figsize=(8, 8))
        for i, (img, true, pred) in enumerate(samples):
            plt.subplot(2, 2, i + 1)

            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img_np = img.permute(1, 2, 0).numpy()
                cmap = 'gray' if img_np.shape[-1] == 1 else None
            else:
                img_np = img.numpy().squeeze()
                cmap = 'gray' if img_np.ndim == 2 else None

            plt.imshow(img_np, cmap=cmap)
            plt.title(f"True: {class_names[true]}\nPred: {class_names[pred]}")
            plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    if correct_samples:
        plot_examples(correct_samples, class_names, "Correct Classifications")
    else:
        print("No correct classification examples to display.")

    if incorrect_samples:
        plot_examples(incorrect_samples, class_names, "Incorrect Classifications")
    else:
        print("No incorrect classification examples to display.")