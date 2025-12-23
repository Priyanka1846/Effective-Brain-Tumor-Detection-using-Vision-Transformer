import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

class SimpleCNN(nn.Module):
    def _init_(self, num_classes):
        super(SimpleCNN, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def predict_image(image_path, model_path, class_names, device='cpu'):
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]

if __name__ == "_main_":
    image_path = r"C:\Users\admin\Desktop\Brain Tumour\dataset\Testing\notumor\Te-pi_0025.jpg"
    model_path = r"C:\Users\admin\Desktop\Brain tumor Detection\best_cnn_model.pth"                        
    
    class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    prediction = predict_image(image_path, model_path, class_names, device)
    
    if prediction is not None:
        print(f"Predicted tumor type: {prediction}")