import torch
from torchvision import transforms
from PIL import Image, ImageDraw   # Add ImageDraw import
from model import create_vit_model
import os

class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

def predict_image(image_path, model_path, device='cpu', highlight_box=None):
    # Check the image file exists before proceeding
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None

    # Load and preprocess the image (same as training/test pipeline)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0).to(device)

    # Load the model and weights
    model = create_vit_model(num_classes=len(class_names), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    # Draw highlight - can customize box position
        # Draw highlight only if tumor is detected
    predicted_class = class_names[pred.item()]
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    if predicted_class != "no_tumor":
        if highlight_box:
            draw.rectangle(highlight_box, outline="red", width=4)
        else:
            W, H = img_draw.size
            x1, y1 = int(W*0.3), int(H*0.3)
            x2, y2 = int(W*0.7), int(H*0.7)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    draw.text((10, 10), predicted_class, fill="yellow")
    img_draw.show()
    img_draw.save("highlighted_output.jpg")

    return predicted_class

# Example usage:
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = r"C:\Users\admin\Desktop\Brain Tumour\dataset\Testing\notumor\Te-pi_0025.jpg"
    model_path = r"C:\Users\admin\Desktop\Brain Tumour\best_vit_model.pth"
    result = predict_image(image_path, model_path, device)
    if result is not None:
        print("Predicted tumor type:", result)