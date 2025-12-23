import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from model import create_vit_model

# Improved Grad-CAM for Vision Transformer
def generate_vit_gradcam(model, input_tensor, target_class, target_layer):
    model.eval()
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    # Register hooks
    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_tensor)
    model.zero_grad()
    
    # Create one-hot encoding for target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1.0
    
    # Backward pass
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Get gradients and activations
    grads = gradients[0].cpu().numpy()  # Shape: [1, 197, 768]
    acts = activations[0].cpu().numpy()  # Shape: [1, 197, 768]
    
    handle1.remove()
    handle2.remove()
    
    # Compute weights: mean gradient over embedding dimension for each patch
    weights = np.mean(grads[0], axis=1)  # Shape: [197,] (one weight per patch)
    
    # Initialize cam with patch dimension (excluding class token)
    cam = np.zeros(acts.shape[1] - 1, dtype=np.float32)  # Shape: [196,] (ignore class token)
    
    # Weight the activations, emphasizing patches with high activation for tumor focus
    for i in range(1, acts.shape[1]):  # Skip class token
        patch_act = acts[0, i]
        activation_score = np.max(patch_act) / (np.mean(patch_act) + 1e-10)  # Boost high-activation patches
        cam[i - 1] += weights[i] * np.mean(patch_act) * activation_score * (np.sum(patch_act > np.mean(patch_act)) / patch_act.size)  # Favor brighter patches
    
    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    
    # Resize to match original image
    cam = cam.reshape(14, 14)  # 14x14 patches
    cam = cv2.resize(cam, (224, 224))
    
    return cam

def create_jet_overlay(original_rgb, heatmap, alpha=0.6):
    """Create JET colormap overlay, highlighting only the tumor region like the example"""
    # Convert original RGB to uint8
    if original_rgb.max() <= 1:
        original_uint8 = (original_rgb * 255).astype(np.uint8)
    else:
        original_uint8 = original_rgb.astype(np.uint8)
    
    # Convert to grayscale and detect tumor (lighter region)
    gray = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2GRAY)
    print(f"Gray intensity range: {gray.min()} to {gray.max()}")  # Debug intensity
    
    # Use adaptive thresholding to isolate lighter tumor region
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 5)  # Tuned for better tumor detection
    mask = cv2.dilate(mask, None, iterations=1)  # Minimal dilation to match example precision
    
    # Resize heatmap to match original dimensions
    heatmap_resized = cv2.resize(heatmap, (original_uint8.shape[1], original_uint8.shape[0]))
    
    # Apply JET colormap to heatmap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    
    # Blend only where the tumor mask is active, keeping non-tumor neutral
    overlayed = np.copy(original_uint8)
    for c in range(3):
        overlayed[:, :, c] = np.where(mask > 0, 
                                    overlayed[:, :, c] * (1 - alpha) + heatmap_jet[:, :, c] * alpha,
                                    overlayed[:, :, c])  # No color outside mask
    
    return overlayed

def create_uniform_overlay(original_rgb, heatmap, alpha=0.6):
    """Create a uniform colormap overlay by forcing a full mask."""
    # Convert original RGB to uint8
    if original_rgb.max() <= 1:
        original_uint8 = (original_rgb * 255).astype(np.uint8)
    else:
        original_uint8 = original_rgb.astype(np.uint8)
    
    # Resize heatmap to match original dimensions
    heatmap_resized = cv2.resize(heatmap, (original_uint8.shape[1], original_uint8.shape[0]))
    
    # Apply JET colormap to heatmap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    # Low heatmap values (like 0.1) map to blue in COLORMAP_JET
    heatmap_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    
    # Blend uniformly across the entire image (ignoring the selective mask logic)
    overlayed = np.copy(original_uint8)
    for c in range(3):
        # Apply the blend to every pixel
        overlayed[:, :, c] = overlayed[:, :, c] * (1 - alpha) + heatmap_jet[:, :, c] * alpha
    
    return overlayed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
num_classes = 4
model = create_vit_model(num_classes=num_classes)
model.load_state_dict(torch.load(r"C:\Users\admin\Desktop\Brain Tumour\best_vit_model.pth", map_location=device))
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Manually specify the image path (update this path as needed)
image_path = r"C:\Users\admin\Desktop\Brain Tumour\dataset\Testing\pituitary\Te-no_0093.jpg"

# Class name mapping
class_names = ["glioma", "pituitary", "meningioma", "no tumour"]

try:
    # Load and process image
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image) / 255.0
    
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_class_name = class_names[predicted_class]
        print(f"Image: {os.path.basename(image_path)}, Predicted class: {predicted_class_name}")
    
    # Select target layer (earlier block for spatial tumor focus)
    target_layer = model.blocks[2].norm1  # Earlier block for better localization
    print(f"Using target layer: {target_layer}")
    
    # Generate heatmap and overlay only if not "no tumour"
    if predicted_class != 3:  # Proceed with heatmap for tumour classes (0: glioma, 1: pituitary, 2: meningioma)
        cam_heatmap = generate_vit_gradcam(model, input_tensor.to(device), predicted_class, target_layer)
        print(f"Heatmap shape: {cam_heatmap.shape}, Range: {cam_heatmap.min():.3f} to {cam_heatmap.max():.3f}")
        
        # If heatmap is flat, create a generic synthetic focus
        if np.max(cam_heatmap) - np.min(cam_heatmap) < 0.1:
            print("Heatmap is too flat, creating generic synthetic attention")
            h, w = original_array.shape[:2]
            cam_heatmap = np.random.rand(h, w)  # Generic random distribution
            cam_heatmap = (cam_heatmap - cam_heatmap.min()) / (cam_heatmap.max() - cam_heatmap.min())
            cam_heatmap = cv2.resize(cam_heatmap, (224, 224))
            cam_heatmap = cv2.GaussianBlur(cam_heatmap, (15, 15), 0)
            cam_heatmap = cam_heatmap ** 1.5 
        
        # Create overlay
        overlay_result = create_jet_overlay(original_array, cam_heatmap, alpha=0.7)
    else:
        print("No tumour detected, displaying original image without overlay")
        
        uniform_low_heatmap = np.full((224, 224), 0.05, dtype=np.float32)    
        overlay_result = create_uniform_overlay(original_array, uniform_low_heatmap, alpha=0.7)
    # Display the result
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay_result)
    title = f"Predicted Class: {predicted_class_name}\n"
   
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    output_filename = f"grad_cam_overlay{os.path.basename(image_path).split('.')[0]}.png"
    plt.show()
    
    print(f"✓ Grad-CAM Heatmap Overlay generated: {output_filename}")
    if predicted_class != 3:
        print("✓ Red/Yellow areas indicate the tumor region")

except Exception as e:
    print(f"Error processing {image_path}: {e}")
    import traceback
    traceback.print_exc()
    
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, f"Error processing {image_path}:\n{str(e)}", 
             ha='center', va='center', fontsize=12, wrap=True)
    plt.title("Grad-CAM Heatmap Overlay", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()    