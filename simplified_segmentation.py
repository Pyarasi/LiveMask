
import torch
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# Load the lightweight FCN ResNet50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fcn_resnet50(pretrained=True).to(device)
model.eval()

# Define image transformations
transform = T.Compose([
    T.ToTensor(),  # Convert to tensor
])

# Function to perform segmentation
def segment_image(image):
    # Transform the input image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    # Get the segmentation mask
    mask = output.argmax(0).byte().cpu().numpy()
    
    return mask

# Function to overlay the segmentation mask on the image
def overlay_mask_on_image(image, mask):
    # Convert the grayscale mask to color for overlay
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel
    
    # Overlay mask with some transparency
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlay

# Example usage (replace 'image_path' with your actual image path)
image_path = 'data/processed/wider_face/val/0_Parade_Parade_0_156.jpg'
image = cv2.imread(image_path)
mask = segment_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
overlayed_image = overlay_mask_on_image(image, mask)

cv2.imshow("Segmented Image", overlayed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
