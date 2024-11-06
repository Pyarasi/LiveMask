
import torch
import numpy as np
from sklearn.metrics import jaccard_score

# Function to calculate pixel accuracy
def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum()
    total = true_mask.size
    return correct / total

# Function to calculate IoU score for each class and then average
def mean_iou(pred_mask, true_mask, num_classes=2):
    iou_scores = []
    for cls in range(num_classes):
        pred = (pred_mask == cls).astype(int)
        true = (true_mask == cls).astype(int)
        intersection = (pred & true).sum()
        union = (pred | true).sum()
        if union > 0:
            iou_scores.append(intersection / union)
    return np.mean(iou_scores)

# Model prediction and evaluation function
def evaluate_segmentation(model, images, masks):
    model.eval()
    pixel_accs = []
    iou_scores = []

    with torch.no_grad():
        for img, true_mask in zip(images, masks):
            # Preprocess the image
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Predict the segmentation mask
            output = model(img_tensor)['out'][0]
            pred_mask = output.argmax(0).cpu().numpy()
            
            # Compute pixel accuracy and mean IoU
            pixel_accs.append(pixel_accuracy(pred_mask, true_mask))
            iou_scores.append(mean_iou(pred_mask, true_mask))
    
    avg_pixel_accuracy = np.mean(pixel_accs)
    avg_iou_score = np.mean(iou_scores)
    
    return avg_pixel_accuracy, avg_iou_score

# Example usage (requires images and ground truth masks loaded as 'images' and 'masks' lists)
# avg_pixel_accuracy, avg_iou_score = evaluate_segmentation(model, images, masks)
# print(f"Pixel Accuracy: {avg_pixel_accuracy:.4f}, Mean IoU: {avg_iou_score:.4f}")
