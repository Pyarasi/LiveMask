import glob
import os
import time
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision.models import resnet18
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# Step 1: Define Custom Dataset
class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        # Match multiple image extensions
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, f'*.{ext}')))
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, f'*.{ext.upper()}')))
        
        # Debugging information
        print(f"Looking for images in: {self.root_dir}")
        print(f"Found {len(self.image_paths)} images.")
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.root_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read image in RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image at path {img_path} could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)
        
        # Example boxes, labels, masks
        _, height, width = image.shape
        target = {
            'boxes': torch.tensor([[0, 0, width, height]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'masks': torch.zeros((1, height, width), dtype=torch.uint8)
        }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# Step 2: Get the Backbone and Pre-Trained Weights
def get_backbone():
    resnet_backbone = resnet18(weights=None)
    # If you have access to pretrained weights, uncomment the following lines
    # from torchvision.models import ResNet18_Weights
    # resnet_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    weights_path = "models/resnet18-f37072fd.pth"  # Path to your local weights file
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path)
        resnet_backbone.load_state_dict(state_dict)
    else:
        print("Warning: Pretrained weights not found. Initializing model without pretrained weights.")

    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    in_channels_list = [128, 256, 512]
    out_channels = 256

    # Create the FPN backbone (defaults to adding the extra 'pool' feature map)
    backbone = BackboneWithFPN(
        resnet_backbone,
        return_layers,
        in_channels_list,
        out_channels,
    )
    return backbone

# Step 3: Initialize Mask R-CNN with the Lightweight Backbone and Custom Anchor Generator
def get_model(num_classes):
    backbone = get_backbone()

    # Define custom anchor generator for 4 feature maps
    anchor_sizes = ((32,), (64,), (128,), (256,))  # Adjusted for 4 feature maps
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Define custom RoI poolers for 4 feature maps
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', 'pool'],
        output_size=7,
        sampling_ratio=2,
    )

    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', 'pool'],
        output_size=14,
        sampling_ratio=2,
    )

    # Create the model with custom settings
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
    )
    return model

# Function to print feature map names for debugging
def print_feature_map_names(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Single batched tensor
    with torch.no_grad():
        features = model.backbone(dummy_input)  # Pass tensor directly
        print("Feature map names:", list(features.keys()))

# Step 4: Train the Model with Progress Indicators
def train_model():
    dataset = FaceDataset("data/processed/wider_face", split='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    model = get_model(num_classes=2)  # Background + Face
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Print feature map names to verify
    print_feature_map_names(model, device)
    # Expected output: Feature map names: ['0', '1', '2', 'pool']

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = 5
    total_steps = num_epochs * len(dataloader)
    current_step = 0

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move images and targets to the device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            current_step += 1

            # Calculate percentage completed
            percent_completed = (current_step / total_steps) * 100

            # Print progress
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {total_loss.item():.4f}, "
                  f"Progress: {percent_completed:.2f}%, "
                  f"Elapsed Time: {elapsed_time/60:.2f} minutes")

        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Total Loss: {epoch_loss:.4f}")
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    train_model()