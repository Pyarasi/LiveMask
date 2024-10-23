import glob
import os
import time
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, f'*.{ext}')))
            self.image_paths.extend(glob.glob(os.path.join(self.root_dir, f'*.{ext.upper()}')))
        
        print(f"Looking for images in: {self.root_dir}")
        print(f"Found {len(self.image_paths)} images.")
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.root_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image at path {img_path} could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)
        
        _, height, width = image.shape
        target = {
            'boxes': torch.tensor([[0, 0, width, height]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'masks': torch.zeros((1, height, width), dtype=torch.uint8)
        }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

from torchvision.models.detection import maskrcnn_resnet50_fpn

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_model():
    dataset = FaceDataset("data/processed/wider_face", split='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    model = get_model(num_classes=2)  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

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
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            current_step += 1

            percent_completed = (current_step / total_steps) * 100
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {total_loss.item():.4f}, "
                  f"Progress: {percent_completed:.2f}%, "
                  f"Elapsed Time: {elapsed_time/60:.2f} minutes")

        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Total Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), 'fine_tuned_mask_rcnn.pth')
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
    print("Model weights saved as 'fine_tuned_mask_rcnn.pth'.")

if __name__ == "__main__":
    train_model()