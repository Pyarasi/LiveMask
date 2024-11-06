import glob #for searching images in directories
import os #directory path operations
import time
import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import cv2 #for webcam related operations
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class FaceDataset(Dataset):  #for loading images from the dataset for training
    def __init__(self, root_dir, split='train'):  # split exists to choose between training and test data
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
        
    def __len__(self):  #returns the number of images
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image at path {img_path} could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #converts to RGB
        image = F.to_tensor(image)  #converts to tensor (numerical representation of images, 3 dimensional array)
        
        _, height, width = image.shape
        target = {
            'boxes': torch.tensor([[0, 0, width, height]], dtype=torch.float32),  #where is the image
            'labels': torch.tensor([1], dtype=torch.int64),  #what is the image
            'masks': torch.zeros((1, height, width), dtype=torch.uint8)  #how is the image filling the box
        }
        
        return image, target

def get_backbone():  #backbone is a cnn that does preliminary screening of images, does feature extraction on the base level
    resnet_backbone = resnet18(weights=None)

    weights_path = "models/resnet18-f37072fd.pth"  
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path)
        resnet_backbone.load_state_dict(state_dict)
    else:
        print("Warning: Pretrained weights not found. Initializing model without pretrained weights.")

    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    in_channels_list = [128, 256, 512]
    out_channels = 256

    backbone = BackboneWithFPN(
        resnet_backbone,
        return_layers,
        in_channels_list,
        out_channels,
    )
    return backbone

def get_model(num_classes):
    backbone = get_backbone()

    anchor_sizes = ((32,), (64,), (128,), (256,))  #creates starter boxes, as a baseline for model to search for in the total image. Like stencils.
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = MultiScaleRoIAlign(  #adjusts each box to a fixed size for the model to have uniformity
        featmap_names=['0', '1', '2', 'pool'],
        output_size=7,
        sampling_ratio=2,
    )

    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', 'pool'],
        output_size=14,
        sampling_ratio=2,
    )

    model = MaskRCNN(  #combines everything into a single model
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
    )
    return model

def print_feature_map_names(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  
    with torch.no_grad():
        features = model.backbone(dummy_input)  
        print("Feature map names:", list(features.keys()))

def train_model():
    dataset = FaceDataset("data/processed/wider_face", split='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn    #loads data in batches (2 images at a time)
    )

    model = get_model(num_classes=2)  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    print_feature_map_names(model, device)

    model.train()

    optimizer = torch.optim.SGD(  #uses stochastic gradient descent, learning rate = 0.005, momentum and weight decay as seen below
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], " # 5 epochs, 488 images, 2 per batch = 244 batches
                  f"Loss: {total_loss.item():.4f}, "  # loss minimizing function is used
                  f"Progress: {percent_completed:.2f}%, "
                  f"Elapsed Time: {elapsed_time/60:.2f} minutes")

        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Total Loss: {epoch_loss:.4f}")
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    train_model()