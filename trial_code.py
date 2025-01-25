import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToTensor, Resize
import os

# Define transforms
def get_transform(train):
    """
    Apply transforms to the image only (not the target).
    """
    transforms = []
    transforms.append(ToTensor())  # Convert PIL image to tensor
    if train:
        transforms.append(Resize((600, 600)))  # Resize images for training
    return Compose(transforms)

# Extract bounding boxes and labels from the target dictionary
def extract_boxes_and_labels(target):
    """
    Extract bounding boxes and labels from the VOC target dictionary.
    """
    boxes = []
    labels = []
    for obj in target["annotation"]["object"]:
        # Get class label
        label = obj["name"]
        labels.append(label)
        
        # Get bounding box coordinates
        bbox = obj["bndbox"]
        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes, labels

# Custom dataset class
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year="2012", image_set="train", transforms=None):
        self.voc = VOCDetection(root, year=year, image_set=image_set, download=True)
        self.transforms = transforms
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
    
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        # Load image and target
        img, target = self.voc[idx]
        
        # Extract bounding boxes and labels from the target dictionary
        boxes, labels = extract_boxes_and_labels(target)
        
        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor([self.classes.index(label) for label in labels], dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        # Apply transforms to the image only
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

# Custom collate function
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

# Load datasets
root = "./data"
train_dataset = VOCDataset(root, year="2012", image_set="train", transforms=get_transform(train=True))
val_dataset = VOCDataset(root, year="2012", image_set="val", transforms=get_transform(train=False))

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Define Faster R-CNN model
backbone = torchvision.models.vgg16(pretrained=True).features
backbone.out_channels = 512
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

model = FasterRCNN(
    backbone,
    num_classes=21,  # 20 classes + background
    rpn_anchor_generator=anchor_generator
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for images, targets in train_dataloader:
        # Move images and targets to the device (GPU or CPU)
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumulate loss for logging
        epoch_loss += losses.item()
    
    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")

# Validation loop
model.eval()
val_loss = 0.0

with torch.no_grad():
    for images, targets in val_dataloader:
        # Move images and targets to the device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Accumulate loss for logging
        val_loss += losses.item()

# Print the average validation loss
print(f"Validation Loss: {val_loss/len(val_dataloader):.4f}")
