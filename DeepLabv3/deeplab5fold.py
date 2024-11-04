import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import segmentation_models_pytorch as smp
from torchvision.transforms import functional as F
import openslide
import random
from torch.amp import autocast, GradScaler
from tifffile import imwrite
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import torchmetrics
from sklearn.model_selection import KFold

logging.basicConfig(
    filename='deeplabv3_5fold.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# Initialize Weights and Biases
wandb.init(project="deeplabv3-tumor-segmentation-5fold", config={
    "learning_rate": 1e-4,
    "epochs": 100,  # Modified epochs
    "batch_size": 2,  # Modified batch size
    "patch_size": 256,  # Modified patch size
    "num_patches": 30,  # Modified number of patches
    "accumulation_steps": 4,
    "folds": 5  # Number of folds for cross-validation
})

config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
wandb.config.update({"device": device})

PATCH_SIZE = config.patch_size
NUM_PATCHES = config.num_patches
BATCH_SIZE = config.batch_size
NUM_EPOCHS = config.epochs
ACCUMULATION_STEPS = config.accumulation_steps
LEARNING_RATE = config.learning_rate
FOLDS = config.folds

IMAGES_DIR = "/data/Sweta_Histopathology/BRCA_BVM_16X/Dataset_tiledtiff/Images_tiledtiff"
MASKS_DIR = "/data/Sweta_Histopathology/BRCA_BVM_16X/Dataset_tiledtiff/Masks_tiledtiff"
CHECKPOINT_DIR = "checkpoints_deeplabv3_5fold"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class TumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, patch_size=512, num_patches=50, transform=None):  # Modified patch size and number of patches
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith(('.tif', '.tiff', '.svs'))]
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, img_filename)

        try:
            slide = openslide.OpenSlide(img_path)
            mask = openslide.OpenSlide(mask_path)
        except Exception as e:
            logging.error(f"Error opening slide {img_filename}: {e}")
            raise e

        width, height = slide.dimensions
        mask_width, mask_height = mask.dimensions

        if (width, height) != (mask_width, mask_height):
            raise ValueError(f"Image and mask misaligned: {img_filename} - Image size: {width}x{height}, Mask size: {mask_width}x{mask_height}")

        image_patches = []
        mask_patches = []

        for _ in range(self.num_patches):
            x = random.randint(0, width - self.patch_size) if width > self.patch_size else 0
            y = random.randint(0, height - self.patch_size) if height > self.patch_size else 0

            image_patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
            mask_patch = mask.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("L")

            image_patch = np.array(image_patch)
            mask_patch = np.array(mask_patch)

            if self.transform:
                augmented = self.transform(image=image_patch, mask=mask_patch)
                image_patch = augmented['image']
                mask_patch = augmented['mask']

            image_patch = torch.from_numpy(image_patch.transpose(2, 0, 1)).float() / 255.0
            mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float() / 255.0
            mask_patch = torch.where(mask_patch > 0, 1.0, 0.0)

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)

        images = torch.stack(image_patches)
        masks = torch.stack(mask_patches)

        return images, masks

# Use torchmetrics for metrics
iou_metric = torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(device)
f1_score_metric = torchmetrics.F1Score(task='binary', threshold=0.5).to(device)

# Loss criterion (BCEWithLogitsLoss)
criterion = nn.BCEWithLogitsLoss()

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),  # Added brightness and contrast augmentation
    A.ElasticTransform(p=0.5),  # Added elastic transformation
    A.GaussianBlur(p=0.5),  # Added Gaussian blur
    A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-5, 5), p=0.5),  # Added affine transformations
])

full_dataset = TumorDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    patch_size=PATCH_SIZE,
    num_patches=NUM_PATCHES,
    transform=transform
)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold = 1

all_train_losses = []
all_val_losses = []

for train_indices, val_indices in kf.split(full_dataset):
    print(f"Starting Fold {fold}/{FOLDS}")
    logging.info(f"Starting Fold {fold}/{FOLDS}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Changed encoder to resnet18 with ImageNet weights
    model = smp.DeepLabV3(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Implement cyclic learning rate
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=LEARNING_RATE, step_size_up=2000, mode='triangular', cycle_momentum=False)

    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_f1 = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Fold {fold} Epoch [{epoch+1}/{NUM_EPOCHS}] - Training", leave=False)
        for batch_idx, (images, masks) in enumerate(loop):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            images = images.view(-1, *images.shape[2:])
            masks = masks.view(-1, 1, PATCH_SIZE, PATCH_SIZE)

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * ACCUMULATION_STEPS
            train_iou += iou_metric(torch.sigmoid(outputs), masks.int()).item()
            train_f1 += f1_score_metric(torch.sigmoid(outputs), masks.int()).item()
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
            torch.cuda.empty_cache()

        if len(train_loader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_f1 /= len(train_loader)
        all_train_losses.append(train_loss)

        # Log training metrics to Weights and Biases
        wandb.log({
            f"fold_{fold}_epoch": epoch + 1,
            f"fold_{fold}_train_loss": train_loss,
            f"fold_{fold}_train_iou": train_iou,
            f"fold_{fold}_train_f1": train_f1
        })

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0

        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Fold {fold} Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation", leave=False)
            for images, masks in loop:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                images = images.view(-1, *images.shape[2:])
                masks = masks.view(-1, 1, PATCH_SIZE, PATCH_SIZE)

                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += iou_metric(torch.sigmoid(outputs), masks.int()).item()
                val_f1 += f1_score_metric(torch.sigmoid(outputs), masks.int()).item()
                loop.set_postfix(loss=loss.item())
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_f1 /= len(val_loader)
        all_val_losses.append(val_loss)

        # Log validation metrics to Weights and Biases
        wandb.log({
            f"fold_{fold}_epoch": epoch + 1,
            f"fold_{fold}_val_loss": val_loss,
            f"fold_{fold}_val_iou": val_iou,
            f"fold_{fold}_val_f1": val_f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"))
            torch.save(model, os.path.join(CHECKPOINT_DIR, f"best_model_full_fold_{fold}.pth"))
            logging.info(f"Saved Best Model for Fold {fold} at Epoch {epoch+1}")
            wandb.log({f"fold_{fold}_best_val_loss": best_val_loss, f"fold_{fold}_best_model_epoch": epoch + 1})

        log_message = (
            f"Fold {fold} Epoch [{epoch+1}/{NUM_EPOCHS}], "
            f"Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}, Train F1: {train_f1:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val IOU: {val_iou:.4f}, Val F1: {val_f1:.4f}"
        )
        print(log_message)
        logging.info(log_message)

    logging.info(f"Fold {fold} Completed.")
    print(f"Fold {fold} Completed.")
    fold += 1

# Calculate and log the mean and standard deviation of train and validation losses
mean_train_loss = np.mean(all_train_losses)
std_train_loss = np.std(all_train_losses)
mean_val_loss = np.mean(all_val_losses)
std_val_loss = np.std(all_val_losses)

final_log_message = (
    f"Training Completed.\n"
    f"Mean Train Loss: {mean_train_loss:.4f}, Std Train Loss: {std_train_loss:.4f}\n"
    f"Mean Val Loss: {mean_val_loss:.4f}, Std Val Loss: {std_val_loss:.4f}"
)

print(final_log_message)
logging.info(final_log_message)

wandb.finish()
