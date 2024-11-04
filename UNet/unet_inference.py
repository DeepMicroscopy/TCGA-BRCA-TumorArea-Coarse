import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast
import torchmetrics
import openslide
from tqdm import tqdm

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models for ensemble
model_paths = [
    "/data/Sweta_Histopathology/BRCA_BVM_16X/FINAL/unet5fold/checkpoints_unet_5fold/best_model_full_fold_1.pth",
    "/data/Sweta_Histopathology/BRCA_BVM_16X/FINAL/unet5fold/checkpoints_unet_5fold/best_model_full_fold_2.pth",
    "/data/Sweta_Histopathology/BRCA_BVM_16X/FINAL/unet5fold/checkpoints_unet_5fold/best_model_full_fold_3.pth",
    "/data/Sweta_Histopathology/BRCA_BVM_16X/FINAL/unet5fold/checkpoints_unet_5fold/best_model_full_fold_4.pth",
    "/data/Sweta_Histopathology/BRCA_BVM_16X/FINAL/unet5fold/checkpoints_unet_5fold/best_model_full_fold_5.pth"
]

models = [torch.load(path, map_location=device) for path in model_paths]
for model in models:
    model.eval()

# Define metrics for evaluation
iou_metric = torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(device=device)
f1_score_metric = torchmetrics.F1Score(task='binary', threshold=0.5).to(device=device)

# Function to perform sliding window inference on an image using ensemble models and calculate metrics
def predict_image_ensemble_sliding_window(image_path, mask_path, models, iou_metric, f1_score_metric, window_size=512, overlap=0.5):
    # Load the image using OpenSlide
    slide = openslide.OpenSlide(image_path)
    width, height = slide.dimensions
    image = slide.read_region((0, 0), 0, (width, height)).convert("RGB")
    image = np.array(image)

    # Load the corresponding mask
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)

    # Initialize empty prediction mask
    prediction_mask = np.zeros((height, width), dtype=np.float32)
    count_mask = np.zeros((height, width), dtype=np.float32)

    # Sliding window parameters
    step_size = int(window_size * (1 - overlap))

    # Loop over the image with a sliding window
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # Extract the window
            x_end = min(x + window_size, width)
            y_end = min(y + window_size, height)
            window = image[y:y_end, x:x_end, :]

            # Padding if the window is smaller than the desired size
            pad_bottom = window_size - (y_end - y)
            pad_right = window_size - (x_end - x)
            if pad_bottom > 0 or pad_right > 0:
                window = np.pad(window, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant', constant_values=0)

            # Convert window to tensor
            window_tensor = torch.from_numpy(window.transpose(2, 0, 1)).float() / 255.0
            window_tensor = window_tensor.unsqueeze(0).to(device)

            # Perform prediction using ensemble of models
            fold_predictions = []
            with torch.no_grad():
                with autocast():
                    for model in models:
                        output = model(window_tensor)
                        prediction = torch.sigmoid(output)
                        fold_predictions.append(prediction.squeeze().cpu().numpy())

            # Average predictions across folds
            avg_prediction = np.mean(fold_predictions, axis=0)

            # Remove padding from prediction if applied
            if pad_bottom > 0 or pad_right > 0:
                avg_prediction = avg_prediction[:y_end - y, :x_end - x]

            # Update the prediction mask
            prediction_mask[y:y_end, x:x_end] += avg_prediction
            count_mask[y:y_end, x:x_end] += 1

    # Average the overlapping areas
    prediction_mask = prediction_mask / count_mask
    prediction_mask = (prediction_mask > 0.5).astype(np.float32)

    # Convert prediction and mask to tensors for metric calculation
    prediction_tensor = torch.from_numpy(prediction_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
    mask_tensor = torch.where(mask_tensor > 0, 1.0, 0.0)

    # Calculate metrics
    iou = iou_metric(prediction_tensor, mask_tensor.int()).item()
    f1 = f1_score_metric(prediction_tensor, mask_tensor.int()).item()

    return iou, f1

# Function to perform inference on a folder of images and masks
def predict_folder_ensemble(image_folder, mask_folder, models, iou_metric, f1_score_metric, log_file_path, window_size=512, overlap=0.5):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.tiff', '.tif'))]
    iou_scores = []
    f1_scores = []

    with open(log_file_path, 'w') as log_file:
        log_file.write("Image, IoU, F1 Score\n")
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_folder, image_file)
            mask_path = os.path.join(mask_folder, image_file)

            if os.path.exists(mask_path):
                iou, f1 = predict_image_ensemble_sliding_window(image_path, mask_path, models, iou_metric, f1_score_metric, window_size, overlap)
                iou_scores.append(iou)
                f1_scores.append(f1)
                log_file.write(f"{image_file}, {iou:.4f}, {f1:.4f}\n")
                print(f"Image: {image_file}, IoU: {iou:.4f}, F1 Score: {f1:.4f}")
            else:
                print(f"Mask not found for image: {image_file}")

        # Calculate mean and standard deviation for IoU and F1 scores
        mean_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        # Log and print the mean and standard deviation
        log_file.write(f"\nMean IoU: {mean_iou:.4f}, Std Dev IoU: {std_iou:.4f}\n")
        log_file.write(f"Mean F1 Score: {mean_f1:.4f}, Std Dev F1 Score: {std_f1:.4f}\n")
        print(f"\nMean IoU: {mean_iou:.4f}, Std Dev IoU: {std_iou:.4f}")
        print(f"Mean F1 Score: {mean_f1:.4f}, Std Dev F1 Score: {std_f1:.4f}")

# Example usage of the folder prediction function
image_folder = "/data/Sweta_Histopathology/BRCA_BVM_16X/Dataset_tiledtiff/images_tiledtiff_test"
mask_folder = "/data/Sweta_Histopathology/BRCA_BVM_16X/Dataset_tiledtiff/masks_tiledtiff_test"
log_file_path = "ensemble_inference_results_coarse.log"
predict_folder_ensemble(image_folder, mask_folder, models, iou_metric, f1_score_metric, log_file_path)
