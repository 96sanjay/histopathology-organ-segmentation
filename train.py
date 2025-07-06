import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import os

# Import from our source files
from src import config
from src.dataset import HubmapWsiDataset
from src.engine import train_epoch, validate_epoch
from src.utils import dice_score # utils.py now only contains dice_score

def main():
    """
    Main function to set up and run the k-fold cross-validation training.
    """
    # --- Augmentation & Loss Setup ---
    train_transforms = A.Compose([
        A.Resize(config.PATCH_SIZE, config.PATCH_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.CoarseDropout(p=0.5),
        A.ElasticTransform(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(config.PATCH_SIZE, config.PATCH_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dice_loss = smp.losses.DiceLoss(mode='binary')
    focal_loss = smp.losses.FocalLoss(mode='binary')
    lovasz_loss = smp.losses.LovaszLoss(mode='binary')
    def combined_loss_advanced(y_pred, y_true):
        return (config.DICE_WEIGHT * dice_loss(y_pred, y_true) + 
                config.FOCAL_WEIGHT * focal_loss(y_pred, y_true) + 
                config.LOVASZ_WEIGHT * lovasz_loss(y_pred, y_true))

    # --- K-Fold Training Execution ---
    df = pd.read_csv(config.METADATA_FILE).reset_index(drop=True)
    # Note: A pre-filtering step for corrupted files could be added here if needed.
    
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n========== FOLD {fold+1}/{config.N_SPLITS} ==========")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        
        train_dataset = HubmapWsiDataset(train_df, augmentations=train_transforms)
        val_dataset = HubmapWsiDataset(val_df, augmentations=val_transforms)
        
        class_weights = {'largeintestine': 1, 'spleen': 1, 'kidney': 1, 'prostate': 1, 'lung': 10}
        sample_weights = [class_weights[organ] for organ in train_df['organ']]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        
        model = smp.Unet(config.MODEL_ENCODER, encoder_weights="imagenet", in_channels=3, classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
        
        best_val_dice = 0.0
        for epoch in range(config.NUM_EPOCHS):
            print(f"--- Fold {fold+1}, Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
            
            avg_train_loss = train_epoch(train_loader, model, optimizer, combined_loss_advanced, device)
            avg_val_loss, avg_val_dice = validate_epoch(val_loader, model, combined_loss_advanced, device)
            scheduler.step()
            
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
            
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), f'models/best_model_fold_{fold}.pth')
                print(f"New best model for Fold {fold+1} saved with Dice Score: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()