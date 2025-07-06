import os
import numpy as np
import openslide
import cv2
from torch.utils.data import Dataset
from . import config
from .utils import rle_decode

class HubmapWsiDataset(Dataset):
    def __init__(self, df, augmentations=None, min_mask_area=200):
        self.df = df.reset_index(drop=True)
        self.augmentations = augmentations
        self.min_mask_area = min_mask_area

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(config.IMAGES_DIR, f"{row['id']}.tiff")
        
        try:
            with openslide.OpenSlide(image_path) as slide:
                full_mask = rle_decode(row['rle'], (row['img_height'], row['img_width']))
                
                for _ in range(20): # Retry up to 20 times
                    max_x = row['img_width'] - config.PATCH_SIZE
                    max_y = row['img_height'] - config.PATCH_SIZE
                    x = np.random.randint(0, max_x if max_x > 0 else 0)
                    y = np.random.randint(0, max_y if max_y > 0 else 0)

                    mask_patch = full_mask[y:y+config.PATCH_SIZE, x:x+config.PATCH_SIZE]
                    
                    if mask_patch.sum() > self.min_mask_area:
                        image_patch = slide.read_region((x, y), 0, (config.PATCH_SIZE, config.PATCH_SIZE)).convert('RGB')
                        break
                else: 
                    image_patch = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE, 3), dtype=np.uint8)
                    mask_patch = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.uint8)
        
        except openslide.lowlevel.OpenSlideUnsupportedFormatError:
            print(f"⚠️ Warning: Corrupted file {image_path}. Returning blank patch.")
            image_patch = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE, 3), dtype=np.uint8)
            mask_patch = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.uint8)

        if self.augmentations:
            augmented = self.augmentations(image=np.array(image_patch), mask=mask_patch)
            image, mask = augmented['image'], augmented['mask']
        
        return image, mask
