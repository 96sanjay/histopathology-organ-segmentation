import os

# --- PATHS ---
BASE_PATH = 'data' 
IMAGES_DIR = os.path.join(BASE_PATH, 'train_images')
METADATA_FILE = os.path.join(BASE_PATH, 'train.csv')

# --- EXPERIMENT CONFIGURATION ---
# A list of dictionaries, where each dictionary defines a model to train.
ARCHITECTURES = [
    {
        "name": "Swin-Unet",
        "encoder": "tu-swin_base_patch4_window12_384",
        "patch_size": 384,
        "batch_size": 4
    },
    {
        "name": "EfficientNet-Unet",
        "encoder": "efficientnet-b5",
        "patch_size": 512,
        "batch_size": 8
    }
]

# --- TRAINING HYPERPARAMETERS ---
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# --- LOSS FUNCTION WEIGHTS ---
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.25
LOVASZ_WEIGHT = 0.25