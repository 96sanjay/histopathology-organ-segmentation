# Advanced Histopathology Segmentation with Transformers

This repository contains a research-level pipeline for segmenting Functional Tissue Units (FTUs) in high-resolution, multi-organ histopathology images from the HuBMAP Kaggle competition. The project employs a state-of-the-art Swin Transformer based U-Net and a suite of advanced techniques to achieve a robust and accurate segmentation model.

## Key Features & Techniques

- **Full-Resolution Patching**: Implements a data pipeline using `openslide` to handle gigapixel Whole-Slide Images (WSI) by extracting high-resolution patches directly from regions with tissue ("Hard Positive Mining"), avoiding wasted computation on empty background.

- **State-of-the-Art Architecture**: Employs a U-Net architecture with a powerful, pre-trained **Swin Transformer** backbone to effectively capture both local and global contextual features.

- **Advanced Training Strategies**:
    - **3-Component Loss**: A combined Dice, Focal, and Lovász loss function to simultaneously optimize for mask area, hard-to-classify pixels, and clean boundaries.
    - **Weighted Sampling**: Implements a `WeightedRandomSampler` to address the severe class imbalance between different organs.
    - **k-Fold Cross-Validation**: Trains an ensemble of 5 models for a more robust and higher-scoring final result.

- **Systematic Optimization**:
    - **Hyperparameter Tuning**: Uses the `Optuna` framework to systematically find the optimal learning rate and loss function weights.
    - **Pseudo-Labeling**: Implements a semi-supervised strategy to generate new training data from the test set, further improving model generalization.
    - **Inference-Time Improvements**: Leverages Test-Time Augmentation (TTA) and post-processing (removing small objects, filling holes) to refine final predictions.

## Project Structure
## How to Run

1.  Place the downloaded Kaggle data into a `data/` folder in the project root.
2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install pandas torch scikit-learn openslide-python opencv-python tqdm albumentations segmentation-models-pytorch optuna
    ```
4.  Execute the training script:
    ```bash
    python train.py
    ```
