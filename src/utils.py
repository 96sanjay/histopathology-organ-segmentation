import torch
import numpy as np

def rle_decode(mask_rle, shape):
    """Decodes a Run-Length Encoding string into a 2D mask."""
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, starts + lengths):
        img[lo:hi] = 1
    return img.reshape(shape).T

def dice_score(preds, targets, smooth=1e-6):
    """Calculates the Dice score for a batch of predictions and targets."""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice
