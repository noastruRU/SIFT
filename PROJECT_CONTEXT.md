# Project Context: Anomalib Nut Defect Detection

## Current State

### Dataset Structure
```
dataset/
├── train/
│   └── good/           (good nut images for training)
└── test/
    ├── good/           (good nut images for testing)
    └── defect/         (defective nut images for testing)
```

### What We Have
- **Capstone_U_Net_Segmentation.ipynb**: A trained U-Net model that:
  1. Downloads COCO-annotated nut dataset from Roboflow
  2. Creates distance maps from COCO segmentation masks
  3. Trains a U-Net to predict distance maps of nuts (cells 1-12)
  4. Tests the model and extracts foreground nuts (cells 13-17)
  
- **Dataset**: Pre-split good/defect images already present in `dataset/train/good` and `dataset/test/`

## Key Insight: Do We Need to Run the Full Capstone Notebook?

**NO** — The Capstone notebook is for **U-Net segmentation training**. For Anomalib, we need a **different approach**:

### What the Capstone Does (NOT needed for Anomalib):
- ✗ Downloads and trains a U-Net segmentation model
- ✗ Creates distance maps from COCO annotations
- ✗ Generates foreground masks and cutouts

### What Anomalib Needs (SEPARATE TASK):
- ✓ Organized train/test splits with good/defect images
- ✓ Train Patchcore on **only good images** (unsupervised anomaly detection)
- ✓ Evaluate on defect images to detect anomalies

**The Capstone notebook is a reference or alternative segmentation approach — not a prerequisite for Anomalib.**

---

## Next Steps for Anomalib Training

### Phase 1: Prepare Anomalib Dataset (Already Done ✓)
- [x] Dataset structure exists: `train/good`, `test/good`, `test/defect`
- [x] Ready for Anomalib training

### Phase 2: Create Jupyter Notebooks for Anomalib Workflow
Create the following as **Jupyter notebooks** (no Python scripts):

1. **`01_anomalib_setup.ipynb`**
   - Install dependencies (anomalib, pytorch-lightning, torch, torchvision)
   - Configure paths and verify dataset structure
   - Quick data validation (count images, check sizes)

2. **`02_anomalib_train_patchcore.ipynb`**
   - Load Patchcore model
   - Train on good images only (`dataset/train/good`)
   - Save trained model checkpoints

3. **`03_anomalib_evaluate.ipynb`**
   - Load trained Patchcore model
   - Test on good and defect images
   - Generate anomaly scores, heatmaps, ROC curves
   - Visualize detection results

4. **`04_anomalib_inference.ipynb`**
   - Inference script for new images
   - Generate anomaly predictions and confidence scores
   - Save results with visualizations

### Phase 3: Optional — Use Capstone for Preprocessing
If needed later (e.g., if new images need segmentation before anomaly detection):
- Cells 1-12 of Capstone: Train U-Net (one-time setup if retraining needed)
- Cells 13-17 of Capstone: Generate foreground masks and cutouts

---

## Capstone Notebook Reference

### Cells to Use (if needed for preprocessing):
- **Cells 1-2**: Download dataset from Roboflow (skip — already have dataset)
- **Cells 5-8**: Create distance maps from COCO annotations (skip — already have split dataset)
- **Cells 9-12**: Train U-Net (skip for now — focus on Anomalib)
- **Cells 13-17**: Test U-Net and extract foreground (use only if need to extract nuts from raw images)

### What NOT to Run:
- Don't run all cells — the Roboflow download and COCO processing are Capstone-specific
- Focus on Anomalib's standard train/test split approach instead

---

## Quick Anomalib Overview

**Patchcore Algorithm**:
- Unsupervised anomaly detection
- Extracts features from a backbone (e.g., ResNet50, WideResNet50)
- Builds a memory bank of normal-sample features during training
- Compares test-sample features to memory bank to detect anomalies
- **No manual masks needed** — only good/defect labels for evaluation

**Training**:
```python
train_loader → Patchcore.train() → Save checkpoint
```

**Evaluation**:
```python
test_good → Anomaly Score ≈ 0 (should be normal)
test_defect → Anomaly Score ≈ 1 (should be anomaly)
```

---

## File Plan

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `Capstone_U_Net_Segmentation.ipynb` | Reference | U-Net segmentation (optional) | Kept |
| `01_anomalib_setup.ipynb` | **Create** | Environment setup & data validation | TODO |
| `02_anomalib_train_patchcore.ipynb` | **Create** | Train Patchcore on good images | TODO |
| `03_anomalib_evaluate.ipynb` | **Create** | Evaluate on test set & visualize | TODO |
| `04_anomalib_inference.ipynb` | **Create** | Inference on new images | TODO |
| `PROJECT_CONTEXT.md` | Reference | This file | ✓ Created |

---

## Commands to Remember

### Install Anomalib (run once in notebook):
```bash
pip install anomalib pytorch-lightning torch torchvision
```

### Verify Dataset:
```python
import os
train_good = len(os.listdir("dataset/train/good"))
test_good = len(os.listdir("dataset/test/good"))
test_defect = len(os.listdir("dataset/test/defect"))
print(f"Train: {train_good}, Test Good: {test_good}, Test Defect: {test_defect}")
```

---

## Next Action

✅ **You are here**: Dataset ready, Capstone understood.

👉 **Next**: Create `01_anomalib_setup.ipynb` — install packages, validate dataset, confirm paths.

