# Pneumonia Detection from Chest X-Ray Images using CNN

Overview

This project implements a Stacked Ensemble of Convolutional Neural Networks (CNNs) using TensorFlow and Keras to automatically classify chest X-ray images as either NORMAL or PNEUMONIA.

Instead of relying on a single CNN, multiple models are combined using a stacking strategy to improve prediction accuracy and robustness.
## Dataset

The dataset contains **5,856 JPEG chest X-ray images** (anterior-posterior view) collected from paediatric patients aged 1 to 5 years at a renowned hospital. These images were captured as part of the routine clinical care of the patients.

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Training | 1,341 | 3,875 | 5,216 |
| Testing | 234 | 390 | 624 |
| Validation | 8 | 8 | 16 |
| **Total** | **1,583** | **4,273** | **5,856** |

**Notable challenge**: There is a significant class imbalance in the training data — PNEUMONIA samples outnumber NORMAL samples by a factor of approximately 2.9. This is handled by applying class weights during training.

---

## Approach

### Data Exploration

The dataset was first explored to understand the distribution of images across classes and splits. Sample images from both categories were visualised to identify visual patterns:

- **NORMAL** chest X-rays display clear and well-defined lung fields.
- **PNEUMONIA** chest X-rays exhibit areas of haziness, white patches, or consolidation within the lung regions.

### Image Preprocessing

All images are resized to **150 x 150 pixels** and pixel values are normalised to a [0, 1] range by dividing by 255. This standardisation ensures consistent input dimensions and helps the model converge faster during training.

### Data Augmentation

Methodology (Updated)
Model Architecture (Stacked CNN Ensemble)

Instead of a single model, a stacking framework is used.

🔹 Base CNN Models

Three CNN models are trained independently:

CNN Model A — original architecture
CNN Model B — slightly deeper (more filters/layers)
CNN Model C — lighter model (fewer layers)

Each model learns different feature representations from the same input data.
### Addressing Class Imbalance

Class weights are calculated inversely proportional to the frequency of each class. This ensures the model pays more attention to the underrepresented NORMAL class during training, preventing it from simply predicting PNEUMONIA for every input.

---

## Methodology

### Model Architecture

A **Sequential CNN** with four convolutional blocks is used:

```
Input (150 x 150 x 3)
    |
    Block 1: Conv2D(32) -> BN -> Conv2D(32) -> BN -> MaxPool(2x2) -> Dropout(0.25)
    Block 2: Conv2D(64) -> BN -> Conv2D(64) -> BN -> MaxPool(2x2) -> Dropout(0.25)
    Block 3: Conv2D(128) -> BN -> Conv2D(128) -> BN -> MaxPool(2x2) -> Dropout(0.25)
    Block 4: Conv2D(256) -> BN -> MaxPool(2x2) -> Dropout(0.25)
    |
    Flatten
    Dense(512, ReLU) -> BN -> Dropout(0.5)
    Dense(256, ReLU) -> BN -> Dropout(0.5)
    Dense(1, Sigmoid) -> Output
```

**Why this architecture?**

- **Four convolutional blocks** with increasing filters (32 -> 64 -> 128 -> 256) allow the network to learn progressively complex features — from simple edges in early layers to high-level lung patterns in deeper layers.
- **Batch Normalisation** after convolutions stabilises training by normalising intermediate activations.
- **Dropout** at 25% in convolutional blocks and 50% in dense layers acts as regularisation to prevent overfitting.
- **Flatten** followed by two dense layers provides sufficient capacity for the final classification decision.
- **Sigmoid output** produces a probability score for binary classification.

### Training Parameters

| Setting | Value |
|---------|-------|
| Optimiser | Adam (lr = 0.0001) |
| Loss function | Binary crossentropy |
| Batch size | 32 |
| Epochs | 25 (max) |
| Early stopping | Patience of 5 epochs on validation loss |
| LR reduction | Factor 0.5 when validation loss plateaus for 3 epochs |
| Model checkpoint | Saves best model by validation accuracy |

---

## Findings

### Performance

The model is expected to achieve **85 – 95% test accuracy**. Key evaluation metrics include:

- **Accuracy** — overall correctness of predictions
- **Precision** — proportion of positive predictions that are truly positive
- **Recall / Sensitivity** — proportion of actual pneumonia cases correctly detected (critical in medical diagnostics)
- **F1 Score** — harmonic mean of precision and recall

In a medical setting, **high recall for PNEUMONIA is prioritised** because failing to detect pneumonia (false negative) has far worse consequences than a false alarm (false positive).

### Key Takeaways

1. **Class weighting is critical** — without it, the model defaults to predicting the majority class (PNEUMONIA) for nearly every input, achieving ~74% accuracy without actually learning meaningful features.
2. **Data augmentation prevents overfitting** — with only ~5,200 training images, the model would memorise the training data without augmentation. Random transformations force the model to learn robust, generalisable features.
3. **The validation set is very small** (16 images), so validation metrics may fluctuate. The test set (624 images) provides a more reliable performance estimate.

### Output Files

After running the script, the following files are saved in the `outputs/` directory:

| File | Description |
|------|-------------|
| `pneumonia_cnn.keras` | Final trained model |
| `best_weights.keras` | Best model checkpoint from training |
| `test_results.txt` | Classification report with all metrics |
| `accuracy_loss_curves.png` | Training/validation accuracy and loss plots |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `sample_xrays.png` | Sample images from both classes |

### Future Improvements

1. **Transfer Learning** — using pre-trained networks like VGG16, ResNet50, or DenseNet121 as feature extractors could significantly improve accuracy.
2. **Larger Validation Set** — allocating a portion of training data for validation would yield more stable metrics.
3. **K-Fold Cross-Validation** — would give a more reliable estimate of model performance across different data splits.
4. **Explainability (Grad-CAM)** — visualising which areas of the X-ray the model focuses on would increase transparency and clinical trust.

---

## Project Structure

```
Pneumonia Detection/
├── chest_xray_analysis.py    # Main script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── x-ray_image/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── outputs/                  # Generated after running
```

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the script

```bash
python chest_xray_analysis.py
```
Why Stacking?
Different CNNs capture different patterns
Reduces overfitting compared to a single model
Improves generalisation on unseen data
Combines strengths of multiple learners
Training Strategy (Updated)
Each CNN model is trained separately using the same dataset
Class weights are applied to handle imbalance
Early stopping and learning rate scheduling are used
Predictions from the validation set are collected
These predictions are used to train the meta-model
Findings (Updated for Stacking)
Performance

The stacked model achieves more stable and improved performance compared to a single CNN model.

Key Takeaways
Stacking improves performance — Combining multiple CNN models leads to better predictive accuracy and robustness than any individual model.
Model diversity is crucial — Using CNNs with different complexities allows the system to capture both simple and complex lung patterns.
High recall remains critical — The stacked model improves detection of pneumonia cases, reducing false negatives, which is essential in medical diagnosis.
Class imbalance handling remains important — Class weights prevent bias toward the majority class.
Better generalisation — The ensemble model performs more consistently across different data splits.
Trade-off: computation cost — Training multiple CNNs increases computational time and resource usage.

The script handles everything automatically — from loading data to saving the trained model and evaluation results.

## Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** — model building and training
- **NumPy** — numerical operations
- **Matplotlib** — plotting training curves and sample images
- **Seaborn** — confusion matrix visualisation
- **Scikit-learn** — classification metrics
