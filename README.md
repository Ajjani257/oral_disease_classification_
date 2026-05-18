# Oral Disease Classification App (DenseNet121)

This project contains:
- A trained PyTorch model file: `densenet_oral_model.pth`
- A training and evaluation notebook: `DenseNet_Oral.ipynb`
- A Streamlit inference app: `app.py`

## What This App Does

The Streamlit app lets a user upload an oral image and then:
- predicts one disease class,
- shows prediction confidence for that image,
- shows class-wise probability distribution,
- displays the notebook's reported test accuracy (`83.32%`) as a reference metric.

## Classes Used

The model predicts one of these 6 classes:
1. Calculus
2. Caries
3. Gingivitis
4. Ulcers
5. Tooth Discoloration
6. Hypodontia

## Notebook Explanation (`DenseNet_Oral.ipynb`)

The notebook trains and evaluates a DenseNet121 classifier.

### 1. Data Setup
- Defines `original_dirs` with class names mapped to source image folders.
- Creates a dataset split structure: `train`, `val`, `test`.
- Copies images into class-wise subfolders for each split.

### 2. Transforms and Preprocessing
- Train transforms include augmentation:
  - random resized crop,
  - horizontal flip,
  - rotation,
  - color jitter,
  - affine transform,
  - normalization (ImageNet mean/std).
- Validation/test transforms are deterministic:
  - resize,
  - center crop,
  - tensor conversion,
  - normalization.

### 3. Dataset and DataLoaders
- Uses `torchvision.datasets.ImageFolder`.
- Creates `train_loader`, `val_loader`, and `test_loader`.

### 4. Model Architecture
- Defines `CustomDenseNet`:
  - backbone: `densenet121`,
  - freezes feature extractor,
  - replaces classifier head with `nn.Linear(in_features, num_classes)`.

### 5. Training
- Uses:
  - optimizer: `AdamW`,
  - loss: `CrossEntropyLoss` with label smoothing,
  - scheduler: `ReduceLROnPlateau`.
- Tracks train/validation loss and accuracy.
- Saves best model using validation loss.

### 6. Evaluation
- Loads best weights.
- Runs test-time augmentation (TTA).
- Reports final test accuracy.
- Shows classification report and confusion matrix.

### 7. Export
- Saves the final weights as:
- `densenet_oral_model.pth`

## App Code Explanation (`app.py`)

The app is designed for single-image inference.

### `CLASS_NAMES`
Defines the exact class order expected by the trained model output layer.

### `CustomDenseNet`
Rebuilds the same model structure used in training:
- DenseNet121 backbone,
- classifier replaced for 6 classes.

### `load_model()`
- Loads `densenet_oral_model.pth`.
- Removes `module.` prefixes if model was saved from `DataParallel`.
- Sets model to eval mode.
- Cached with `@st.cache_resource` so loading is done once.

### `preprocess_image(image)`
Applies validation/test-time transforms compatible with training:
- resize to 256,
- center crop to 224,
- normalize with ImageNet mean/std,
- adds batch dimension.

### `predict_image(model, image)`
- Runs forward pass with `torch.no_grad()`.
- Applies softmax to get probabilities.
- Returns:
  - predicted class label,
  - confidence score,
  - vector of all class probabilities.

### `main()`
- Builds Streamlit UI.
- Displays notebook test accuracy (dataset-level metric).
- Accepts image upload.
- On "Predict":
  - runs inference,
  - displays class and confidence,
  - plots class probability bar chart.

## Important Note About Accuracy vs Confidence

- **Test accuracy (83.32%)** is a dataset-level metric computed over many test images.
- **Confidence** is a per-image probability from the model for one uploaded image.
- They are different metrics and should not be used interchangeably.

## Setup and Run

From the project folder, run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Project Files

- `DenseNet_Oral.ipynb`: training/evaluation workflow
- `densenet_oral_model.pth`: trained model weights
- `app.py`: Streamlit inference app
- `requirements.txt`: Python dependencies
- `README.md`: documentation
