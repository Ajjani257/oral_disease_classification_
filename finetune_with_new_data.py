"""
Fine-tune DenseNet121 Oral Disease Model with New Data
=======================================================
Designed to run on Google Colab (GPU runtime).

This script:
  1. Extracts zip files (archive 1, 3, 6) and organises images into 8 classes.
  2. Downloads 3 extra Kaggle datasets for Oral Cancer / Non Cancer classes.
  3. Loads the pre-trained 6-class model (densenet_oral_model_improved.pth).
  4. Expands the classifier head from 6 → 8 classes, preserving existing weights.
  5. Fine-tunes using a 2-phase strategy similar to the original training.
  6. Saves the result as 'densenet_oral_model_finetuned.pth'.

How to use on Colab:
  1. Upload the following files to /content/:
       - archive (1).zip
       - archive (3).zip
       - archive (6).zip
       - densenet_oral_model_improved.pth
       - finetune_with_new_data.py
  2. Run:  !pip install kagglehub huggingface_hub
  3. Run:  !python finetune_with_new_data.py
"""

import os
import shutil
import random
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
BATCH_SIZE = 32
PHASE1_EPOCHS = 10      # Frozen backbone warm‑up
PHASE2_EPOCHS = 50      # Fine‑tune with unfrozen last 2 DenseBlocks
PATIENCE = 12           # Early stopping patience (applies in Phase 2)
CLASSIFIER_LR = 1e-4    # Lower LR for fine‑tuning (was 1e-3 for initial training)
BACKBONE_LR = 1e-5      # Lower LR for fine‑tuning (was 1e-4 for initial training)
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2

# Paths – adjust if you mount Google Drive
SAVE_TO_DRIVE = True  # Set to True to save the model directly to Google Drive

if SAVE_TO_DRIVE and os.path.exists('/content/drive/MyDrive'):
    BASE_PATH = "/content/drive/MyDrive"
    print("\n✅ Google Drive detected! Model will be saved to your Drive.")
else:
    print("\n⚠️ Google Drive not mounted. Model will be saved locally to /content.")
    print("   To save to Drive, run this in a Colab cell BEFORE running this script:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    BASE_PATH = "/content"

PRETRAINED_MODEL_PATH = os.path.join("/content", "densenet_oral_model_improved.pth")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "densenet_oral_model_finetuned.pth")
WORK_DIR = os.path.join("/content", "dataset_finetune")
EXTRACT_DIR = os.path.join("/content", "_zip_extracted")

ZIP_FILES = [
    os.path.join("/content", "archive (1).zip"),
    os.path.join("/content", "archive (3).zip"),
    os.path.join("/content", "archive (6).zip"),
]

# Original 6 classes (order must match the trained model)
OLD_CLASSES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Tooth Discoloration",
    "Ulcers",
]

# New classes to add (appended after existing ones)
NEW_CLASSES = [
    "Non Cancer",
    "Oral Cancer",
]

ALL_CLASSES = OLD_CLASSES + NEW_CLASSES  # 8 classes total

# Extra Kaggle datasets to download automatically (Oral Cancer data)
KAGGLE_DATASETS = [
    "shivam17299/oral-cancer-lips-and-tongue-images",
    "samxengineer/augumented-oral-cancer-dataset",
    "obulisainaren/multi-cancer",
]

# Hugging Face datasets
HF_DATASETS = [
    "HRruiH/Dataset-of-oral-mucosal-diseases",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────────────────
# 1. Download Kaggle Datasets, Extract Zips & Organise Data
# ────────────────────────────────────────────────────────────
def download_kaggle_datasets():
    """Download extra datasets from Kaggle via kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("  ⚠️  kagglehub not installed. Run: pip install kagglehub")
        print("  Skipping Kaggle dataset downloads.")
        return []

    downloaded_paths = []
    for slug in KAGGLE_DATASETS:
        print(f"  📥 Downloading {slug} …")
        try:
            path = kagglehub.dataset_download(slug)
            print(f"     → {path}")
            downloaded_paths.append(path)
        except Exception as e:
            print(f"  ⚠️  Failed to download {slug}: {e}")
    return downloaded_paths


def download_hf_datasets():
    """Download extra datasets from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ⚠️  huggingface_hub not installed. Run: pip install huggingface_hub")
        print("  Skipping Hugging Face dataset downloads.")
        return []

    downloaded_paths = []
    for repo_id in HF_DATASETS:
        print(f"  🤗 Downloading {repo_id} from Hugging Face …")
        try:
            path = snapshot_download(repo_id=repo_id, repo_type="dataset")
            print(f"     → {path}")
            downloaded_paths.append(path)
        except Exception as e:
            print(f"  ⚠️  Failed to download {repo_id}: {e}")
    return downloaded_paths


def extract_zips():
    """Extract zip files into a temporary directory."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    for zf_path in ZIP_FILES:
        if not os.path.exists(zf_path):
            print(f"  ⚠️  Zip not found, skipping: {zf_path}")
            continue
        print(f"  📦 Extracting {os.path.basename(zf_path)} …")
        with zipfile.ZipFile(zf_path, "r") as zf:
            zf.extractall(EXTRACT_DIR)
    print("  ✅ Extraction complete.\n")


def collect_images_from_dir(directory, extensions=(".jpg", ".jpeg", ".png")):
    """Walk a directory tree and return list of image file paths."""
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                images.append(os.path.join(root, f))
    return images


def safe_pil_loader(path: str):
    """Load an image and always convert to RGB, handling palette/RGBA images cleanly."""
    from PIL import Image
    img = Image.open(path)
    return img.convert("RGBA").convert("RGB")


# Mapping: each entry is (source_directory_relative_to_EXTRACT_DIR, target_class_name)
SOURCE_MAP = [
    # ── archive (1) ──
    ("Dental diseases_Model/Caries", "Caries"),
    ("Dental diseases_Model/Gingivitis", "Gingivitis"),
    ("Dental diseases_Model/Mouth Ulcer", "Ulcers"),
    ("Dental diseases_Model/Tooth_discoloration_augmented", "Tooth Discoloration"),
    ("Dental diseases_Model/Hypodontia", "Hypodontia"),

    # ── archive (3) ──
    ("Oral Cancer/Oral Cancer Dataset/CANCER", "Oral Cancer"),
    ("Oral Cancer/Oral Cancer Dataset/NON CANCER", "Non Cancer"),
    ("Oral cancer Dataset 2.0/OC Dataset kaggle new/CANCER", "Oral Cancer"),
    ("Oral cancer Dataset 2.0/OC Dataset kaggle new/NON CANCER", "Non Cancer"),

    # ── archive (6) ──
    ("Calculus/Calculus", "Calculus"),
    ("Data caries/Data caries/caries augmented data set/preview", "Caries"),
    ("Gingivitis/Gingivitis", "Gingivitis"),
    ("Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview", "Ulcers"),
    ("Tooth Discoloration/Tooth Discoloration /Tooth_discoloration_augmented_dataser/preview", "Tooth Discoloration"),
    ("hypodontia/hypodontia", "Hypodontia"),
    ("Data caries/Data caries/caries orignal data set/done", "Caries"),
    ("Mouth Ulcer/Mouth Ulcer/ulcer original dataset/ulcer original dataset", "Ulcers"),
    ("Tooth Discoloration/Tooth Discoloration /tooth discoloration original dataset/tooth discoloration original dataset", "Tooth Discoloration"),
]


def build_kaggle_source_map(kaggle_paths):
    """
    Walk each downloaded Kaggle dataset root and auto-map subfolders that look
    like cancer/non-cancer classification directories.
    For multi-cancer datasets, only include oral-related folders.
    """
    CANCER_KEYWORDS  = ["cancer", "malignant", "oscc", "abnormal"]
    HEALTHY_KEYWORDS = ["non cancer", "non_cancer", "noncancer", "normal",
                        "benign", "healthy"]
    # Skip folders clearly belonging to non-oral cancer types
    EXCLUDE_KEYWORDS = ["breast", "cervix", "kidney", "lung", "brain",
                        "colon", "liver", "prostate", "skin", "thyroid",
                        "stomach", "uterus", "ovarian", "bladder",
                        "lymphoma", "leukemia"]

    extra_map = []
    for root_path in kaggle_paths:
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Only consider leaf directories that contain images
            image_files = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not image_files:
                continue
            full_path_lower = dirpath.lower()
            folder_name = os.path.basename(dirpath).lower()

            # Skip non-oral cancer types
            if any(kw in full_path_lower for kw in EXCLUDE_KEYWORDS):
                continue

            if any(kw in folder_name for kw in CANCER_KEYWORDS):
                extra_map.append((dirpath, "Oral Cancer"))
            elif any(kw in folder_name for kw in HEALTHY_KEYWORDS):
                extra_map.append((dirpath, "Non Cancer"))
    return extra_map


def build_hf_source_map(hf_paths):
    """
    Map specific folders from downloaded HF datasets.
    HRruiH/Dataset-of-oral-mucosal-diseases uses '1' for Normal, '5' for Oral Cancer.
    """
    extra_map = []
    for root_path in hf_paths:
        for dirpath, dirnames, filenames in os.walk(root_path):
            image_files = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not image_files:
                continue
            folder_name = os.path.basename(dirpath).lower()
            if folder_name == "1":
                extra_map.append((dirpath, "Non Cancer"))
            elif folder_name == "5":
                extra_map.append((dirpath, "Oral Cancer"))
    return extra_map


def organise_dataset(extra_source_map=None):
    """Collect images from extracted zips + Kaggle downloads, deduplicate, split."""
    print("🔀 Organising dataset …")
    splits = ["train", "val", "test"]
    for split in splits:
        for cls in ALL_CLASSES:
            os.makedirs(os.path.join(WORK_DIR, split, cls), exist_ok=True)

    # Gather images per class — from uploaded zips
    class_images = {cls: [] for cls in ALL_CLASSES}
    for rel_dir, cls_name in SOURCE_MAP:
        src_dir = os.path.join(EXTRACT_DIR, rel_dir)
        if not os.path.isdir(src_dir):
            print(f"  ⚠️  Source dir not found: {rel_dir}")
            continue
        imgs = collect_images_from_dir(src_dir)
        class_images[cls_name].extend(imgs)
        print(f"  {cls_name}: found {len(imgs)} images in {os.path.basename(rel_dir)}")

    # Gather images from Kaggle downloads
    if extra_source_map:
        print("\n  📂 Adding images from Kaggle datasets …")
        for abs_dir, cls_name in extra_source_map:
            imgs = collect_images_from_dir(abs_dir)
            class_images[cls_name].extend(imgs)
            print(f"  {cls_name}: found {len(imgs)} images in {os.path.basename(abs_dir)}")

    # De‑duplicate based on file basename (same image may appear in archives 1 & 6)
    for cls_name in ALL_CLASSES:
        seen_names = set()
        unique_images = []
        for path in class_images[cls_name]:
            basename = os.path.basename(path)
            if basename not in seen_names:
                seen_names.add(basename)
                unique_images.append(path)
        class_images[cls_name] = unique_images
        print(f"  {cls_name}: {len(unique_images)} unique images after dedup")

    # Split 70/20/10 and copy
    print("\n  Splitting into train / val / test …")
    for cls_name in ALL_CLASSES:
        images = class_images[cls_name]
        if not images:
            print(f"  ⚠️  No images for class '{cls_name}'")
            continue
        train_paths, test_paths = train_test_split(images, test_size=0.1, random_state=SEED)
        train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=SEED)

        for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for img_path in paths:
                dst = os.path.join(WORK_DIR, split_name, cls_name, os.path.basename(img_path))
                if not os.path.exists(dst):
                    shutil.copy2(img_path, dst)

    # Print statistics
    print()
    for split in splits:
        for cls in ALL_CLASSES:
            count = len(os.listdir(os.path.join(WORK_DIR, split, cls)))
            print(f"  {split:>5s}/{cls}: {count}")

    # Cleanup extracted files to save disk space
    print("\n  🧹 Cleaning up extracted files …")
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)


# ────────────────────────────────────────────────────────────
# 2. Transforms (enhanced augmentation)
# ────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ────────────────────────────────────────────────────────────
# 3. Model Definition
# ────────────────────────────────────────────────────────────
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = densenet121(weights="IMAGENET1K_V1")

        # Freeze entire backbone initially
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace classifier head
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def unfreeze_last_n_denseblocks(self, n: int = 2) -> None:
        """Unfreeze the last `n` DenseBlocks + transition layers for fine‑tuning."""
        blocks = [name for name, _ in self.model.features.named_children()]
        unfreeze_from = blocks[-n * 2:]
        for name, module in self.model.features.named_children():
            if name in unfreeze_from:
                for param in module.parameters():
                    param.requires_grad = True
        unfrozen = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total    = sum(p.numel() for p in self.model.parameters())
        print(f"  Unfrozen {unfrozen:,} / {total:,} parameters "
              f"({unfrozen / total * 100:.1f}%)")


def load_and_expand_model(pretrained_path: str, old_num: int, new_num: int) -> CustomDenseNet:
    """
    Load the 6‑class pre‑trained model and expand the classifier head to `new_num` classes.
    Weights for the original classes are preserved; new class weights are initialised randomly.
    """
    print(f"\n📂 Loading pre-trained model from {pretrained_path} …")

    # Build a model with the OLD number of classes to load weights
    old_model = CustomDenseNet(num_classes=old_num)
    state = torch.load(pretrained_path, map_location="cpu")
    old_model.load_state_dict(state)
    print(f"  ✅ Loaded {old_num}‑class model weights.")

    if old_num == new_num:
        print("  Class count unchanged; returning model as‑is.")
        return old_model

    # Build the NEW model with expanded classifier
    new_model = CustomDenseNet(num_classes=new_num)

    # Copy backbone weights (features) from old model
    new_model.model.features.load_state_dict(old_model.model.features.state_dict())

    # Copy compatible classifier weights
    old_classifier = old_model.model.classifier
    new_classifier = new_model.model.classifier

    # Layer 1 (Linear: in_features → 512) – same dimensions, copy fully
    new_classifier[1].weight.data.copy_(old_classifier[1].weight.data)
    new_classifier[1].bias.data.copy_(old_classifier[1].bias.data)

    # Layer 4 (Linear: 512 → num_classes) – copy first `old_num` rows
    new_classifier[4].weight.data[:old_num].copy_(old_classifier[4].weight.data)
    new_classifier[4].bias.data[:old_num].copy_(old_classifier[4].bias.data)

    print(f"  ✅ Expanded classifier from {old_num} → {new_num} classes.")
    print(f"     Original class weights preserved; new class weights randomly initialised.")

    return new_model


# ────────────────────────────────────────────────────────────
# 4. Training Helpers
# ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def train_one_epoch(model, loader, optimizer, scheduler, epoch_num):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step(epoch_num)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    # Step 1 — Download extra Kaggle & Hugging Face datasets
    print("\n📥 Step 1a: Download extra Kaggle & Hugging Face datasets\n")
    kaggle_paths = download_kaggle_datasets()
    kaggle_map = build_kaggle_source_map(kaggle_paths)
    
    hf_paths = download_hf_datasets()
    hf_map = build_hf_source_map(hf_paths)
    
    extra_map = kaggle_map + hf_map
    print(f"  Found {len(extra_map)} cancer/non-cancer folders from downloaded datasets.\n")

    # Step 1b — Extract uploaded zips & organise all data
    print("📥 Step 1b: Extract zips & organise data\n")
    extract_zips()
    organise_dataset(extra_source_map=extra_map)

    # Step 2 — Datasets & DataLoaders
    print("\n📦 Step 2: Create DataLoaders\n")
    train_dir = os.path.join(WORK_DIR, "train")
    val_dir   = os.path.join(WORK_DIR, "val")
    test_dir  = os.path.join(WORK_DIR, "test")

    train_dataset = ImageFolder(train_dir, transform=train_transforms, loader=safe_pil_loader)
    val_dataset   = ImageFolder(val_dir,   transform=val_transforms,  loader=safe_pil_loader)
    test_dataset  = ImageFolder(test_dir,  transform=val_transforms,  loader=safe_pil_loader)

    # Verify class‑to‑index mapping matches ALL_CLASSES
    print(f"  ImageFolder classes: {train_dataset.classes}")
    print(f"  Expected classes:    {ALL_CLASSES}")

    # Weighted sampler for class imbalance
    class_counts = Counter(train_dataset.targets)
    total = sum(class_counts.values())
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"\n  Classes ({num_classes}): {train_dataset.classes}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Step 3 — Load & expand model
    model = load_and_expand_model(
        pretrained_path=PRETRAINED_MODEL_PATH,
        old_num=len(OLD_CLASSES),
        new_num=num_classes,
    ).to(DEVICE)

    # ────────────────────────────────────────────────────────────
    # Phase 1 — Frozen Backbone Warm‑Up
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier head only (backbone frozen)")
    print("=" * 60)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CLASSIFIER_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    best_val_loss = float("inf")

    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"  Epoch [{epoch + 1}/{PHASE1_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # ────────────────────────────────────────────────────────────
    # Phase 2 — Fine‑Tune with Unfrozen DenseBlocks
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Fine‑tuning (last 2 DenseBlocks unfrozen)")
    print("=" * 60)

    model.unfreeze_last_n_denseblocks(n=2)

    # Differential learning rates
    param_groups = [
        {"params": [p for n, p in model.model.features.named_parameters() if p.requires_grad],
         "lr": BACKBONE_LR},
        {"params": model.model.classifier.parameters(),
         "lr": CLASSIFIER_LR},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    epochs_no_improve = 0

    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"  Epoch [{epoch + 1}/{PHASE2_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("    ✅ Saved new best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    ⏹ Early stopping after {epoch + 1} epochs "
                      f"(no improvement for {PATIENCE}).")
                break

    # ────────────────────────────────────────────────────────────
    # Test‑Time Evaluation with TTA
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION on Test Set (with TTA)")
    print("=" * 60)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    tta_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    NUM_TTA = 5
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = torch.softmax(model(images), dim=1)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

            for _ in range(NUM_TTA):
                denorm = images * std + mean
                augmented = torch.stack([
                    tta_transforms(transforms.ToPILImage()(img.cpu()))
                    for img in denorm
                ]).to(DEVICE)
                outputs += torch.softmax(model(augmented), dim=1)

            avg_outputs = outputs / (NUM_TTA + 1)
            preds = avg_outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    class_names = train_dataset.classes
    print(f"\n🎯 Test Accuracy (with TTA): {test_accuracy * 100:.2f}%")
    print(f"\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
    print(f"\n📦 Fine-tuned model saved to: {MODEL_SAVE_PATH}")
    print(f"\nDone! Download the .pth file and place it alongside app.py.")


if __name__ == "__main__":
    main()
