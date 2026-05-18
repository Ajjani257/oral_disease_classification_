import os

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import densenet121

# Class order must match the ImageFolder alphabetical order used during training.
CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Tooth Discoloration",
    "Ulcers",
]

MODEL_PATH = "densenet_oral_model_improved.pth"  # 6-class improved model
REPORTED_TEST_ACCURACY = 0.95  # Improved 6-class model

# Risk level thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50


class CustomDenseNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = densenet121(weights=None)
        in_features = self.model.classifier.in_features
        # Improved classifier head (matches train_improved.py architecture)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@st.cache_resource
def load_model() -> CustomDenseNet:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = CustomDenseNet(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # Handles checkpoints saved from DataParallel.
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return val_transforms(image).unsqueeze(0)


def get_display_image(image: Image.Image) -> np.ndarray:
    """Resize and crop the image to match model input, return as normalized float32 array."""
    display_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    img = display_transforms(image)
    return np.array(img, dtype=np.float32) / 255.0


def predict_image(model: nn.Module, image: Image.Image) -> tuple[str, float, torch.Tensor]:
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return CLASS_NAMES[predicted_idx.item()], confidence.item(), probabilities


def generate_gradcam(model: CustomDenseNet, image: Image.Image) -> np.ndarray:
    """Generate Grad-CAM heatmap for the given image."""
    input_tensor = preprocess_image(image)
    rgb_img = get_display_image(image)

    # Target the last DenseBlock's final norm layer for Grad-CAM
    target_layer = model.model.features.norm5

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # None = use top predicted class
    grayscale_cam = grayscale_cam[0, :]

    # Overlay heatmap on original image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image


def get_risk_level(confidence: float, predicted_class: str) -> tuple[str, str, str]:
    """
    Determine risk level based on confidence and predicted class.
    Returns: (risk_level, color, description)
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        risk_level = "🔴 HIGH"
        color = "red"
        description = (
            f"The model is **highly confident ({confidence * 100:.1f}%)** that this image shows "
            f"**{predicted_class}**. We recommend consulting a dental professional promptly."
        )
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        risk_level = "🟡 MEDIUM"
        color = "orange"
        description = (
            f"The model has **moderate confidence ({confidence * 100:.1f}%)** in detecting "
            f"**{predicted_class}**. The result is suggestive but not conclusive. "
            f"Consider a professional evaluation for confirmation."
        )
    else:
        risk_level = "🟢 LOW"
        color = "green"
        description = (
            f"The model has **low confidence ({confidence * 100:.1f}%)** in its prediction of "
            f"**{predicted_class}**. The image may not clearly indicate any condition, "
            f"or the model is uncertain. A professional assessment is still advisable."
        )

    return risk_level, color, description


def main() -> None:
    st.set_page_config(page_title="Oral Disease Predictor", page_icon="🦷", layout="centered")
    st.title("🦷 Oral Disease Predictor (DenseNet121)")
    st.write("Upload an oral image to get a prediction from your trained model.")
    st.info(
        f"Model test accuracy from notebook: {REPORTED_TEST_ACCURACY * 100:.2f}% "
        "(dataset-level metric)."
    )

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probabilities = predict_image(model, image)
                cam_image = generate_gradcam(model, image)

            # --- Prediction Result ---
            st.success(f"**Prediction: {predicted_class}**")
            st.metric("Prediction confidence", f"{confidence * 100:.2f}%")

            # --- Risk Level Indicator ---
            st.subheader("⚠️ Risk Assessment")
            risk_level, color, description = get_risk_level(confidence, predicted_class)

            st.markdown(
                f"""
                <div style="
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border-left: 5px solid {color};
                    background-color: rgba(0,0,0,0.05);
                    margin-bottom: 1rem;
                ">
                    <h3 style="margin-top: 0;">Risk Level: {risk_level}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "⚕️ **Disclaimer:** This tool is for educational purposes only. "
                "It is NOT a substitute for professional dental diagnosis. "
                "Always consult a qualified dental professional."
            )

            # --- Grad-CAM Heatmap ---
            st.subheader("🔥 Grad-CAM Heatmap")
            st.caption(
                "The heatmap highlights the regions of the image the model focused on "
                "to make its prediction. Warmer colors (red/yellow) indicate higher importance."
            )
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    get_display_image(image),
                    caption="Original (cropped)",
                    use_container_width=True,
                )
            with col2:
                st.image(
                    cam_image,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True,
                )

            # --- Class Probabilities ---
            st.subheader("📊 Class Probabilities")
            prob_data = {CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(len(CLASS_NAMES))}
            st.bar_chart(prob_data)

            st.caption("Confidence is the model's probability for this image, not true accuracy.")


if __name__ == "__main__":
    main()
