import torch
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from utils.ldpolyp_models import (
    UNet,
    MiniUNet,
    ResUNet,
    DeepUNet,
    AttentionUNet
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# LOAD MODELS
# =====================================================

models_dict = {

    "UNet": UNet(),

    "MiniUNet": MiniUNet(),

    "ResUNet": ResUNet(),

    "DeepUNet": DeepUNet(),

    "AttentionUNet": AttentionUNet()
}

loaded_models = {}

for name, model in models_dict.items():

    path = f"models/{name}.pth"

    model.load_state_dict(
        torch.load(
            path,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    loaded_models[name] = model

    print(f"Loaded Model: {name}")

# =====================================================
# ENSEMBLE PREDICTION
# =====================================================

def ensemble_predict(image):

    # ================================================
    # STORE ORIGINAL SIZE
    # ================================================

    original_width, original_height = image.size

    image = image.convert("RGB")

    # ================================================
    # TRANSFORM
    # ================================================

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    # ================================================
    # STORE PREDICTIONS
    # ================================================

    predictions = {}

    ensemble_preds = []

    # ================================================
    # MODEL LOOP
    # ================================================

    for name, model in loaded_models.items():

        with torch.no_grad():

            # ========================================
            # MODEL PREDICTION
            # ========================================

            pred = model(input_tensor)

            pred = torch.sigmoid(pred)

            pred = pred.squeeze().cpu().numpy()

            # ========================================
            # FIX DIFFERENT OUTPUT SHAPES
            # ========================================

            pred = cv2.resize(
                pred,
                (256, 256)
            )

            # ========================================
            # BINARY MASK
            # ========================================

            pred = (
                pred > 0.2
            ).astype(np.uint8)

            predictions[name] = pred

            ensemble_preds.append(pred)

    # ================================================
    # ENSEMBLE AVERAGING
    # ================================================

    ensemble_mask = np.mean(
        np.stack(ensemble_preds),
        axis=0
    )

    ensemble_mask = (
        ensemble_mask > 0.5
    ).astype(np.uint8)

    # ================================================
    # RESIZE BACK TO ORIGINAL IMAGE SIZE
    # ================================================

    ensemble_mask = cv2.resize(
        ensemble_mask,
        (original_width, original_height)
    )

    # ================================================
    # ALSO RESIZE INDIVIDUAL PREDICTIONS
    # ================================================

    resized_predictions = {}

    for name, pred in predictions.items():

        pred = cv2.resize(
            pred,
            (original_width, original_height)
        )

        resized_predictions[name] = pred

    return resized_predictions, ensemble_mask