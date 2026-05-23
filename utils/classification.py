import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL
# -----------------------------
def initialize_classifier():

    model = models.resnet18(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(
        "models/resnet18_phase1_checkpoint.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    return model


classifier_model = initialize_classifier()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def classify_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = classifier_model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    label = "Polyp" if predicted.item() == 1 else "Non-Polyp"

    return label, confidence.item()