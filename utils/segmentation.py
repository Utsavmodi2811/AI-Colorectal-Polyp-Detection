import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import models, transforms
from PIL import Image

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# ATTENTION BLOCK
# =====================================================

class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):

        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)

        psi = self.psi(psi)

        return x * psi

# =====================================================
# CONV BLOCK
# =====================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)

# =====================================================
# RESNET50 ATTENTION U-NET
# =====================================================

class ResNet50AttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        resnet = models.resnet50(weights=None)

        # -----------------------------
        # ENCODER
        # -----------------------------

        self.enc0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.pool = resnet.maxpool

        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # -----------------------------
        # DECODER
        # -----------------------------

        self.up4 = nn.ConvTranspose2d(
            2048,
            1024,
            2,
            stride=2
        )

        self.att4 = AttentionBlock(
            1024,
            1024,
            512
        )

        self.dec4 = ConvBlock(
            2048,
            1024
        )

        self.up3 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.att3 = AttentionBlock(
            512,
            512,
            256
        )

        self.dec3 = ConvBlock(
            1024,
            512
        )

        self.up2 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.att2 = AttentionBlock(
            256,
            256,
            128
        )

        self.dec2 = ConvBlock(
            512,
            256
        )

        self.up1 = nn.ConvTranspose2d(
            256,
            64,
            2,
            stride=2
        )

        self.att1 = AttentionBlock(
            64,
            64,
            32
        )

        self.dec1 = ConvBlock(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # -----------------------------
        # ENCODER
        # -----------------------------

        e0 = self.enc0(x)

        p0 = self.pool(e0)

        e1 = self.enc1(p0)

        e2 = self.enc2(e1)

        e3 = self.enc3(e2)

        e4 = self.enc4(e3)

        # -----------------------------
        # DECODER
        # -----------------------------

        d4 = self.up4(e4)

        a4 = self.att4(d4, e3)

        d4 = torch.cat([d4, a4], dim=1)

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        a3 = self.att3(d3, e2)

        d3 = torch.cat([d3, a3], dim=1)

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        a2 = self.att2(d2, e1)

        d2 = torch.cat([d2, a2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        a1 = self.att1(d1, e0)

        d1 = torch.cat([d1, a1], dim=1)

        d1 = self.dec1(d1)

        out = self.final(d1)

        out = F.interpolate(
            out,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        return torch.sigmoid(out)

# =====================================================
# LOAD MODEL
# =====================================================

model_seg = ResNet50AttentionUNet().to(device)

model_seg.load_state_dict(
    torch.load(
        "models/resnet50_attention_fixed.pth",
        map_location=device
    )
)

model_seg.eval()

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# SEGMENTATION FUNCTION
# =====================================================

def segment_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model_seg(input_tensor)

    mask = output.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8)

    return mask