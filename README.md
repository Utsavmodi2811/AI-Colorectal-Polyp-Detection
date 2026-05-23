# 🩺 ColonoVision AI

AI-Based Colorectal Polyp Detection & Segmentation System

---

# 📌 Overview

ColonoVision AI is a deep learning-based medical imaging system developed for:

- Colorectal polyp detection
- Polyp segmentation
- Batch frame analysis
- Colonoscopy video analysis
- Ensemble segmentation research

The platform combines:
- CNN-based classification
- Attention U-Net segmentation
- Multi-model ensemble learning
- Medical visualization techniques

---

# 🚀 Features

## 🔍 Image Classification
- ResNet18
- Polyp vs Non-Polyp detection
- Validation Accuracy: **99.91%**

## 🧠 Segmentation
- Attention U-Net
- Dice Score: **0.834**

## 🧬 LDPolyp Ensemble Models
Models Used:
- UNet
- MiniUNet
- ResUNet
- DeepUNet
- AttentionUNet

Ensemble Dice Score:
- **0.745**

## 🎥 Video Analysis
- Frame-by-frame processing
- AI overlay visualization
- Automated polyp detection

## 🧩 Batch Analysis
- Multi-frame processing
- Clinical metrics
- CSV report export

---

# 🖥️ Technologies Used

- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Pandas
- PIL

---

# 📂 Project Structure

```bash
AI-Colorectal-Polyp-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── classification.py
│   ├── segmentation.py
│   ├── visualization.py
│   ├── video_utils.py
│   ├── ldpolyp_models.py
│   └── ldpolyp_inference.py
│
└── models/
```

---

# 📥 Model Weights

Download all model weights from Google Drive:

👉 [Download Model Weights](https://drive.google.com/drive/folders/1YFYGuNMWH0mYryXE8PEhAFJgL14xlyPD?usp=sharing)

After downloading:

Place all `.pth` files inside:

```bash
models/
```

---

# ▶️ Run Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Utsavmodi2811/AI-Colorectal-Polyp-Detection.git
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Download Model Weights

Place all `.pth` files inside:

```bash
models/
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📊 Results

| Model | Metric |
|---|---|
| ResNet18 | 99.91% Accuracy |
| Attention U-Net | Dice: 0.834 |
| LDPolyp Ensemble | Dice: 0.745 |

---

# 👨‍💻 Author

## Utsav Modi

- GitHub: https://github.com/Utsavmodi2811
- LinkedIn: https://www.linkedin.com/in/utsav-modi-223064253/

---
