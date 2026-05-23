import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time
import cv2

from utils.classification import classify_image
from utils.segmentation import segment_image
from utils.visualization import create_overlay
from utils.video_utils import process_video
from utils.ldpolyp_inference import ensemble_predict

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ColonoVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #333;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}

.polyp {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
}

.non-polyp {
    background: linear-gradient(90deg, #00c853, #64dd17);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ System Controls")

confidence_threshold = st.sidebar.slider(
    "Detection Threshold",
    0.50,
    1.00,
    0.85
)

st.sidebar.info("""
### Threshold Explanation

Threshold defines the minimum AI confidence
required before segmentation is performed.

Higher Threshold:
- More strict
- Fewer false positives

Lower Threshold:
- More sensitive
- Detects more possible polyps
""")

st.sidebar.markdown("---")

st.sidebar.info("""
## About Models

### 🔍 Classification Model
- ResNet18
- Accuracy: 99.91%

---

### 🧠 Segmentation Model
- ResNet50 Attention U-Net
- Dice Score: 0.834

---

### 🧬 LDPolyp Ensemble Model
Models Used:
- UNet
- MiniUNet
- ResUNet
- DeepUNet
- AttentionUNet

Ensemble Dice Score:
- 0.745
""")

# =====================================================
# HEADER
# =====================================================

st.title("🩺 ColonoVision AI")

st.markdown("""
### AI-Based Colorectal Polyp Detection & Segmentation System

This intelligent medical imaging platform uses:
- Deep Learning Classification
- Attention U-Net Segmentation
- Ensemble Segmentation Models
- Automated Polyp Localization
- Clinical Visualization
""")

st.markdown("---")

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Image Analysis",
    "🧩 Batch Frame Analysis",
    "🎥 Video Analysis",
    "🧬 LDPolyp Ensemble AI"
])

# =====================================================
# TAB 1 — IMAGE ANALYSIS
# =====================================================

with tab1:

    st.subheader("🖼️ Single Image AI Analysis")

    uploaded_file = st.file_uploader(
        "Upload Colonoscopy Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with st.spinner(
            "Analyzing Image using AI Models..."
        ):

            time.sleep(1)

            label, confidence = classify_image(image)

        # ==========================================
        # RESULT BOX
        # ==========================================

        if label == "Polyp":

            st.markdown(
                '<div class="result-box polyp">⚠️ POLYP DETECTED</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-box non-polyp">✅ NO POLYP DETECTED</div>',
                unsafe_allow_html=True
            )

        # ==========================================
        # METRICS
        # ==========================================

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Prediction",
                label
            )

        with m2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        with m3:

            status = (
                "High Risk"
                if label == "Polyp"
                else "Normal"
            )

            st.metric(
                "Status",
                status
            )

        st.subheader("Confidence Score")

        st.progress(float(confidence))

        # ==========================================
        # MAIN SEGMENTATION
        # ==========================================

        if (
            label == "Polyp"
            and confidence >= confidence_threshold
        ):

            st.markdown("---")

            st.subheader(
                "🧠 Attention U-Net Segmentation"
            )

            mask = segment_image(image)

            overlay = create_overlay(
                image,
                mask
            )

            c1, c2 = st.columns(2)

            with c1:

                st.image(
                    mask * 255,
                    caption="Predicted Mask",
                    use_container_width=True
                )

            with c2:

                st.image(
                    overlay,
                    caption="Overlay Visualization",
                    use_container_width=True
                )

        elif label == "Polyp":

            st.warning(
                "Confidence below threshold. Segmentation skipped."
            )

        else:

            st.success(
                "Healthy Scan — No Segmentation Required"
            )

        # =====================================================
        # LDPOLYP ENSEMBLE RESULT
        # =====================================================

        st.markdown("---")

        st.subheader(
            "🧬 LDPolyp Ensemble Analysis"
        )

        with st.spinner(
            "Running LDPolyp Ensemble Models..."
        ):

            ld_predictions, ld_ensemble_mask = (
                ensemble_predict(image)
            )

        st.success(
            "✅ LDPolyp Ensemble Completed"
        )

        st.metric(
            "Ensemble Dice Score",
            "0.745"
        )

        # ==========================================
        # INDIVIDUAL MODELS
        # ==========================================

        st.subheader(
            "Individual Ensemble Models"
        )

        cols = st.columns(5)

        model_names = list(
            ld_predictions.keys()
        )

        for idx, name in enumerate(model_names):

            with cols[idx]:

                st.image(
                    ld_predictions[name] * 255,
                    caption=name,
                    use_container_width=True
                )

        # ==========================================
        # FINAL ENSEMBLE RESULT
        # ==========================================

        st.markdown("---")

        st.subheader(
            "Final Ensemble Prediction"
        )

        ld_overlay = create_overlay(
            image,
            ld_ensemble_mask
        )

        c5, c6 = st.columns(2)

        with c5:

            st.image(
                ld_ensemble_mask * 255,
                caption="LDPolyp Ensemble Mask",
                use_container_width=True
            )

        with c6:

            st.image(
                ld_overlay,
                caption="LDPolyp Overlay Result",
                use_container_width=True
            )

        # ==========================================
        # METRICS
        # ==========================================

        ld_area = np.sum(ld_ensemble_mask)

        ld_total = (
            ld_ensemble_mask.shape[0]
            * ld_ensemble_mask.shape[1]
        )

        ld_percentage = (
            ld_area / ld_total
        ) * 100

        st.markdown("---")

        m5, m6 = st.columns(2)

        with m5:

            st.metric(
                "LDPolyp Area",
                f"{ld_percentage:.2f}%"
            )

        with m6:

            st.metric(
                "LDPolyp Pixels",
                int(ld_area)
            )

# =====================================================
# TAB 2 — BATCH FRAME ANALYSIS
# =====================================================

with tab2:

    st.subheader(
        "🧩 Advanced Batch Frame Analysis"
    )

    uploaded_frames = st.file_uploader(
        "Upload Multiple Frames",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if uploaded_frames:

        results = []

        progress_bar = st.progress(0)

        for idx, file in enumerate(uploaded_frames):

            image = Image.open(file)

            label, confidence = classify_image(image)

            confidence_percent = round(
                confidence * 100,
                2
            )

            results.append({
                "Frame": file.name,
                "Prediction": label,
                "Confidence (%)": confidence_percent
            })

            with st.expander(f"📷 {file.name}"):

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.image(
                        image,
                        caption="Original Frame",
                        use_container_width=True
                    )

                if (
                    label == "Polyp"
                    and confidence >= confidence_threshold
                ):

                    mask = segment_image(image)

                    overlay = create_overlay(
                        image,
                        mask
                    )

                    with c2:

                        st.image(
                            mask * 255,
                            caption="Segmentation Mask",
                            use_container_width=True
                        )

                    with c3:

                        st.image(
                            overlay,
                            caption="Overlay Result",
                            use_container_width=True
                        )

                    st.error(
                        f"⚠️ POLYP DETECTED | Confidence: {confidence_percent}%"
                    )

                else:

                    with c2:

                        st.success(
                            "✅ No Polyp Detected"
                        )

                    with c3:

                        st.info(
                            "Segmentation Not Required"
                        )

            progress_bar.progress(
                (idx + 1) / len(uploaded_frames)
            )

        st.success(
            "✅ Batch Analysis Completed"
        )

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True
        )

# =====================================================
# TAB 3 — VIDEO ANALYSIS
# =====================================================

with tab3:

    st.subheader(
        "🎥 Real-Time Colonoscopy Video Analysis"
    )

    uploaded_video = st.file_uploader(
        "Upload Colonoscopy Video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button(
            "🚀 Start AI Video Analysis"
        ):

            with st.spinner(
                "Processing Video..."
            ):

                results = process_video(
                    uploaded_video,
                    confidence_threshold
                )

            st.success(
                "✅ Video Processing Completed"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Total Frames",
                    results["total_frames"]
                )

            with c2:

                st.metric(
                    "Polyp Frames",
                    results["polyp_frames"]
                )

            with c3:

                st.metric(
                    "Normal Frames",
                    results["normal_frames"]
                )

            st.markdown("---")

            video_file = open(
                results["output_video"],
                "rb"
            )

            video_bytes = video_file.read()

            st.video(video_bytes)

            st.download_button(
                label="📥 Download Processed Video",
                data=video_bytes,
                file_name="processed_polyp_video.mp4",
                mime="video/mp4"
            )

# =====================================================
# TAB 4 — LDPOLYP ENSEMBLE ONLY
# =====================================================

with tab4:

    st.subheader(
        "🧬 LDPolyp Ensemble Segmentation"
    )

    uploaded_ld_image = st.file_uploader(
        "Upload Polyp Image",
        type=["jpg", "jpeg", "png"],
        key="ldpolyp_upload"
    )

    if uploaded_ld_image:

        image = Image.open(uploaded_ld_image)

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        with st.spinner(
            "Running Ensemble Segmentation..."
        ):

            predictions, ensemble_mask = (
                ensemble_predict(image)
            )

        st.success(
            "✅ Ensemble Prediction Completed"
        )

        st.metric(
            "Ensemble Dice Score",
            "0.745"
        )

        st.markdown("---")

        cols = st.columns(5)

        model_names = list(predictions.keys())

        for idx, name in enumerate(model_names):

            with cols[idx]:

                st.image(
                    predictions[name] * 255,
                    caption=name,
                    use_container_width=True
                )

        st.markdown("---")

        overlay = create_overlay(
            image,
            ensemble_mask
        )

        c1, c2 = st.columns(2)

        with c1:

            st.image(
                ensemble_mask * 255,
                caption="Ensemble Mask",
                use_container_width=True
            )

        with c2:

            st.image(
                overlay,
                caption="Overlay Result",
                use_container_width=True
            )