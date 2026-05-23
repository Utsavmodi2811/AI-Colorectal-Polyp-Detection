import cv2
import os
import tempfile
import numpy as np
from PIL import Image

from utils.classification import classify_image
from utils.segmentation import segment_image
from utils.visualization import create_overlay

# =========================================================
# VIDEO PROCESSING FUNCTION
# =========================================================

def process_video(
    uploaded_video,
    confidence_threshold=0.85
):

    # =====================================================
    # SAVE TEMP VIDEO
    # =====================================================

    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(uploaded_video.read())

    input_path = temp_input.name

    # =====================================================
    # OUTPUT VIDEO
    # =====================================================

    output_path = input_path.replace(
        ".mp4",
        "_output.mp4"
    )

    # =====================================================
    # VIDEO CAPTURE
    # =====================================================

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # =====================================================
    # VIDEO WRITER
    # =====================================================

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # =====================================================
    # STATISTICS
    # =====================================================

    polyp_frames = 0

    normal_frames = 0

    processed_frames = 0

    # =====================================================
    # PROCESS LOOP
    # =====================================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # =============================================
        # CONVERT FRAME
        # =============================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb_frame)

        # =============================================
        # CLASSIFICATION
        # =============================================

        label, confidence = classify_image(
            pil_image
        )

        # =============================================
        # POSITIVE FRAME
        # =============================================

        if (
            label == "Polyp"
            and confidence >= confidence_threshold
        ):

            polyp_frames += 1

            # =========================================
            # SEGMENTATION
            # =========================================

            mask = segment_image(pil_image)

            overlay = create_overlay(
                pil_image,
                mask
            )

            overlay = cv2.resize(
                overlay,
                (width, height)
            )

            overlay_bgr = cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )

            # =========================================
            # TEXT
            # =========================================

            cv2.putText(
                overlay_bgr,
                f"POLYP DETECTED ({confidence*100:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            out.write(overlay_bgr)

        # =============================================
        # NEGATIVE FRAME
        # =============================================

        else:

            normal_frames += 1

            cv2.putText(
                frame,
                "NO POLYP",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            out.write(frame)

        processed_frames += 1

    # =====================================================
    # RELEASE
    # =====================================================

    cap.release()

    out.release()

    return {
        "output_video": output_path,
        "total_frames": processed_frames,
        "polyp_frames": polyp_frames,
        "normal_frames": normal_frames
    }