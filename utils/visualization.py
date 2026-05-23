import cv2
import numpy as np

# =====================================================
# CREATE OVERLAY
# =====================================================

def create_overlay(original_image, mask):

    # ================================================
    # CONVERT PIL IMAGE TO NUMPY
    # ================================================

    image = np.array(original_image)

    # ================================================
    # GET IMAGE SIZE
    # ================================================

    height, width = image.shape[:2]

    # ================================================
    # RESIZE MASK TO MATCH IMAGE
    # ================================================

    mask = cv2.resize(
        mask.astype(np.uint8),
        (width, height)
    )

    # ================================================
    # CREATE OVERLAY
    # ================================================

    overlay = image.copy()

    overlay[mask == 1] = [255, 0, 0]

    # ================================================
    # BLEND IMAGES
    # ================================================

    blended = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    return blended