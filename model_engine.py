import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime
from db_config import get_connection

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASSES     = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MODEL_PATH  = "models/brain_tumor_model.h5"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("[INFO] Loading CNN model...")
model = load_model(MODEL_PATH)

# ── Warm up model with a dummy input so all layers have defined outputs ───────
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("[INFO] Model loaded.")


def preprocess_image(image_path):
    """Load and preprocess MRI image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict(image_path):
    """
    Run prediction on MRI image.
    Returns top 3 predictions with confidence scores.
    """
    input_arr, _ = preprocess_image(image_path)
    preds        = model.predict(input_arr, verbose=0)[0]
    top_idx      = np.argsort(preds)[::-1][:3]
    results      = [(CLASSES[i], round(float(preds[i]) * 100, 2))
                    for i in top_idx]
    return results


def generate_gradcam(image_path, class_idx=None):
    """
    Generate Grad-CAM heatmap overlay on MRI image.
    Uses a reliable approach that works with EfficientNetB0 + Sequential wrapper.
    Returns path to saved heatmap image.
    """
    input_arr, orig_img = preprocess_image(image_path)

    # Get predicted class if not provided
    preds = model.predict(input_arr, verbose=0)
    if class_idx is None:
        class_idx = int(np.argmax(preds[0]))

    try:
        # ── Method 1: Find last conv layer inside EfficientNetB0 base ────────
        base_model     = model.layers[0]  # EfficientNetB0
        last_conv_name = None

        # Find last conv-like layer with 4D output
        for layer in reversed(base_model.layers):
            try:
                if hasattr(layer, "output") and len(layer.output.shape) == 4:
                    last_conv_name = layer.name
                    break
            except Exception:
                continue

        if last_conv_name is None:
            raise ValueError("No conv layer found")

        # Build grad model
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv_name).output,
                     base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, base_preds = grad_model(input_arr)
            # Use full model predictions for correct class
            full_preds = model(input_arr)
            loss       = full_preds[:, class_idx]

        grads        = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_out_np  = conv_out[0].numpy()

        # Weight feature maps by gradients
        for i in range(pooled_grads.shape[-1]):
            conv_out_np[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_out_np, axis=-1)

    except Exception as method1_err:
        print(f"[INFO] Grad-CAM Method 1 failed ({method1_err}), trying Method 2...")

        try:
            # ── Method 2: Use GlobalAveragePooling input ──────────────────────
            gap_layer  = None
            for layer in model.layers:
                if "global_average" in layer.name.lower():
                    gap_layer = layer
                    break

            if gap_layer is None:
                raise ValueError("No GAP layer found")

            grad_model2 = tf.keras.models.Model(
                inputs=model.input,
                outputs=[gap_layer.input, model.output]
            )

            with tf.GradientTape() as tape:
                conv_out2, preds2 = grad_model2(input_arr)
                loss2 = preds2[:, class_idx]

            grads2        = tape.gradient(loss2, conv_out2)
            pooled_grads2 = tf.reduce_mean(grads2, axis=(0, 1, 2)).numpy()
            conv_out2_np  = conv_out2[0].numpy()

            for i in range(pooled_grads2.shape[-1]):
                conv_out2_np[:, :, i] *= pooled_grads2[i]

            heatmap = np.mean(conv_out2_np, axis=-1)

        except Exception as method2_err:
            print(f"[INFO] Grad-CAM Method 2 failed ({method2_err}), using fallback...")

            # ── Method 3: Colour-based fallback heatmap ───────────────────────
            orig_bgr  = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            gray      = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
            blurred   = cv2.GaussianBlur(gray, (15, 15), 0)
            _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
            heatmap_c = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
            overlay   = cv2.addWeighted(orig_bgr, 0.6, heatmap_c, 0.4, 0)

            ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(RESULTS_DIR, f"gradcam_{ts}.jpg")
            cv2.imwrite(save_path, overlay)
            return save_path

    # ── Process and overlay heatmap ───────────────────────────────────────────
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap = heatmap / heatmap.max()

    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.resize(orig_bgr, (IMG_SIZE, IMG_SIZE))
    overlay  = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(RESULTS_DIR, f"gradcam_{ts}.jpg")
    cv2.imwrite(save_path, overlay)
    return save_path


def highlight_tumor_region(image_path, heatmap_path):
    """
    Draw bounding box around the highest activation region.
    Returns path to annotated image.
    """
    try:
        img     = cv2.imread(image_path)
        img     = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        heatmap = cv2.imread(heatmap_path)
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

        # Convert heatmap to grayscale for contour detection
        gray      = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = img.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw label background
            cv2.rectangle(annotated, (x, y-22), (x+w, y), (0, 255, 0), -1)
            cv2.putText(annotated, "Detected Region",
                        (x+4, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1)

        ts        = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(RESULTS_DIR, f"annotated_{ts}.jpg")
        cv2.imwrite(save_path, annotated)
        return save_path

    except Exception as ann_err:
        print(f"[ERROR] Annotation failed: {ann_err}")
        # Return original image path as fallback
        return image_path


def get_tumor_info(tumor_type):
    """Fetch tumor details from SQL Server database"""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TumorType, Description, SeverityLevel,
                   Recommendation, Specialist, Urgency
            FROM TumorInfo
            WHERE TumorType = ?
        """, tumor_type)
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "type":           row[0],
                "description":    row[1],
                "severity":       row[2],
                "recommendation": row[3],
                "specialist":     row[4],
                "urgency":        row[5],
            }
    except Exception as db_err:
        print(f"[ERROR] Tumor info DB: {db_err}")
    return None


def save_scan(patient_name, age, gender, image_path,
               heatmap_path, predictions, diagnosis,
               severity, recommendation):
    """Save scan result to SQL Server database"""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        p1 = predictions[0] if len(predictions) > 0 else ("", 0)
        p2 = predictions[1] if len(predictions) > 1 else ("", 0)
        p3 = predictions[2] if len(predictions) > 2 else ("", 0)
        cursor.execute("""
            INSERT INTO ScanHistory
            (PatientName, PatientAge, PatientGender,
             ImagePath, HeatmapPath,
             Prediction1, Confidence1,
             Prediction2, Confidence2,
             Prediction3, Confidence3,
             FinalDiagnosis, SeverityLevel, Recommendation)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_name, age, gender,
            image_path, heatmap_path,
            p1[0], p1[1],
            p2[0], p2[1],
            p3[0], p3[1],
            diagnosis, severity, recommendation
        ))
        conn.commit()
        conn.close()
        print(f"[INFO] Scan saved: {diagnosis} — {p1[1]:.1f}%")
    except Exception as db_err:
        print(f"[ERROR] Save scan: {db_err}")