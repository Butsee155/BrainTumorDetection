# save as debug_batch.py
from model_engine import predict, generate_gradcam, get_tumor_info
import traceback
import os

# Update this path to one of your actual MRI images
test_img = r"D:\Data Projects\Data Science Projects\BrainTumorDetection\data\Testing\meningioma\Te-me_272.jpg"

print(f"File exists: {os.path.exists(test_img)}")

try:
    print("\n[1] Testing predict...")
    preds = predict(test_img)
    print(f"Predictions: {preds}")

    print("\n[2] Testing gradcam...")
    heatmap = generate_gradcam(test_img)
    print(f"Heatmap saved: {heatmap}")

    print("\n[3] Testing tumor info...")
    info = get_tumor_info(preds[0][0])
    print(f"Info: {info}")

    print("\n[SUCCESS] All steps working!")

except Exception:
    print("\n[ERROR] Full traceback:")
    traceback.print_exc()