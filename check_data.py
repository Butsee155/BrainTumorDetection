# save as check_data.py
import os

for split in ["Training", "Testing"]:
    path = f"data/{split}"
    if not os.path.exists(path):
        print(f"[ERROR] NOT FOUND: {path}")
        continue
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        cls_path = os.path.join(path, cls)
        if os.path.exists(cls_path):
            count = len(os.listdir(cls_path))
            print(f"[OK] {split}/{cls}: {count} images")
        else:
            print(f"[MISSING] {split}/{cls} — folder not found!")