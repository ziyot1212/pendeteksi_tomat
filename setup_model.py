import os
import urllib.request

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
TARGET_PT = os.path.join(MODEL_DIR, "best.pt")

# Mengunduh model YOLO tomat kebun siap pakai
URL = "https://huggingface.co/arnabdhar/YOLOv8-Tomato-Leaf-Disease-and-Fruit-Detection/resolve/main/best.pt"

print("[PROSES] Mengunduh model YOLOv8 khusus tomat kebun...")
try:
    urllib.request.urlretrieve(URL, TARGET_PT)
    print(f"[BERHASIL] Model disimpan di: {TARGET_PT}")
except Exception as e:
    # Fallback ke sumber cadangan jika link pertama gagal
    ALT_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
    print(f"Mencoba sumber alternatif... ({e})")
    urllib.request.urlretrieve(ALT_URL, TARGET_PT)
    print(f"[BERHASIL] Model alternatif disimpan di: {TARGET_PT}")