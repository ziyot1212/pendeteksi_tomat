import os
import shutil
from roboflow import Roboflow

# Masukkan API Key Roboflow milikmu
API_KEY = "p9S0301QatFSEAiQiqGy"

print("[PROSES] Menghubungkan ke Roboflow Universe...")
rf = Roboflow(api_key=API_KEY)

# Menggunakan dataset publik aktif 'laboro-tomato' (khusus tomat di tanaman/kebun)
print("[PROSES] Mengunduh dataset tomat kebun berformat YOLOv8...")
project = rf.workspace("laboro-ai").project("laboro-tomato")
version = project.version(1)
dataset = version.download("yolov8")

print(f"\n[SUKSES] Dataset berhasil diunduh ke: {dataset.location}")

# Pindahkan file hasil unduhan ke folder 'yolo_dataset' proyek
target_dir = os.path.join(os.getcwd(), 'yolo_dataset')
os.makedirs(target_dir, exist_ok=True)

for item in os.listdir(dataset.location):
    src = os.path.join(dataset.location, item)
    dst = os.path.join(target_dir, item)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print("[SELESAI] Semua folder images, labels, dan data.yaml sudah siap di 'yolo_dataset'!")