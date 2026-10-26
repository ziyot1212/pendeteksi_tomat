import os
import shutil
from roboflow import Roboflow

# Mengambil dataset publik 'tomato-detection' yang sudah beranotasi lengkap
rf = Roboflow(api_key="")  # dataset publik bisa diakses langsung via link/download publik
project_name = "tomato-detection-kebun"

print("[PROSES] Mengunduh dataset tomat kebun yang sudah beranotasi...")
# Alternatif langsung menggunakan dataset terbuka Ultralytics / HuggingFace jika tanpa API key
os.system("curl -L -o dataset_yolo.zip https://universe.roboflow.com/ds/7tXm54WfS7?key=sample") 