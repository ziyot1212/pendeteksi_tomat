import os
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Gunakan path absolut agar tidak pernah salah lokasi
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(base_dir, 'yolo_dataset', 'data.yaml')

    # Deteksi apakah CUDA GPU aktif
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Training berjalan pada device: {device}")

    model = YOLO('yolov8s.pt')
    model.train(
        data=data_yaml_path,
        epochs=40,
        imgsz=640,
        batch=8,
        device=device,
        name='tomato_garden_detector'
    )