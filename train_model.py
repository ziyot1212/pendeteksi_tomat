import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = os.path.join('model', 'tomato_model.h5')
CLASS_MAP_PATH = os.path.join('model', 'class_indices.json')
os.makedirs('model', exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 25

def train_smart_tomato_ai():
    # Validasi ketersediaan folder 'bukan_tomat'
    neg_dir = os.path.join(DATASET_DIR, 'bukan_tomat')
    if not os.path.exists(neg_dir) or len(os.listdir(neg_dir)) < 10:
        print("\n[PERINGATAN] Folder 'dataset/bukan_tomat' masih kosong atau kurang dari 10 gambar!")
        print("Jalankan 'python download_negatives.py' terlebih dahulu agar AI belajar mengenali ceri & apel.\n")
        return

    # Rescale 1./255 wajib agar selaras dengan app.py
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.80, 1.20],
        validation_split=0.20
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("\n=== PEMETAAN KELAS AI ===")
    for k, v in train_gen.class_indices.items():
        print(f"Indeks {v} : {k}")

    # Simpan urutan kelas ke file JSON agar app.py otomatis membacanya tanpa salah urutan
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=4)

    num_classes = len(train_gen.class_indices)

    # Gunakan MobileNetV2 Pretrained ImageNet (Sangat pintar membedakan tekstur Ceri, Apel, dan Tomat)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=(128, 128, 3))
    )

    # Buka 30 layer terakhir untuk Fine-Tuning spesifik tomat vs ceri/apel
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.45)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0002),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]

    print("\n=== MEMULAI TRAINING AI 4 KELAS (DENGAN ANTI-CERI & ANTI-APEL) ===")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"\n[BERHASIL] Model baru telah tersimpan di: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_smart_tomato_ai()