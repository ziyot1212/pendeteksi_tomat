import os
import shutil
from bing_image_downloader import downloader

TARGET_DIR = os.path.join('dataset', 'bukan_tomat')
TEMP_DIR = 'temp_negatives'
os.makedirs(TARGET_DIR, exist_ok=True)

# Daftar objek pengecoh yang mirip tomat (Hard Negatives)
SEARCH_QUERIES = [
    "red cherry fruit isolated",
    "fresh cherries with stem",
    "red apple fruit",
    "green granny smith apple",
    "fresh orange fruit",
    "strawberry fruit",
    "red plum fruit",
    "red bell pepper",
    "red ball toy"
]

def build_negative_dataset():
    print("=== Mengunduh Gambar Benda Asing & Buah Mirip Tomat (Ceri, Apel, dll) ===")
    count = 0

    for query in SEARCH_QUERIES:
        print(f"\n[PROSES] Mengunduh kategori: {query}...")
        try:
            downloader.download(
                query,
                limit=25,
                output_dir=TEMP_DIR,
                adult_filter_off=True,
                force_replace=False,
                timeout=15,
                verbose=False
            )
            query_folder = os.path.join(TEMP_DIR, query)
            if os.path.exists(query_folder):
                for fname in os.listdir(query_folder):
                    fpath = os.path.join(query_folder, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        count += 1
                        new_name = f"non_tomato_{count:04d}{os.path.splitext(fname)[1]}"
                        shutil.move(fpath, os.path.join(TARGET_DIR, new_name))
        except Exception as e:
            print(f"Lewati sebagian ({e})")

    # Bersihkan folder sementara
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    total_files = len(os.listdir(TARGET_DIR))
    print(f"\n[SELESAI] Total {total_files} gambar berhasil disiapkan di '{TARGET_DIR}'!")

if __name__ == '__main__':
    build_negative_dataset()