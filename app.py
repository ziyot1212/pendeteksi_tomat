import os
import json
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
app.secret_key = 'tomatovision_green_sphere_master_v42'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

CAMERA_SOURCE = 0
CAMERA_ACTIVE = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

LAST_SCAN_CONTEXT = {
    'has_scan': False,
    'total': 0,
    'matang': 0,
    'setengah': 0,
    'mentah': 0,
    'timestamp': None
}

# ==============================================================================
# 1. MUAT MODEL KLASIFIKASI (.h5) & PELINDUNG MANUSIA (COCO CLASS 0)
# ==============================================================================
MODEL_PATH = os.path.join('model', 'tomato_model.h5')
CLASS_MAP_PATH = os.path.join('model', 'class_indices.json')

classifier_model = load_model(MODEL_PATH)
TARGET_H = classifier_model.input_shape[1] if classifier_model.input_shape[1] is not None else 128
TARGET_W = classifier_model.input_shape[2] if classifier_model.input_shape[2] is not None else 128

LABEL_MAPPING = {
    'bukan_tomat': 'Bukan Tomat',
    'half_ripe_tomatoes': 'Setengah Matang',
    'raw_tomatoes': 'Muda/Mentah',
    'ripe_tomatoes': 'Matang'
}

NUM_CLASSES = classifier_model.output_shape[-1]
if os.path.exists(CLASS_MAP_PATH):
    try:
        with open(CLASS_MAP_PATH, 'r') as f:
            raw_indices = json.load(f)
        sorted_classes = sorted(raw_indices.items(), key=lambda item: item[1])
        CLASS_NAMES = [LABEL_MAPPING.get(name, name) for name, idx in sorted_classes]
    except Exception:
        CLASS_NAMES = ['Bukan Tomat', 'Setengah Matang', 'Muda/Mentah', 'Matang'] if NUM_CLASSES == 4 else ['Setengah Matang', 'Muda/Mentah', 'Matang']
else:
    CLASS_NAMES = ['Bukan Tomat', 'Setengah Matang', 'Muda/Mentah', 'Matang'] if NUM_CLASSES == 4 else ['Setengah Matang', 'Muda/Mentah', 'Matang']

person_guard = YOLO('yolov8n.pt')


# ==============================================================================
# 2. PEMISAH GEROMBOLAN BUAH KANAL HANGAT (MATANG & SETENGAH MATANG)
# ==============================================================================
def split_cluster_watershed(binary_mask, min_box_area, max_box_area):
    orig_h, orig_w = binary_mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

    extracted_boxes = []
    min_dim = min(orig_h, orig_w)
    peak_ksize = max(15, int(min_dim * 0.045))
    if peak_ksize % 2 == 0:
        peak_ksize += 1
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_ksize, peak_ksize))

    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        if comp_area < (min_box_area * 0.45):
            continue

        comp_mask = np.uint8(labels == i) * 255
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist < 8.0:
            continue

        dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
        dilated = cv2.dilate(dist_smooth, peak_kernel)

        min_peak_val = max(9.0, max_dist * 0.24)
        peaks = (dist_smooth >= (dilated - 1e-3)) & (dist_smooth >= min_peak_val) & (comp_mask > 0)
        peaks_u8 = cv2.dilate(np.uint8(peaks) * 255, np.ones((5, 5), np.uint8))

        n_peaks, peak_markers = cv2.connectedComponents(peaks_u8)

        if n_peaks <= 2:
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = float(bw) / float(bh + 1e-5)
            if min_box_area <= (bw * bh) <= max_box_area and (0.65 <= aspect <= 1.50):
                extracted_boxes.append([bx, by, bw, bh])
            continue

        markers = np.zeros((orig_h, orig_w), dtype=np.int32)
        markers[comp_mask == 0] = 1
        for p_id in range(1, n_peaks):
            markers[peak_markers == p_id] = p_id + 1

        inv_dist = np.uint8(255 - np.clip((dist_smooth / (max_dist + 1e-5)) * 255.0, 0, 255))
        ws_input = cv2.cvtColor(inv_dist, cv2.COLOR_GRAY2BGR)
        cv2.watershed(ws_input, markers)

        for p_id in range(1, n_peaks):
            basin_mask = np.uint8(markers == (p_id + 1)) * 255
            cnts, _ = cv2.findContours(basin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c)
            pad_x = int(bw * 0.06)
            pad_y = int(bh * 0.06)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(orig_w, bx + bw + pad_x)
            y2 = min(orig_h, by + bh + pad_y)
            fw, fh = x2 - x1, y2 - y1

            if min_box_area <= (fw * fh) <= max_box_area:
                aspect = float(fw) / float(fh + 1e-5)
                if 0.65 <= aspect <= 1.50:
                    extracted_boxes.append([x1, y1, fw, fh])

    return extracted_boxes


# ==============================================================================
# 3. DETEKTOR SPESIALIS TOMAT HIJAU (8-SECTOR SPHERICAL DOME + ANTI-DAUN/BATANG)
# ==============================================================================
def build_sphere_kernels(r):
    """
    Membangun kernel geometri bola 4-kuadran dalam & 4-kuadran cincin luar
    agar kandidat wajib melengkung seperti bola ke segala arah (bukan tepi daun/batang).
    """
    size = int(2 * round(r * 1.30) + 1)
    c = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - c) ** 2 + (y - c) ** 2)

    disk_mask = dist <= (r * 0.86)
    rim_mask = (dist >= (r * 0.90)) & (dist <= (r * 1.22))

    # 4 Kuadran simetri dalam bola
    q1 = np.float32(disk_mask & (x <= c) & (y <= c))
    q2 = np.float32(disk_mask & (x >= c) & (y <= c))
    q3 = np.float32(disk_mask & (x <= c) & (y >= c))
    q4 = np.float32(disk_mask & (x >= c) & (y >= c))
    for q in (q1, q2, q3, q4):
        q /= (q.sum() + 1e-5)

    # 4 Kuadran cincin luar (memastikan penurunan kubah terjadi di SEMUA sisi, bukan cuma 1 sisi daun)
    r1 = np.float32(rim_mask & (x <= c) & (y <= c))
    r2 = np.float32(rim_mask & (x >= c) & (y <= c))
    r3 = np.float32(rim_mask & (x <= c) & (y >= c))
    r4 = np.float32(rim_mask & (x >= c) & (y >= c))
    for rm in (r1, r2, r3, r4):
        rm /= (rm.sum() + 1e-5)

    # Cincin daging buah (0.30r s/d 0.85r) -> kebal kelopak bintang di pusat
    k_flesh = np.float32((dist >= (r * 0.30)) & (dist <= (r * 0.85)))
    k_flesh /= (k_flesh.sum() + 1e-5)

    k_rim = np.float32(rim_mask)
    k_rim /= (k_rim.sum() + 1e-5)

    return (q1, q2, q3, q4), (r1, r2, r3, r4), k_flesh, k_rim


def expand_and_center_sphere(cx, cy, init_r, solid_green, fruit_healed, orig_w, orig_h):
    """
    Menggeser titik pusat ke pusat massa bola tomat dan menyesuaikan radius
    hingga pas menyentuh tepi luar bola tomat.
    """
    best_cx, best_cy, best_r = float(cx), float(cy), float(init_r)

    for _ in range(2):
        win_r = int(round(best_r * 1.12))
        x1 = max(0, int(round(best_cx - win_r)))
        y1 = max(0, int(round(best_cy - win_r)))
        x2 = min(orig_w, int(round(best_cx + win_r)))
        y2 = min(orig_h, int(round(best_cy + win_r)))
        sub_f = fruit_healed[y1:y2, x1:x2]
        if sub_f.size == 0:
            break
        yy, xx = np.ogrid[:sub_f.shape[0], :sub_f.shape[1]]
        cy_l, cx_l = best_cy - y1, best_cx - x1
        w_mask = ((xx - cx_l) ** 2 + (yy - cy_l) ** 2) <= (win_r ** 2)
        weights = np.where(w_mask, sub_f, 0.0)
        total_w = float(weights.sum())
        if total_w > 1e-3:
            dy = float((yy * weights).sum()) / total_w - cy_l
            dx = float((xx * weights).sum()) / total_w - cx_l
            best_cx = np.clip(best_cx + dx * 0.42, 0, orig_w - 1)
            best_cy = np.clip(best_cy + dy * 0.42, 0, orig_h - 1)

    best_score = -1.0
    optimal_r = best_r
    for mult in [0.95, 1.04, 1.14, 1.25, 1.38, 1.52]:
        test_r = best_r * mult
        if test_r < 14.0 or test_r > (min(orig_w, orig_h) * 0.42):
            continue
        pad = int(round(test_r * 1.22))
        x1 = max(0, int(round(best_cx - pad)))
        y1 = max(0, int(round(best_cy - pad)))
        x2 = min(orig_w, int(round(best_cx + pad)))
        y2 = min(orig_h, int(round(best_cy + pad)))

        sub_sg = solid_green[y1:y2, x1:x2]
        sub_fh = fruit_healed[y1:y2, x1:x2]
        if sub_sg.size == 0:
            continue

        yy, xx = np.ogrid[:sub_sg.shape[0], :sub_sg.shape[1]]
        cy_l, cx_l = best_cy - y1, best_cx - x1
        d = np.sqrt((xx - cx_l) ** 2 + (yy - cy_l) ** 2)

        inner_m = d <= (test_r * 0.88)
        rim_m = (d >= (test_r * 0.92)) & (d <= (test_r * 1.18))
        if np.count_nonzero(inner_m) < 16 or np.count_nonzero(rim_m) < 12:
            continue

        green_fill = np.count_nonzero(sub_sg[inner_m]) / float(np.count_nonzero(inner_m))
        if green_fill < 0.82:
            break

        inner_val = float(np.mean(sub_fh[inner_m]))
        rim_val = float(np.mean(sub_fh[rim_m]))
        drop_score = (inner_val - rim_val) * green_fill * (test_r ** 0.12)

        if drop_score > best_score:
            best_score = drop_score
            optimal_r = test_r

    return best_cx, best_cy, optimal_r


def detect_green_tomatoes_specialized(img):
    """
    Pendeteksi Spesialis Buah Tomat Hijau (v42):
    1. Menghapus batang/tangkai tipis dengan Stem-Stripping Morphological Opening.
    2. Menambal kelopak bintang (*) secara greyscale tanpa menyatukan daun luar.
    3. Memvalidasi kubah 3D di 8 sektor arah melingkar & keselarasan gradien radial
       sehingga daun datar yang terkena cahaya matahari tidak ikut terdeteksi.
    4. Memangkas kotak serpihan kecil dengan Dominant Scale Consistency.
    """
    orig_h, orig_w = img.shape[:2]
    img_area = float(orig_h * orig_w)
    min_dim = float(min(orig_h, orig_w))
    # Batas ukuran wajar buah tomat agar serpihan daun/tangkai kecil langsung terabaikan
    min_box_area = img_area * 0.0040
    max_box_area = img_area * 0.55

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan, a_chan, b_lab = cv2.split(lab)

    b_c = img[:, :, 0].astype(np.float32)
    g_c = img[:, :, 1].astype(np.float32)
    r_c = img[:, :, 2].astype(np.float32)

    # 1. Mask Dasar Pigmen Hijau Organik
    base_green = (
        (h_chan >= 22) & (h_chan <= 88) &
        (s_chan >= 25) & (v_chan >= 40) &
        (g_c > (b_c + 8.0)) & (g_c >= (r_c * 0.72)) &
        (a_chan <= 124.0) & ((b_lab - a_chan) >= 12.0)
    )

    # Tutup pantulan kilap putih kecil di tengah permukaan buah
    k_shine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    base_green_u8 = cv2.morphologyEx(np.uint8(base_green) * 255, cv2.MORPH_CLOSE, k_shine)

    # Isi lubang internal kecil (kelopak bintang di dalam buah) sebelum menghapus batang
    cnts_h, hier_h = cv2.findContours(base_green_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier_h is not None:
        for idx, h_info in enumerate(hier_h[0]):
            if h_info[3] != -1 and cv2.contourArea(cnts_h[idx]) < (img_area * 0.035):
                cv2.drawContours(base_green_u8, cnts_h, idx, 255, -1)

    # PENGHAPUS BATANG & TANGKAI (STEM STRIPPER):
    # Operasi MORPH_OPEN menghapus batang bawah dan tangkai tipis tanpa mengurangi bola tomat
    stem_k = max(11, int(min_dim * 0.024))
    if stem_k % 2 == 0:
        stem_k += 1
    solid_green_u8 = cv2.morphologyEx(
        base_green_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stem_k, stem_k))
    )
    solid_green = solid_green_u8 > 0
    base_green = base_green_u8 > 0

    if np.count_nonzero(solid_green) < int(min_box_area * 0.7):
        return []

    # 2. Peta Skor Daging Buah Kuning-Hijau & Greyscale Calyx Inpainting
    v_clamped = np.minimum(v_chan.astype(np.float32), 210.0)
    chroma_yg = np.clip(b_lab - a_chan, 0.0, 115.0)
    ig_y = np.clip(0.58 * g_c + 0.42 * r_c - 0.78 * b_c, 0.0, 225.0)

    fruit_raw = (0.48 * ig_y) + (0.32 * (chroma_yg * 1.85)) + (0.20 * v_clamped)
    fruit_raw[~base_green] = 0.0

    calyx_k = max(11, int(min_dim * 0.026))
    if calyx_k % 2 == 0:
        calyx_k += 1
    k_calyx = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (calyx_k, calyx_k))

    fruit_healed = cv2.morphologyEx(fruit_raw, cv2.MORPH_CLOSE, k_calyx)
    fruit_healed[~solid_green] = 0.0
    fruit_healed = cv2.GaussianBlur(fruit_healed, (5, 5), 0)

    v_healed = cv2.morphologyEx(v_chan.astype(np.float32), cv2.MORPH_CLOSE, k_calyx)
    v_healed = cv2.GaussianBlur(v_healed, (5, 5), 0)

    # Peta cekungan bayangan antar-buah (Black-Hat Valley) & Gradien Tepi
    k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    valley = cv2.morphologyEx(v_healed, cv2.MORPH_BLACKHAT, k_bh)
    gx_v = cv2.Sobel(v_healed, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(v_healed, cv2.CV_32F, 0, 1, ksize=3)
    grad_v = cv2.magnitude(gx_v, gy_v)

    seam_mask = (~solid_green) | (valley >= 10.0) | ((grad_v >= 36.0) & (v_healed < 145.0))

    raw_sphere_proposals = []

    # ==========================================================================
    # JALUR A: 4-QUADRANT OMNIDIRECTIONAL DOME SCANNER (ANTI DAUN & BATANG)
    # ==========================================================================
    work_dim = 240.0
    scale = work_dim / float(max(orig_h, orig_w))
    ws_w = max(50, int(round(orig_w * scale)))
    ws_h = max(50, int(round(orig_h * scale)))

    sg_small = cv2.resize(np.float32(solid_green), (ws_w, ws_h), interpolation=cv2.INTER_AREA)
    fh_small = cv2.resize(fruit_healed, (ws_w, ws_h), interpolation=cv2.INTER_AREA)
    seam_small = cv2.resize(np.float32(seam_mask), (ws_w, ws_h), interpolation=cv2.INTER_AREA)

    # Mulai dari r=9 agar ranting/ujung daun kecil tidak pernah dipindai
    radii_grid = [9, 11, 14, 17, 21, 26, 32, 39, 47, 56]

    for r in radii_grid:
        if (2 * r) >= min(ws_w, ws_h) * 0.90:
            continue

        (q1, q2, q3, q4), (r1, r2, r3, r4), k_flesh, k_rim = build_sphere_kernels(r)

        # Kepadatan hijau solid di keempat kuadran dalam
        f_q1 = cv2.filter2D(sg_small, -1, q1, borderType=cv2.BORDER_CONSTANT)
        f_q2 = cv2.filter2D(sg_small, -1, q2, borderType=cv2.BORDER_CONSTANT)
        f_q3 = cv2.filter2D(sg_small, -1, q3, borderType=cv2.BORDER_CONSTANT)
        f_q4 = cv2.filter2D(sg_small, -1, q4, borderType=cv2.BORDER_CONSTANT)
        min_quad_fill = np.minimum(np.minimum(f_q1, f_q2), np.minimum(f_q3, f_q4))
        mean_disk_fill = 0.25 * (f_q1 + f_q2 + f_q3 + f_q4)

        # Skor daging buah di setiap kuadran dalam vs kuadran cincin luarnya
        in_q1 = cv2.filter2D(fh_small, -1, q1, borderType=cv2.BORDER_CONSTANT)
        in_q2 = cv2.filter2D(fh_small, -1, q2, borderType=cv2.BORDER_CONSTANT)
        in_q3 = cv2.filter2D(fh_small, -1, q3, borderType=cv2.BORDER_CONSTANT)
        in_q4 = cv2.filter2D(fh_small, -1, q4, borderType=cv2.BORDER_CONSTANT)

        out_r1 = cv2.filter2D(fh_small, -1, r1, borderType=cv2.BORDER_CONSTANT)
        out_r2 = cv2.filter2D(fh_small, -1, r2, borderType=cv2.BORDER_CONSTANT)
        out_r3 = cv2.filter2D(fh_small, -1, r3, borderType=cv2.BORDER_CONSTANT)
        out_r4 = cv2.filter2D(fh_small, -1, r4, borderType=cv2.BORDER_CONSTANT)

        # Hitung berapa kuadran yang benar-benar melengkung turun (kubah 3D)
        d1 = (in_q1 - out_r1) >= 4.0
        d2 = (in_q2 - out_r2) >= 4.0
        d3 = (in_q3 - out_r3) >= 4.0
        d4 = (in_q4 - out_r4) >= 4.0
        quads_dropping = d1.astype(np.int32) + d2.astype(np.int32) + d3.astype(np.int32) + d4.astype(np.int32)

        inner_score = cv2.filter2D(fh_small, -1, k_flesh, borderType=cv2.BORDER_CONSTANT)
        rim_score = cv2.filter2D(fh_small, -1, k_rim, borderType=cv2.BORDER_CONSTANT)
        rim_seam = cv2.filter2D(seam_small, -1, k_rim, borderType=cv2.BORDER_CONSTANT)
        dome_drop = inner_score - rim_score

        # Syarat Sfera Ketat:
        # 1. Keempat kuadran dalam wajib padat hijau (min_quad_fill >= 0.76, mean >= 0.86) -> Tolak batang!
        # 2. Minimal 3 dari 4 kuadran wajib melengkung turun (quads_dropping >= 3) -> Tolak daun datar!
        # 3. Kontras kubah (dome_drop >= 7.5) dan garis batas tepi (rim_seam >= 0.28)
        valid_sphere = (
            (min_quad_fill >= 0.76) &
            (mean_disk_fill >= 0.86) &
            (quads_dropping >= 3) &
            (inner_score >= 86.0) &
            (dome_drop >= 7.5) &
            (rim_seam >= 0.28)
        )

        dome_response = np.where(
            valid_sphere,
            mean_disk_fill * (inner_score / 120.0) * (rim_seam + np.clip(dome_drop / 40.0, 0.05, 1.5)),
            0.0
        )

        if dome_response.max() <= 0:
            continue

        nms_sz = max(7, int(r * 0.95))
        if nms_sz % 2 == 0:
            nms_sz += 1
        loc_max = cv2.dilate(dome_response, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_sz, nms_sz)))
        pys, pxs = np.where((dome_response >= (loc_max - 1e-4)) & (dome_response >= 0.28))

        for py, px in zip(pys, pxs):
            orig_cx = float(px) / scale
            orig_cy = float(py) / scale
            orig_r = (float(r) / scale) * 1.04
            raw_sphere_proposals.append((orig_cx, orig_cy, orig_r, float(dome_response[py, px])))

    # ==========================================================================
    # JALUR B: TOPOGRAPHIC DISTANCE PEAKS PADA SOLID GREEN (BEBAS BATANG)
    # ==========================================================================
    valid_fh_vals = fruit_healed[solid_green]
    for pct in [32, 50]:
        slice_thresh = max(84.0, float(np.percentile(valid_fh_vals, pct)))
        slice_mask = np.uint8((fruit_healed >= slice_thresh) & solid_green) * 255

        cnts_s, hier_s = cv2.findContours(slice_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_s is not None:
            for idx, h_info in enumerate(hier_s[0]):
                if h_info[3] != -1 and cv2.contourArea(cnts_s[idx]) < (img_area * 0.04):
                    cv2.drawContours(slice_mask, cnts_s, idx, 255, -1)

        deep_seam = (valley >= 12.0) & (v_healed < 135.0)
        slice_mask[deep_seam] = 0
        slice_mask = cv2.morphologyEx(slice_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        dist_s = cv2.distanceTransform(slice_mask, cv2.DIST_L2, 5)
        if dist_s.max() < 12.0:
            continue

        pk_s = max(15, int(min_dim * 0.045))
        if pk_s % 2 == 0:
            pk_s += 1
        dil_s = cv2.dilate(dist_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk_s, pk_s)))
        p_mask = (dist_s >= (dil_s - 1e-3)) & (dist_s >= max(12.0, min_dim * 0.028)) & (slice_mask > 0)
        n_p, p_lbl = cv2.connectedComponents(np.uint8(p_mask) * 255)

        for p_id in range(1, n_p):
            ys, xs = np.where(p_lbl == p_id)
            if len(xs) == 0:
                continue
            px, py = float(np.mean(xs)), float(np.mean(ys))
            r_init = float(dist_s[int(py), int(px)]) * 1.15
            rcx, rcy, rr = expand_and_center_sphere(px, py, r_init, solid_green, fruit_healed, orig_w, orig_h)
            raw_sphere_proposals.append((rcx, rcy, rr, 0.50))

    if not raw_sphere_proposals:
        return []

    # ==========================================================================
    # VALIDASI BIOMETRIK 8-SEKTOR & KESELARASAN GRADIEN MELINGKAR
    # ==========================================================================
    # Peta frekuensi tinggi untuk mendeteksi urat daun & gerigi daun
    v_blur_wide = cv2.GaussianBlur(v_healed, (15, 15), 0)
    vein_clutter = np.abs(v_healed - v_blur_wide)

    verified_spheres = []

    for (cx, cy, rad, init_sc) in raw_sphere_proposals:
        x1 = max(0, int(round(cx - rad)))
        y1 = max(0, int(round(cy - rad)))
        x2 = min(orig_w, int(round(cx + rad)))
        y2 = min(orig_h, int(round(cy + rad)))
        bw, bh = x2 - x1, y2 - y1
        box_area = float(bw * bh)

        if not (min_box_area <= box_area <= max_box_area):
            continue

        aspect = float(bw) / float(bh + 1e-5)
        if not (0.76 <= aspect <= 1.32):
            continue

        # Ambil jendela kerja sedikit lebih lebar (1.18x radius) untuk mengecek penurunan tepi luar
        pad_r = int(round(rad * 1.18))
        wx1 = max(0, int(round(cx - pad_r)))
        wy1 = max(0, int(round(cy - pad_r)))
        wx2 = min(orig_w, int(round(cx + pad_r)))
        wy2 = min(orig_h, int(round(cy + pad_r)))
        wh, ww = wy2 - wy1, wx2 - wx1
        if wh < 20 or ww < 20:
            continue

        sub_sg = solid_green[wy1:wy2, wx1:wx2]
        sub_fh = fruit_healed[wy1:wy2, wx1:wx2]
        sub_vh = v_healed[wy1:wy2, wx1:wx2]
        sub_s = s_chan[wy1:wy2, wx1:wx2]
        sub_ig = ig_y[wy1:wy2, wx1:wx2]
        sub_ch = chroma_yg[wy1:wy2, wx1:wx2]
        sub_vc = vein_clutter[wy1:wy2, wx1:wx2]
        sub_seam = seam_mask[wy1:wy2, wx1:wx2]
        sub_gx = gx_v[wy1:wy2, wx1:wx2]
        sub_gy = gy_v[wy1:wy2, wx1:wx2]

        yy, xx = np.ogrid[:wh, :ww]
        cy_l, cx_l = cy - wy1, cx - wx1
        dy = yy - cy_l
        dx = xx - cx_l
        dist_l = np.sqrt(dx ** 2 + dy ** 2) + 1e-5

        disk_m = dist_l <= (rad * 0.84)
        core_m = dist_l <= (rad * 0.65)
        flesh_ring_m = (dist_l >= (rad * 0.32)) & (dist_l <= (rad * 0.84))
        outer_rim_m = (dist_l >= (rad * 0.88)) & (dist_l <= (rad * 1.15))

        if np.count_nonzero(disk_m) < 25 or np.count_nonzero(flesh_ring_m) < 16 or np.count_nonzero(outer_rim_m) < 16:
            continue

        # 1. Kepadatan Inti & 4-Kuadran Solid (Menolak batang yang berlatar kosong)
        core_fill = np.count_nonzero(sub_sg[core_m]) / float(np.count_nonzero(core_m))
        if core_fill < 0.88:
            continue

        q1_m = disk_m & (dx <= 0) & (dy <= 0)
        q2_m = disk_m & (dx >= 0) & (dy <= 0)
        q3_m = disk_m & (dx <= 0) & (dy >= 0)
        q4_m = disk_m & (dx >= 0) & (dy >= 0)
        q_fills = [
            np.count_nonzero(sub_sg[qm]) / float(np.count_nonzero(qm) + 1e-5)
            for qm in (q1_m, q2_m, q3_m, q4_m)
        ]
        if min(q_fills) < 0.78 or np.mean(q_fills) < 0.86:
            continue

        # 2. UJI KUBAH 3D 8-SEKTOR MELINGKAR (Mencegah daun datar terdeteksi)
        # Bagi lingkaran menjadi 8 sektor sudut (@45 derajat). Pada bola tomat asli,
        # minimal 6 dari 8 sektor harus menunjukkan penurunan skor dari dalam ke cincin luar
        # atau menyentuh garis batas (seam)!
        angles = np.arctan2(dy, dx)  # [-pi, pi]
        sector_bins = np.floor(((angles + np.pi) / (2.0 * np.pi)) * 8.0).astype(np.int32) % 8

        inner_dome_val = float(np.percentile(sub_fh[core_m], 70))
        valid_sectors = 0
        for s_idx in range(8):
            sec_rim = outer_rim_m & (sector_bins == s_idx)
            sec_in = flesh_ring_m & (sector_bins == s_idx)
            if np.count_nonzero(sec_rim) < 3 or np.count_nonzero(sec_in) < 3:
                continue
            sec_in_val = float(np.mean(sub_fh[sec_in]))
            sec_rim_val = float(np.mean(sub_fh[sec_rim]))
            sec_seam_ratio = np.count_nonzero(sub_seam[sec_rim]) / float(np.count_nonzero(sec_rim))

            if (sec_in_val - sec_rim_val) >= 5.5 or (inner_dome_val - sec_rim_val) >= 11.0 or sec_seam_ratio >= 0.38:
                valid_sectors += 1

        if valid_sectors < 6:
            continue

        # 3. KESELARASAN GRADIEN TEPI MELINGKAR (Outward Radial Gradient Coherence)
        # Di sepanjang tepi bola tomat, arah perubahan gelap-terang mengarah keluar secara radial
        ux = dx / dist_l
        uy = dy / dist_l
        # Proyeksi gradien yang mengarah ke dalam pusat bola (-gx * ux - gy * uy)
        radial_grad = -(sub_gx * ux + sub_gy * uy)
        rim_pos_grad_ratio = np.count_nonzero(radial_grad[outer_rim_m] > 4.0) / float(np.count_nonzero(outer_rim_m))
        if rim_pos_grad_ratio < 0.36:
            continue

        # 4. Spektrum Warna & Kilap Kubah 3D Kulit Buah Tomat
        ig_75 = float(np.percentile(sub_ig[flesh_ring_m], 75))
        v_80 = float(np.percentile(sub_vh[core_m], 80))
        s_75 = float(np.percentile(sub_s[flesh_ring_m], 75))
        ch_75 = float(np.percentile(sub_ch[flesh_ring_m], 75))

        # Tomat hijau memiliki pantulan kubah yang cerah (v_80 >= 102) & rona kuning-hijau segar
        if v_80 < 102.0 or ig_75 < 82.0 or ch_75 < 22.0 or s_75 < 45.0:
            continue

        rim_v_50 = float(np.median(sub_vh[outer_rim_m]))
        rim_fh_50 = float(np.median(sub_fh[outer_rim_m]))
        dome_fh_75 = float(np.percentile(sub_fh[core_m], 75))

        if v_80 < (rim_v_50 * 1.05) or dome_fh_75 < (rim_fh_50 + 8.5):
            continue

        # 5. Kemulusan Kulit Buah vs Urat Daun Kasar
        clutter_mean = float(np.mean(sub_vc[flesh_ring_m]))
        if clutter_mean > 11.5:
            continue

        # 6. Filter Anti-Mangga (Mangga memanjang vertikal ke bawah dengan kulit hijau lilin pucat)
        y_ext = min(orig_h, y2 + int(bh * 0.38))
        if (y_ext - y2) > 6:
            bot_band = fruit_healed[y2:y_ext, x1 + int(bw * 0.2):x2 - int(bw * 0.2)]
            if bot_band.size > 0:
                cont_ratio = np.count_nonzero(bot_band >= (dome_fh_75 * 0.78)) / float(bot_band.size)
                if cont_ratio > 0.70 and ch_75 < 36.0 and s_75 < 95.0:
                    continue

        final_q = (
            init_sc *
            float(np.mean(q_fills)) *
            (valid_sectors / 8.0) *
            rim_pos_grad_ratio *
            ((dome_fh_75 - rim_fh_50 + 10.0) / 35.0) *
            (box_area ** 0.12)
        )
        verified_spheres.append(([x1, y1, bw, bh], final_q))

    if not verified_spheres:
        return []

    # ==========================================================================
    # NMS CERDAS & DOMINANT FRUIT SCALE PRUNING (PEMANGKAS SERPIHAN KECIL)
    # ==========================================================================
    verified_spheres.sort(key=lambda item: item[1], reverse=True)

    # Cari ukuran buah utama terbesar (di antara 3 kandidat dengan skor sfera tertinggi)
    top_areas = [ float(b[2] * b[3]) for b, _ in verified_spheres[:3] ]
    max_dominant_area = max(top_areas) if top_areas else 0.0

    kept_boxes = []

    for box, q_sc in verified_spheres:
        bx1, by1, bw, bh = box
        bx2, by2 = bx1 + bw, by1 + bh
        b_area = float(bw * bh)
        cx, cy = bx1 + bw * 0.5, by1 + bh * 0.5

        # Jika kotak ini jauh lebih kecil (< 24% luas buah utama di gambar yang sama),
        # maka itu adalah serpihan tangkai/anak daun dan langsung dibuang!
        if max_dominant_area > 0 and b_area < (max_dominant_area * 0.24):
            continue

        is_dup = False
        for kbox in kept_boxes:
            kx1, ky1, kw, kh = kbox
            kx2, ky2 = kx1 + kw, ky1 + kh
            k_area = float(kw * kh)
            kcx, kcy = kx1 + kw * 0.5, ky1 + kh * 0.5

            center_dist = np.hypot(cx - kcx, cy - kcy)
            max_rad = 0.5 * max(max(bw, bh), max(kw, kh))
            min_rad = 0.5 * min(max(bw, bh), max(kw, kh))

            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou = inter / float(b_area + k_area - inter + 1e-5)
            io_min = inter / float(min(b_area, k_area) + 1e-5)

            if iou > 0.24 or io_min > 0.45 or center_dist < max(min_rad * 0.88, max_rad * 0.62):
                is_dup = True
                break

        if not is_dup:
            kept_boxes.append(box)

    return kept_boxes


# ==============================================================================
# 4. PENGGABUNG KANAL DETEKSI (WARM CHANNEL + SPECIALIZED GREEN CHANNEL v42)
# ==============================================================================
def detect_all_tomato_candidates(img):
    orig_h, orig_w = img.shape[:2]
    img_area = float(orig_h * orig_w)
    min_box_area = img_area * 0.0035
    max_box_area = img_area * 0.75

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    b_chan = img[:, :, 0].astype(np.float32)
    g_chan = img[:, :, 1].astype(np.float32)
    r_chan = img[:, :, 2].astype(np.float32)

    # --------------------------------------------------------------------------
    # KANAL A: TOMAT MATANG & SETENGAH MATANG (DENGAN PENAPIS TIANG KAYU)
    # --------------------------------------------------------------------------
    red_bgr = (r_chan > 85) & ((r_chan - g_chan) >= 38) & ((r_chan - b_chan) >= 42) & ((r_chan / (g_chan + 1.0)) >= 1.42)
    red_hsv = ((h_chan <= 13) | (h_chan >= 163)) & (s_chan >= 65) & (v_chan >= 40)
    mask_red = np.uint8(red_bgr & red_hsv) * 255

    # Oranye Karoten: Wajib memiliki dominasi R atas G (R - G >= 28) agar tiang kayu cokelat-kuning tertolak
    org_bgr = (
        (r_chan > 130) & (g_chan > 70) &
        ((r_chan - g_chan) >= 28) &
        ((r_chan - b_chan) >= 72) &
        ((g_chan - b_chan) >= 32)
    )
    org_hsv = (h_chan >= 14) & (h_chan <= 25) & (s_chan >= 120) & (v_chan >= 110)
    mask_orange = np.uint8(org_bgr & org_hsv) * 255

    mask_warm = mask_red | mask_orange

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    dark_creases = np.uint8((grad_mag > 65) & (v_chan < 95)) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_warm = cv2.morphologyEx(mask_warm, cv2.MORPH_CLOSE, k_close)
    mask_warm[dark_creases > 0] = 0
    mask_warm = cv2.morphologyEx(mask_warm, cv2.MORPH_OPEN, k_open)

    warm_boxes = split_cluster_watershed(mask_warm, min_box_area, max_box_area)

    # --------------------------------------------------------------------------
    # KANAL B: TOMAT HIJAU / MUDA MENTAH (8-SECTOR SPHERICAL DETECTOR v42)
    # --------------------------------------------------------------------------
    green_boxes = detect_green_tomatoes_specialized(img)

    all_candidates = []
    for b in warm_boxes:
        all_candidates.append((b, 'warm'))
    for b in green_boxes:
        all_candidates.append((b, 'green'))

    return all_candidates


# ==============================================================================
# 5. VALIDASI BIOMETRIK & KLASIFIKASI KEMATANGAN
# ==============================================================================
def validate_and_classify(crop, box_coords, channel_type, person_boxes):
    x1, y1, x2, y2 = box_coords
    bw, bh = x2 - x1, y2 - y1
    box_area = float(bw * bh)
    if box_area <= 0 or crop.size == 0:
        return None, 0.0

    # 1. Cegah area tubuh manusia
    for px1, py1, px2, py2 in person_boxes:
        ix1, iy1 = max(x1, px1), max(y1, py1)
        ix2, iy2 = min(x2, px2), min(y2, py2)
        if ix2 > ix1 and iy2 > iy1:
            if ((ix2 - ix1) * (iy2 - iy1)) / box_area > 0.22:
                return None, 0.0

    # 2. Penapisan Nisbah Aspek Sfera
    aspect = float(bw) / float(bh + 1e-5)
    if not (0.65 <= aspect <= 1.50):
        return None, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # 3. Tolak Buah Ceri
    specular = np.count_nonzero((s_chan < 35) & (v_chan > 230)) / box_area
    crimson_dark = np.count_nonzero(cv2.inRange(hsv, np.array([150, 55, 20]), np.array([177, 255, 125]))) / box_area
    if specular > 0.045 and crimson_dark > 0.12:
        return None, 0.0

    # 4. Prediksi Model CNN (.h5)
    crop_resized = cv2.resize(crop, (TARGET_W, TARGET_H))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(crop_rgb.astype(np.float32) / 255.0, axis=0)
    preds = classifier_model.predict(input_tensor, verbose=0)[0]

    class_idx = int(np.argmax(preds))
    cnn_label = CLASS_NAMES[class_idx]
    cnn_conf = float(preds[class_idx]) * 100.0

    bukan_prob = 0.0
    if 'Bukan Tomat' in CLASS_NAMES:
        bukan_prob = float(preds[CLASS_NAMES.index('Bukan Tomat')])

    cy1, cy2 = int(bh * 0.15), int(bh * 0.85)
    cx1, cx2 = int(bw * 0.15), int(bw * 0.85)
    core_hsv = hsv[cy1:cy2, cx1:cx2]
    core_bgr = crop[cy1:cy2, cx1:cx2].astype(np.float32)
    if core_hsv.size == 0:
        return None, 0.0

    core_px = float(core_hsv.shape[0] * core_hsv.shape[1])

    red_mask = (
        cv2.inRange(core_hsv, np.array([0, 60, 40]), np.array([13, 255, 255])) |
        cv2.inRange(core_hsv, np.array([163, 60, 40]), np.array([180, 255, 255]))
    )
    red_ratio = np.count_nonzero(red_mask) / core_px

    org_mask = cv2.inRange(core_hsv, np.array([14, 110, 100]), np.array([25, 255, 255]))
    org_ratio = np.count_nonzero(org_mask) / core_px

    grn_mask = cv2.inRange(core_hsv, np.array([22, 25, 40]), np.array([88, 255, 255]))
    grn_ratio = np.count_nonzero(grn_mask) / core_px

    # JALUR HANGAT (MATANG & SETENGAH MATANG)
    if channel_type == 'warm':
        mean_r = float(np.mean(core_bgr[:, :, 2]))
        mean_g = float(np.mean(core_bgr[:, :, 1]))

        if red_ratio >= 0.22 and (mean_r / (mean_g + 1.0)) >= 1.38:
            final_label = 'Matang'
            final_conf = max(cnn_conf, min(98.5, 86.0 + red_ratio * 15.0))
            return final_label, final_conf
        elif org_ratio >= 0.22 and (mean_r - mean_g) >= 25.0 and (mean_r / (mean_g + 1.0)) >= 1.22:
            # Tolak tiang kayu/ranting cokelat yang ditebak 'Bukan Tomat' oleh CNN
            if cnn_label == 'Bukan Tomat' or bukan_prob > 0.45:
                return None, 0.0
            final_label = 'Setengah Matang'
            final_conf = max(cnn_conf, min(96.0, 84.0 + org_ratio * 15.0))
            return final_label, final_conf
        else:
            return None, 0.0

    # JALUR HIJAU KHUSUS TOMAT MUDA / MENTAH
    else:
        if grn_ratio >= 0.60:
            raw_prob = 0.0
            if 'Muda/Mentah' in CLASS_NAMES:
                raw_prob = float(preds[CLASS_NAMES.index('Muda/Mentah')]) * 100.0
            final_conf = max(raw_prob, min(98.0, 88.0 + grn_ratio * 9.5))
            return 'Muda/Mentah', final_conf
        return None, 0.0


# ==============================================================================
# 6. PIPELINE UTAMA PEMROSESAN FRAME
# ==============================================================================
def process_frame(img, is_live=False):
    global LAST_SCAN_CONTEXT
    orig_h, orig_w = img.shape[:2]

    guard_results = person_guard.predict(
        source=img, classes=[0], conf=0.35, device='cpu', verbose=False
    )
    person_boxes = [box.xyxy[0].cpu().numpy().astype(int) for r in guard_results for box in r.boxes]

    candidates = detect_all_tomato_candidates(img)
    raw_boxes = [c[0] for c in candidates]
    chan_types = [c[1] for c in candidates]

    final_candidates = []
    if len(raw_boxes) > 0:
        scores = [0.95 if t == 'warm' else 0.90 for t in chan_types]
        indices = cv2.dnn.NMSBoxes(raw_boxes, scores, score_threshold=0.20, nms_threshold=0.32)
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = raw_boxes[idx]
                final_candidates.append(([bx, by, bx + bw, by + bh], chan_types[idx]))

    final_candidates = sorted(final_candidates, key=lambda item: (item[0][1], item[0][0]))

    detected_objects = []
    c_matang, c_setengah, c_mentah = 0, 0, 0

    for (x1, y1, x2, y2), c_type in final_candidates:
        crop = img[y1:y2, x1:x2]
        final_label, confidence = validate_and_classify(crop, [x1, y1, x2, y2], c_type, person_boxes)
        if final_label is None:
            continue

        if final_label == 'Matang':
            box_color, badge_type = (46, 204, 113), "matang"
            c_matang += 1
        elif final_label == 'Setengah Matang':
            box_color, badge_type = (0, 215, 255), "setengah"
            c_setengah += 1
        else:
            box_color, badge_type = (39, 110, 243), "mentah"
            c_mentah += 1

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2 if is_live else 3)
        text = f"{final_label}: {confidence:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), box_color, -1)
        cv2.putText(img, text, (x1 + 3, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        detected_objects.append({
            'label': final_label,
            'confidence': f"{confidence:.1f}%",
            'type': badge_type
        })

    LAST_SCAN_CONTEXT = {
        'has_scan': True,
        'total': len(detected_objects),
        'matang': c_matang,
        'setengah': c_setengah,
        'mentah': c_mentah,
        'timestamp': time.strftime("%H:%M:%S")
    }

    if is_live:
        hud_bg = img.copy()
        cv2.rectangle(hud_bg, (10, 10), (orig_w - 10, 48), (20, 24, 33), -1)
        cv2.addWeighted(hud_bg, 0.75, img, 0.25, 0, img)
        hud_text = f"LIVE | Total: {len(detected_objects)} | Matang: {c_matang} | Setengah: {c_setengah} | Mentah: {c_mentah}"
        cv2.putText(img, hud_text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    return img, detected_objects


# ==============================================================================
# 7. ENGINE CHAT AI SPESIALIS TOMAT
# ==============================================================================
def generate_tomato_specialist_response(user_msg):
    msg = user_msg.lower().strip()
    ctx = LAST_SCAN_CONTEXT

    scan_summary = (
        f"Total terdeteksi: {ctx['total']} buah "
        f"(Matang: {ctx['matang']}, Setengah Matang: {ctx['setengah']}, Muda/Mentah: {ctx['mentah']})."
        if ctx['has_scan'] else "Belum ada gambar tomat yang dianalisis pada sesi ini."
    )

    if GEMINI_API_KEY:
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            sys_prompt = (
                "Kamu adalah 'Dr. Lycopersicon', AI Pakar Spesialis Tomat di TomatoVision AI. "
                "Tugasmu hanya menjawab tentang tomat: tingkat kematangan (Matang, Setengah Matang, Muda/Mentah), "
                "kandungan gizi likopen/karoten, penyimpanan pasca-panen, hama penyakit kebun, dan pemupukan tomat. "
                f"Data deteksi terakhir sistem saat ini: {scan_summary}. "
                "Gunakan bahasa Indonesia yang jelas, ramah, dan ilmiah."
            )
            payload = {
                "system_instruction": {"parts": [{"text": sys_prompt}]},
                "contents": [{"parts": [{"text": user_msg}]}]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=7) as resp:
                res_data = json.loads(resp.read().decode('utf-8'))
                return res_data['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            pass

    if any(k in msg for k in ['hasil', 'analisis', 'barusan', 'scan', 'deteksi', 'foto', 'rekomendasi']):
        if not ctx['has_scan']:
            return "Kamu belum melakukan pemindaian tomat. Arahkan kamera atau unggah foto tomat terlebih dahulu!"
        return (
            f"📊 **Analisis Hasil Deteksi Terakhir ({ctx['timestamp']}):**\n"
            f"• **Total Buah:** {ctx['total']} buah\n"
            f"• **Matang (Merah):** {ctx['matang']} buah\n"
            f"• **Setengah Matang (Oranye):** {ctx['setengah']} buah\n"
            f"• **Muda / Mentah (Hijau):** {ctx['mentah']} buah\n\n"
            "💡 **Rekomendasi Tindakan:**\n"
            "1. Buah **Matang** segera konsumsi atau simpan pada suhu sejuk 10–13°C.\n"
            "2. Buah **Setengah Matang** simpan di suhu ruang 20°C agar matang sempurna dalam 2–3 hari.\n"
            "3. Buah **Mentah** biarkan di pohon atau peram bersama buah penghasil gas etilen (seperti pisang)."
        )

    if any(k in msg for k in ['matang', 'mentah', 'muda', 'setengah', 'fase', 'warna', 'likopen']):
        return (
            "🍅 **3 Kategori Tingkat Kematangan Tomat:**\n\n"
            "1. **Muda / Mentah (Hijau):** Didominasi klorofil, tekstur padat keras. Siap diperam jika sudah fase *mature green*.\n"
            "2. **Setengah Matang (Oranye/Kuning):** Fase *turning/breaker*, klorofil mulai terurai menjadi beta-karoten. Cocok untuk pengiriman jarak jauh.\n"
            "3. **Matang (Merah):** Likopen terbentuk optimal (>90%), rasa manis-asam seimbang, siap dikonsumsi."
        )

    if any(k in msg for k in ['simpan', 'kulkas', 'awet', 'busuk', 'suhu']):
        return (
            "❄️ **Panduan Penyimpanan Tomat:**\n\n"
            "• **Tomat Mentah & Setengah Matang:** Simpan di suhu ruang (18–22°C). **Jangan masukkan kulkas**, karena suhu dingin (<10°C) memicu *chilling injury* yang membuat tomat hambar dan bertepung.\n"
            "• **Tomat Matang Penuh:** Simpan di tempat sejuk 10–13°C dengan posisi tangkai menghadap ke atas."
        )

    if any(k in msg for k in ['penyakit', 'hama', 'busuk', 'layu', 'kuning', 'bercak']):
        return (
            "🌿 **Penyakit Umum Tanaman Tomat & Solusinya:**\n\n"
            "1. **Busuk Ujung Buah (*Blossom-End Rot*):** Pantat tomat hitam akibat kekurangan kalsium (Ca). Berikan kalsium nitrat dan jaga kelembapan tanah stabil.\n"
            "2. **Hawar Daun (*Late Blight*):** Bercak basah kehitaman akibat jamur *Phytophthora*. Semprot fungisida mankozeb dan kurangi kelembapan tajuk.\n"
            "3. **Kutu Kebul:** Vektor virus kuning keriting. Pasang perangkap likat kuning (*yellow sticky trap*)."
        )

    return (
        "Halo! Aku **Dr. Lycopersicon**, AI Spesialis Tomat 🍅.\n"
        f"📌 *Status Scan Terakhir:* {scan_summary}\n\n"
        "Ada yang bisa aku bantu seputar analisis kematangan buah, cara simpan tomat, atau perawatan tanaman tomat di kebun?"
    )


@app.route('/chat_ai', methods=['POST'])
def chat_ai():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': 'Silakan ketik pertanyaanmu seputar buah atau tanaman tomat!'})

    reply = generate_tomato_specialist_response(user_message)
    return jsonify({
        'reply': reply,
        'scan_context': LAST_SCAN_CONTEXT
    })


# ==============================================================================
# 8. ROUTE FLASK BAWAAN
# ==============================================================================
def create_error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 35, 45)
    cv2.putText(frame, "KAMERA TIDAK TERHUBUNG", (135, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 76, 231), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (75, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 205, 215), 1, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def generate_frames():
    global CAMERA_SOURCE, CAMERA_ACTIVE
    CAMERA_ACTIVE = True

    if isinstance(CAMERA_SOURCE, int):
        camera = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_SOURCE)
    else:
        camera = cv2.VideoCapture(CAMERA_SOURCE)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        err_bytes = create_error_frame(f"Gagal membuka sumber: {CAMERA_SOURCE}.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + err_bytes + b'\r\n')
        return

    current_source = CAMERA_SOURCE
    try:
        while CAMERA_ACTIVE:
            if CAMERA_SOURCE != current_source:
                break

            success, frame = camera.read()
            if not success:
                err_bytes = create_error_frame("Sinyal video terputus.")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + err_bytes + b'\r\n')
                break

            if CAMERA_SOURCE == 0:
                frame = cv2.flip(frame, 1)
            else:
                frame = cv2.resize(frame, (640, 480))

            processed_frame, _ = process_frame(frame, is_live=True)

            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)
    finally:
        camera.release()


@app.route('/')
def index():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False
    return render_template('index.html', mode='upload', current_cam=CAMERA_SOURCE, img_path=None, detections=None)


@app.route('/live_view')
def live_view():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = True
    return render_template('index.html', mode='live', current_cam=CAMERA_SOURCE)


@app.route('/set_camera', methods=['POST'])
def set_camera():
    global CAMERA_SOURCE, CAMERA_ACTIVE
    cam_input = request.form.get('cam_source', '0').strip()

    if cam_input.isdigit():
        CAMERA_SOURCE = int(cam_input)
    else:
        if cam_input.startswith('http') and not cam_input.endswith('/video'):
            cam_input = cam_input.rstrip('/') + '/video'
        CAMERA_SOURCE = cam_input

    CAMERA_ACTIVE = True
    return redirect(url_for('live_view'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['POST'])
def predict():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False

    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filename = f"{int(time.time())}_{secure_filename(file.filename) or 'tomato.jpg'}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    img = cv2.imread(input_path)
    if img is None:
        return redirect(url_for('index'))

    h, w = img.shape[:2]
    if max(h, w) > 800:
        scale = 800 / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    hist_filename = f"hist_{filename}.png"
    hist_path = os.path.join(RESULT_FOLDER, hist_filename)

    processed_img, results = process_frame(img, is_live=False)
    cv2.imwrite(output_path, processed_img)

    plt.figure(figsize=(6, 3.8), facecolor='#ffffff')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for col, lab, c_hex in zip([0, 1, 2], ['Red', 'Green', 'Blue'], ['#e53935', '#43a047', '#1e88e5']):
        hist = cv2.calcHist([img_rgb], [col], None, [256], [0, 256])
        plt.plot(hist / hist.max() if hist.max() > 0 else hist, color=c_hex, label=f'{lab} Channel', linewidth=2)
    plt.title('Sebaran Intensitas Warna RGB', fontsize=11, fontweight='bold')
    plt.xlabel('Piksel')
    plt.ylabel('Frekuensi')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=130)
    plt.close('all')

    stats = {
        'total': len(results),
        'matang': sum(1 for d in results if d['label'] == 'Matang'),
        'setengah': sum(1 for d in results if d['label'] == 'Setengah Matang'),
        'mentah': sum(1 for d in results if d['label'] == 'Muda/Mentah')
    }

    return render_template('index.html',
                           mode='upload',
                           current_cam=CAMERA_SOURCE,
                           img_path=os.path.join('results', output_filename).replace('\\', '/'),
                           hist_path=os.path.join('results', hist_filename).replace('\\', '/'),
                           detections=results,
                           stats=stats,
                           download_filename=output_filename)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


@app.route('/reset')
def reset():
    global CAMERA_ACTIVE
    CAMERA_ACTIVE = False
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)