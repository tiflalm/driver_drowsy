import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import time
import math
from imutils import face_utils
from pygame import mixer
import imutils
import requests
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
from dotenv import load_dotenv
import os

# Memuat environment variables dari file .env
load_dotenv()

API_KEY = os.getenv('API_KEY')
SERVER_URL = os.getenv('SERVER_URL')

# Debugging: Pastikan variabel dimuat dengan benar
print(f"API_KEY: {API_KEY}")
print(f"SERVER_URL: {SERVER_URL}")

if not API_KEY:
    print("API_KEY tidak ditemukan. Pastikan sudah diatur di file .env.")
if not SERVER_URL:
    print("SERVER_URL tidak ditemukan. Pastikan sudah diatur di file .env.")

# Periksa apakah SERVER_URL sudah benar
if not SERVER_URL:
    raise ValueError("SERVER_URL tidak diatur. Periksa file .env Anda.")

# Inisialisasi mixer untuk alarm
mixer.init()
mixer.music.load("music.wav")


def calculate_duration(start_time):
    # Hitung durasi dalam detik
    total_seconds = time.time() - start_time

    # Konversi menjadi jam, menit, dan detik
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format hasil durasi
    if hours > 0:
        return f"{int(hours)} jam, {int(minutes)} menit, {int(seconds)} detik"
    elif minutes > 0:
        return f"{int(minutes)} menit, {int(seconds)} detik"
    else:
        return f"{int(seconds)} detik"


# Fungsi untuk mengirim data deteksi ke server
def send_detection(driver_id, eye_state, mouth_state, head_pose, yawning, drowsiness_status, start_time, end_time,
                   duration):
    try:
        # Konversi numpy.bool_ ke bool bawaan Python
        yawning = bool(yawning)
        drowsiness_status = "Drowsy"

        payload = {
            'driver_id': driver_id,
            'eye_state': eye_state,
            'mouth_state': mouth_state,
            'head_pose': head_pose,
            'yawning': yawning,
            'drowsiness_status': drowsiness_status,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration
        }
        headers = {
            'x-api-key': API_KEY,
            'Content-Type': 'application/json'
        }
        print(f"Mengirim permintaan POST ke {SERVER_URL} dengan payload: {payload} dan headers: {headers}")
        response = requests.post(SERVER_URL, json=payload, headers=headers)
        if response.status_code == 200:
            print('Data berhasil dikirim ke server.')
        else:
            print(f'Gagal mengirim data ke server. Status code: {response.status_code}')
            print(f'Response: {response.text}')
    except Exception as e:
        print(f'Error saat mengirim data ke server: {e}')


# Menghitung Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Menghitung Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


# Menghitung bounding box dari landmark
def calculate_bbox(landmarks):
    x_min = np.min(landmarks[:, 0])
    x_max = np.max(landmarks[:, 0])
    y_min = np.min(landmarks[:, 1])
    y_max = np.max(landmarks[:, 1])
    return x_min, y_min, x_max, y_max


# Menghitung sudut kemiringan kepala berdasarkan landmark mata dan hidung
def head_tilt_angle(landmarks):
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    nose = landmarks[27]
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle


# Mengaplikasikan Gamma Correction
def apply_gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# CLAHE untuk peningkatan kontras lokal
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# YOLOv8
model = YOLO('yolov8n.pt')  # Pastikan model YOLOv8 tersedia di direktori yang sama atau path yang benar

# Deteksi wajah dlib dan prediktor landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Threshold EAR untuk deteksi kantuk
thresh = 0.25
ear_thresholds = {
    'Low': 0.23,
    'Normal_Low': 0.25,
    'Normal_High': 0.30,
    'High': 0.32
}

mar_thresholds = {
    'Normal': 0.30,
    'High': 0.35
}

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Menginisialisasi video capture
cap = cv2.VideoCapture(0)
flag = 0

# Menghitung FPS
if not cap.isOpened():
    print("Gagal membuka file video")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps}')


# Variabel untuk melacak waktu saat mata, mulut, dan kemiringan kepala
start_time_mouth = time.time()
start_time_eye = time.time()
head_tilted_start = time.time()
start_time_drowsy = time.time()

# Definisikan warna untuk berbagai jenis objek
WARNA_ORANG = (0, 255, 0)  # Hijau
WARNA_WAJAH = (255, 0, 0)  # Biru
WARNA_MATA = (0, 255, 255)  # Kuning
WARNA_MULUT = (255, 0, 255)  # Magenta

# Inisialisasi variabel untuk menghindari pengiriman data berlebihan
previous_status = None


# Inisialisasi variabel untuk melacak waktu mulai drowsiness
start_time_drowsy = None  # Waktu mulai saat kondisi drowsy pertama kali terdeteksi


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inisialisasi ulang status drowsy pada setiap iterasi frame
    drowsy = False
    current_drowsy = False

    # Penyesuaian pencahayaan menggunakan Gamma Correction
    frame = apply_gamma_correction(frame, gamma=1.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)

    # Proses frame untuk deteksi YOLO
    hasil = model(frame)

    drowsy_detected = False  # Menandai apakah kondisi drowsy terdeteksi pada frame ini

    for deteksi in hasil[0].boxes:
        x1, y1, x2, y2 = deteksi.xyxy[0].cpu().numpy()[:4]
        conf = deteksi.conf[0]
        cls = deteksi.cls[0]

        if conf > 0.5 and int(cls) == 0:
            nama_kelas = model.names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), WARNA_ORANG, 2)
            label = f'{nama_kelas} {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNA_ORANG, 2)

            # Ekstrak ROI untuk analisis lebih lanjut
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            abu_abu = gray[int(y1):int(y2), int(x1):int(x2)]
            deteksi_wajah = detector(abu_abu)
            for deteksi_wajah_single in deteksi_wajah:
                shape = predictor(abu_abu, deteksi_wajah_single)
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                # Menghitung EAR, MAR, dan kemiringan kepala (angle)
                mata_kiri = landmarks[lStart:lEnd]
                mata_kanan = landmarks[rStart:rEnd]
                ear_left = eye_aspect_ratio(mata_kiri)
                ear_right = eye_aspect_ratio(mata_kanan)
                mar = mouth_aspect_ratio(landmarks[48:68])

                # Hitung sudut kemiringan kepala dan simpan di variabel `angle`
                angle = head_tilt_angle(landmarks)

                # Deteksi drowsiness
                if mar > 0.5 or ear_left < thresh or ear_right < thresh or abs(angle) > 15:
                    drowsy_detected = True
                    drowsy = True
                    start_time_drowsy = time.time()  # Setel start_time_drowsy setiap kali drowsiness terdeteksi
                    cv2.putText(frame, 'PERINGATAN: PENGEMUDI MENGANTUK!', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    mixer.music.play()
                else:
                    # Reset start_time_drowsy jika kondisi aman
                    start_time_drowsy = None

        # Tentukan nilai current_drowsy berdasarkan status drowsy
        current_drowsy = drowsy

        # Kirim data ke server jika drowsy terdeteksi
        if drowsy_detected:
            send_detection(
                driver_id='Driver',
                eye_state='Closed' if (ear_left < thresh or ear_right < thresh) else 'Open',
                mouth_state='Open' if mar > 0.5 else 'Closed',
                head_pose='Tilted' if abs(angle) > 15 else 'Normal',  # Menggunakan angle yang dihitung di atas
                yawning=mar > 0.5,
                drowsiness_status=current_drowsy,
                start_time=start_time_drowsy,
                end_time=time.time(),
                duration=calculate_duration(start_time_drowsy)
            )

    # Menampilkan frame
    cv2.imshow('Deteksi Kantuk Pengemudi', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup video capture dan jendela
cap.release()
cv2.destroyAllWindows()

