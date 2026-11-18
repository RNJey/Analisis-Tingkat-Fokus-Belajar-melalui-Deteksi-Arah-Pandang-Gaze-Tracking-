import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Unduh otomatis file model face_landmarker.task jika belum ada di folder proyek
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model face_landmarker.task (~3.7 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Unduhan selesai!")

# 2. Baca model sebagai byte buffer (aman untuk path folder Windows berspasi/kurung)
with open(MODEL_PATH, "rb") as f:
    model_buffer = f.read()

base_options = python.BaseOptions(model_asset_buffer=model_buffer)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. Jalankan webcam dengan FaceLandmarker
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    start_time = time.monotonic()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp wajib selalu bertambah (monotonic) dalam satuan milidetik untuk mode VIDEO
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Indeks 468 (pusat iris kiri) dan 473 (pusat iris kanan)
                for idx in [468, 473]:
                    pt = face_landmarks[idx]
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        cv2.imshow("Test Iris & Face Landmarker (Tasks API)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()