import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp


class FaceMeshDetector:
    """
    Modul Anggota 1: Ekstraksi 478 titik koordinat wajah dan iris mata.
    Mendukung mp.solutions.face_mesh maupun mp.tasks.python.vision.FaceLandmarker.
    """

    # Indeks 6 titik mata untuk perhitungan Eye Aspect Ratio (EAR)
    # Urutan: [ujung_luar, atas_1, atas_2, ujung_dalam, bawah_2, bawah_1]
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

    # Indeks titik pusat iris mata
    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_CENTER = 468

    # Indeks keliling iris mata
    LEFT_IRIS_RING = [474, 475, 476, 477]
    RIGHT_IRIS_RING = [469, 470, 471, 472]

    # Indeks 6 titik kunci wajah untuk estimasi posisi kepala 3D (solvePnP)
    # Urutan: [Hidung, Dagu, Sudut Mata Kanan, Sudut Mata Kiri, Mulut Kanan, Mulut Kiri]
    POSE_IDX = [1, 152, 33, 263, 61, 291]

    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.use_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        self.start_time = time.monotonic()

        if self.use_legacy:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detector = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf
            )
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Simpan model di folder utama proyek
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root_dir, "face_landmarker.task")
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

            if not os.path.exists(model_path):
                print("Mengunduh model face_landmarker.task...")
                urllib.request.urlretrieve(model_url, model_path)

            with open(model_path, "rb") as f:
                model_buffer = f.read()

            base_options = python.BaseOptions(model_asset_buffer=model_buffer)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=min_detection_conf,
                min_face_presence_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_features(self, frame):
        """
        Memproses frame BGR dari OpenCV dan mengembalikan dictionary berisi
        koordinat piksel (2D) dari mata, iris, dan titik referensi kepala.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_landmarks = None

        if self.use_legacy:
            results = self.detector.process(rgb_frame)
            if results.multi_face_landmarks:
                raw_landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
            if results.face_landmarks:
                raw_landmarks = results.face_landmarks[0]

        if raw_landmarks is None:
            return None

        # Konversi seluruh 478 titik ke koordinat piksel (x, y)
        mesh_2d = np.array(
            [(int(pt.x * w), int(pt.y * h)) for pt in raw_landmarks],
            dtype=np.int32
        )

        # Koordinat presisi float khusus untuk perhitungan solvePnP
        pose_2d = np.array(
            [( raw_landmarks[idx].x * w, raw_landmarks[idx].y * h ) for idx in self.POSE_IDX],
            dtype=np.float64
        )

        return {
            "mesh_2d": mesh_2d,
            "left_eye": mesh_2d[self.LEFT_EYE_IDX],
            "right_eye": mesh_2d[self.RIGHT_EYE_IDX],
            "left_iris": mesh_2d[self.LEFT_IRIS_CENTER],
            "right_iris": mesh_2d[self.RIGHT_IRIS_CENTER],
            "left_iris_ring": mesh_2d[self.LEFT_IRIS_RING],
            "right_iris_ring": mesh_2d[self.RIGHT_IRIS_RING],
            "pose_2d": pose_2d,
            "frame_size": (h, w)
        }

    def draw_eye_landmarks(self, frame, features):
        """Menggambar kontur kelopak mata dan lingkaran iris pada frame."""
        if features is None:
            return frame

        # Gambar poligon kontur mata kiri dan kanan
        cv2.polylines(frame, [features["left_eye"]], True, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [features["right_eye"]], True, (255, 255, 0), 1, cv2.LINE_AA)

        # Gambar lingkaran iris dan titik pusat pupil
        for ring_key, center_key in [("left_iris_ring", "left_iris"), ("right_iris_ring", "right_iris")]:
            (cx, cy), radius = cv2.minEnclosingCircle(features[ring_key])
            cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, tuple(features[center_key]), 2, (0, 0, 255), -1)

        return frame