import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp


class FaceMeshDetector:
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_CENTER = 468

    LEFT_IRIS_RING = [474, 475, 476, 477]
    RIGHT_IRIS_RING = [469, 470, 471, 472]

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

            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(root_dir, "face_landmarker.task")
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

            if not os.path.exists(model_path):
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

    @staticmethod
    def apply_clahe(frame):
        """Meningkatkan kontras wajah secara adaptif pada kondisi ruangan redup."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def extract_features(self, frame, use_clahe=True):
        h, w, _ = frame.shape
        proc_frame = self.apply_clahe(frame) if use_clahe else frame
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
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

        mesh_2d = np.array(
            [(int(pt.x * w), int(pt.y * h)) for pt in raw_landmarks],
            dtype=np.int32
        )

        pose_2d = np.array(
            [(raw_landmarks[idx].x * w, raw_landmarks[idx].y * h) for idx in self.POSE_IDX],
            dtype=np.float64
        )

        # Bounding box wajah untuk visualisasi reticle
        x_min, y_min = np.min(mesh_2d, axis=0)
        x_max, y_max = np.max(mesh_2d, axis=0)

        return {
            "mesh_2d": mesh_2d,
            "left_eye": mesh_2d[self.LEFT_EYE_IDX],
            "right_eye": mesh_2d[self.RIGHT_EYE_IDX],
            "left_iris": mesh_2d[self.LEFT_IRIS_CENTER],
            "right_iris": mesh_2d[self.RIGHT_IRIS_CENTER],
            "left_iris_ring": mesh_2d[self.LEFT_IRIS_RING],
            "right_iris_ring": mesh_2d[self.RIGHT_IRIS_RING],
            "pose_2d": pose_2d,
            "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
            "frame_size": (h, w)
        }

    def draw_sci_fi_hud(self, frame, features, is_focused, nose_line=None):
        """Menggambar kontur mata, iris, dan bingkai sudut (reticle) modern."""
        if features is None:
            return frame

        main_color = (129, 185, 16) if is_focused else (94, 63, 244)  # BGR: Emerald / Rose
        cyan_color = (248, 189, 56)

        # 1. Gambar kontur kelopak mata & iris
        cv2.polylines(frame, [features["left_eye"]], True, cyan_color, 1, cv2.LINE_AA)
        cv2.polylines(frame, [features["right_eye"]], True, cyan_color, 1, cv2.LINE_AA)

        for ring_key, center_key in [("left_iris_ring", "left_iris"), ("right_iris_ring", "right_iris")]:
            (cx, cy), radius = cv2.minEnclosingCircle(features[ring_key])
            cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 128), 1, cv2.LINE_AA)
            cv2.circle(frame, tuple(features[center_key]), 2, (0, 215, 255), -1)

        # 2. Gambar vektor arah hadap kepala dari ujung hidung
        if nose_line is not None:
            p1, p2 = nose_line
            cv2.line(frame, p1, p2, (255, 110, 199), 2, cv2.LINE_AA)
            cv2.circle(frame, p2, 4, (255, 110, 199), -1)

        # 3. Gambar Corner Reticle (bingkai sudut wajah)
        x1, y1, x2, y2 = features["bbox"]
        pad = 18
        x1, y1 = max(10, x1 - pad), max(10, y1 - pad)
        x2, y2 = min(frame.shape[1] - 10, x2 + pad), min(frame.shape[0] - 10, y2 + pad)
        corner_len = 22

        # Sudut Kiri Atas & Kanan Atas
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), main_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), main_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), main_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), main_color, 2, cv2.LINE_AA)

        # Sudut Kiri Bawah & Kanan Bawah
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), main_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), main_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), main_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), main_color, 2, cv2.LINE_AA)

        return frame