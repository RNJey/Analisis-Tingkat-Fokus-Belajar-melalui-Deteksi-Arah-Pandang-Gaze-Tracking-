import cv2
import numpy as np


class GazeAnalyzer:
    MODEL_3D_POINTS = np.array([
        (0.0, 0.0, 0.0),            # 1: Ujung Hidung
        (0.0, 330.0, 65.0),         # 152: Dagu
        (-225.0, -170.0, 135.0),    # 33: Sudut Mata Kiri
        (225.0, -170.0, 135.0),     # 263: Sudut Mata Kanan
        (-150.0, 150.0, 125.0),     # 61: Sudut Mulut Kiri
        (150.0, 150.0, 125.0)       # 291: Sudut Mulut Kanan
    ], dtype=np.float64)

    def __init__(
        self,
        ear_threshold=0.20,
        gaze_left_thresh=0.38,
        gaze_right_thresh=0.62,
        yaw_threshold=20.0,
        pitch_threshold=18.0,
        drowsy_frames_limit=15,
        distraction_grace_frames=8
    ):
        self.ear_threshold = ear_threshold
        self.gaze_left_thresh = gaze_left_thresh
        self.gaze_right_thresh = gaze_right_thresh
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.drowsy_frames_limit = drowsy_frames_limit
        self.distraction_grace_frames = distraction_grace_frames

        # Offset kalibrasi
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.gaze_offset = 0.0
        self.is_calibrated = False

        # Filter Exponential Moving Average (EMA)
        self.ema_alpha = 0.35
        self.smooth_gaze = 0.50
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0

        # Counter frame & analitik distraksi
        self.closed_eye_frames = 0
        self.distracted_streak = 0
        self.total_frames = 0
        self.focused_frames = 0
        self.distraction_counts = {
            "Melirik": 0,
            "Menoleh": 0,
            "Menunduk/Atas": 0,
            "Mengantuk": 0,
            "Tanpa Wajah": 0
        }

    @staticmethod
    def _euclidean_dist(pt1, pt2):
        return np.linalg.norm(pt1.astype(np.float64) - pt2.astype(np.float64))

    def calculate_single_ear(self, eye_pts):
        vert_1 = self._euclidean_dist(eye_pts[1], eye_pts[5])
        vert_2 = self._euclidean_dist(eye_pts[2], eye_pts[4])
        horiz = self._euclidean_dist(eye_pts[0], eye_pts[3])
        if horiz == 0:
            return 0.0
        return float((vert_1 + vert_2) / (2.0 * horiz))

    def calculate_gaze_ratio(self, eye_pts, iris_center):
        p_ext_1, p_ext_2 = eye_pts[0], eye_pts[3]
        left_corner = p_ext_1 if p_ext_1[0] < p_ext_2[0] else p_ext_2
        right_corner = p_ext_2 if p_ext_1[0] < p_ext_2[0] else p_ext_1

        eye_vec = right_corner.astype(np.float64) - left_corner.astype(np.float64)
        iris_vec = iris_center.astype(np.float64) - left_corner.astype(np.float64)

        eye_len_sq = np.dot(eye_vec, eye_vec)
        if eye_len_sq == 0:
            return 0.5

        ratio = np.dot(iris_vec, eye_vec) / eye_len_sq
        return float(np.clip(ratio, 0.0, 1.0))

    def estimate_head_pose(self, pose_2d, frame_size):
        h, w = frame_size
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)

        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_3D_POINTS,
            pose_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, None

        rot_mat, _ = cv2.Rodrigues(rvec)
        pitch = float(np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[2, 2])))
        yaw = float(np.degrees(np.arctan2(rot_mat[0, 2], np.sqrt(rot_mat[1, 2] ** 2 + rot_mat[2, 2] ** 2))))
        roll = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))

        nose_end_3d = np.array([(0.0, 0.0, -450.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs)
        p1 = (int(pose_2d[0][0]), int(pose_2d[0][1]))
        p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

        return pitch, yaw, roll, (p1, p2)

    def calibrate(self, features):
        if features is None:
            return False

        gaze_left = self.calculate_gaze_ratio(features["left_eye"], features["left_iris"])
        gaze_right = self.calculate_gaze_ratio(features["right_eye"], features["right_iris"])
        raw_gaze = (gaze_left + gaze_right) / 2.0

        raw_pitch, raw_yaw, _, _ = self.estimate_head_pose(
            features["pose_2d"], features["frame_size"]
        )

        self.pitch_offset = raw_pitch
        self.yaw_offset = raw_yaw
        self.gaze_offset = raw_gaze - 0.50
        self.smooth_gaze = 0.50
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.is_calibrated = True
        return True

    def analyze(self, features):
        self.total_frames += 1

        if features is None:
            self.distraction_counts["Tanpa Wajah"] += 1
            return {
                "is_focused": False,
                "status_text": "Wajah Tidak Terdeteksi",
                "gaze_direction": "N/A",
                "ear": 0.0,
                "gaze_ratio": 0.5,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "focus_score": self.get_focus_percentage(),
                "nose_line": None
            }

        # 1. Hitung EAR
        ear_left = self.calculate_single_ear(features["left_eye"])
        ear_right = self.calculate_single_ear(features["right_eye"])
        avg_ear = (ear_left + ear_right) / 2.0

        # 2. Hitung Gaze Ratio mentah + offset kalibrasi
        gaze_left = self.calculate_gaze_ratio(features["left_eye"], features["left_iris"])
        gaze_right = self.calculate_gaze_ratio(features["right_eye"], features["right_iris"])
        raw_gaze = (gaze_left + gaze_right) / 2.0
        calibrated_gaze = float(np.clip(raw_gaze - self.gaze_offset, 0.0, 1.0))

        # 3. Hitung Head Pose + offset kalibrasi
        raw_pitch, raw_yaw, roll, nose_line = self.estimate_head_pose(
            features["pose_2d"], features["frame_size"]
        )
        calibrated_pitch = raw_pitch - self.pitch_offset
        calibrated_yaw = raw_yaw - self.yaw_offset

        # 4. Haluskan sinyal dengan EMA (setelah nilai di atas selesai dihitung)
        self.smooth_gaze = (self.ema_alpha * calibrated_gaze) + ((1.0 - self.ema_alpha) * self.smooth_gaze)
        self.smooth_pitch = (self.ema_alpha * calibrated_pitch) + ((1.0 - self.ema_alpha) * self.smooth_pitch)
        self.smooth_yaw = (self.ema_alpha * calibrated_yaw) + ((1.0 - self.ema_alpha) * self.smooth_yaw)

        avg_gaze = self.smooth_gaze
        pitch = self.smooth_pitch
        yaw = self.smooth_yaw

        # 5. Klasifikasi arah lirikan
        if avg_gaze < self.gaze_left_thresh:
            gaze_dir = "Melirik Kiri"
        elif avg_gaze > self.gaze_right_thresh:
            gaze_dir = "Melirik Kanan"
        else:
            gaze_dir = "Tengah (Layar)"

        # 6. Deteksi kedipan / kantuk
        if avg_ear < self.ear_threshold:
            self.closed_eye_frames += 1
        else:
            self.closed_eye_frames = 0

        # 7. Evaluasi status fokus dengan Grace Period
        instant_focused = True
        reason_text = "FOKUS OPTIMAL (Menatap Layar)"
        distraction_key = None

        if self.closed_eye_frames >= self.drowsy_frames_limit:
            instant_focused = False
            reason_text = "Mengantuk / Mata Tertutup"
            distraction_key = "Mengantuk"
        elif abs(yaw) > self.yaw_threshold:
            instant_focused = False
            arah_toleh = "Kanan" if yaw > 0 else "Kiri"
            reason_text = f"Menoleh ke {arah_toleh}"
            distraction_key = "Menoleh"
        elif abs(pitch) > self.pitch_threshold:
            instant_focused = False
            arah_angguk = "Bawah (Menunduk)" if pitch < 0 else "Atas"
            reason_text = f"Kepala Menghadap {arah_angguk}"
            distraction_key = "Menunduk/Atas"
        elif gaze_dir != "Tengah (Layar)":
            instant_focused = False
            reason_text = f"Mata {gaze_dir}"
            distraction_key = "Melirik"

        if not instant_focused:
            self.distracted_streak += 1
        else:
            self.distracted_streak = 0

        # Hanya nyatakan TIDAK FOKUS jika melewati batas toleransi frame (atau sedang mengantuk)
        if not instant_focused and (
            self.distracted_streak >= self.distraction_grace_frames
            or distraction_key == "Mengantuk"
        ):
            is_focused = False
            status_text = f"DISTRKASI: {reason_text}"
            if distraction_key:
                self.distraction_counts[distraction_key] += 1
        else:
            is_focused = True
            status_text = "FOKUS OPTIMAL (Menatap Layar)"
            self.focused_frames += 1

        return {
            "is_focused": is_focused,
            "status_text": status_text,
            "gaze_direction": gaze_dir,
            "ear": avg_ear,
            "gaze_ratio": avg_gaze,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "focus_score": self.get_focus_percentage(),
            "nose_line": nose_line
        }

    def get_focus_percentage(self):
        if self.total_frames == 0:
            return 100.0
        return float((self.focused_frames / self.total_frames) * 100.0)