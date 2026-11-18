import cv2
import numpy as np


class GazeAnalyzer:

    MODEL_3D_POINTS = np.array([
        (0.0, 0.0, 0.0),            # 1: Ujung Hidung
        (0.0, 330.0, 65.0),         # 152: Dagu
        (-225.0, -170.0, 135.0),    # 33: Sudut Mata Kiri di layar cermin
        (225.0, -170.0, 135.0),     # 263: Sudut Mata Kanan di layar cermin
        (-150.0, 150.0, 125.0),     # 61: Sudut Mulut Kiri di layar cermin
        (150.0, 150.0, 125.0)       # 291: Sudut Mulut Kanan di layar cermin
    ], dtype=np.float64)

    def __init__(
        self,
        ear_threshold=0.20,
        gaze_left_thresh=0.38,
        gaze_right_thresh=0.62,
        yaw_threshold=20.0,
        pitch_threshold=18.0,
        drowsy_frames_limit=15
    ):
        self.ear_threshold = ear_threshold
        self.gaze_left_thresh = gaze_left_thresh
        self.gaze_right_thresh = gaze_right_thresh
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.drowsy_frames_limit = drowsy_frames_limit

        # Offset kalibrasi untuk menyesuaikan posisi duduk & sudut kamera
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.gaze_offset = 0.0
        self.is_calibrated = False

        self.closed_eye_frames = 0
        self.total_frames = 0
        self.focused_frames = 0

    @staticmethod
    def _euclidean_dist(pt1, pt2):
        return np.linalg.norm(pt1.astype(np.float64) - pt2.astype(np.float64))

    def calculate_single_ear(self, eye_pts):
        vert_1 = self._euclidean_dist(eye_pts[1], eye_pts[5])
        vert_2 = self._euclidean_dist(eye_pts[2], eye_pts[4])
        horiz = self._euclidean_dist(eye_pts[0], eye_pts[3])

        if horiz == 0:
            return 0.0
        return (vert_1 + vert_2) / (2.0 * horiz)

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

        pitch = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[2, 2]))
        yaw = np.degrees(np.arctan2(rot_mat[0, 2], np.sqrt(rot_mat[1, 2] ** 2 + rot_mat[2, 2] ** 2)))
        roll = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))

        nose_end_3d = np.array([(0.0, 0.0, -450.0)], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs)
        p1 = (int(pose_2d[0][0]), int(pose_2d[0][1]))
        p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

        return pitch, yaw, roll, (p1, p2)

    def calibrate(self, features):
        """
        Menyimpan sudut kepala dan posisi iris saat ini sebagai titik tengah (0 derajat dan rasio 0.50).
        Panggil fungsi ini saat pengguna sedang menatap lurus ke tengah layar.
        """
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
        self.is_calibrated = True
        return True

    def analyze(self, features):
        self.total_frames += 1

        if features is None:
            return {
                "is_focused": False,
                "status_text": "TIDAK FOKUS: Wajah Tidak Terdeteksi",
                "gaze_direction": "N/A",
                "ear": 0.0,
                "gaze_ratio": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "focus_score": self.get_focus_percentage(),
                "nose_line": None
            }

        ear_left = self.calculate_single_ear(features["left_eye"])
        ear_right = self.calculate_single_ear(features["right_eye"])
        avg_ear = (ear_left + ear_right) / 2.0

        gaze_left = self.calculate_gaze_ratio(features["left_eye"], features["left_iris"])
        gaze_right = self.calculate_gaze_ratio(features["right_eye"], features["right_iris"])
        raw_gaze = (gaze_left + gaze_right) / 2.0
        avg_gaze = float(np.clip(raw_gaze - self.gaze_offset, 0.0, 1.0))

        raw_pitch, raw_yaw, roll, nose_line = self.estimate_head_pose(
            features["pose_2d"], features["frame_size"]
        )
        pitch = raw_pitch - self.pitch_offset
        yaw = raw_yaw - self.yaw_offset

        if avg_gaze < self.gaze_left_thresh:
            gaze_dir = "Melirik Kiri"
        elif avg_gaze > self.gaze_right_thresh:
            gaze_dir = "Melirik Kanan"
        else:
            gaze_dir = "Tengah (Layar)"

        if avg_ear < self.ear_threshold:
            self.closed_eye_frames += 1
        else:
            self.closed_eye_frames = 0

        is_focused = True
        status_text = "FOKUS (Menatap Layar)"

        if self.closed_eye_frames >= self.drowsy_frames_limit:
            is_focused = False
            status_text = "TIDAK FOKUS: Mengantuk / Mata Tertutup"
        elif abs(yaw) > self.yaw_threshold:
            is_focused = False
            arah_toleh = "Kanan" if yaw > 0 else "Kiri"
            status_text = f"TIDAK FOKUS: Menoleh ke {arah_toleh}"
        elif abs(pitch) > self.pitch_threshold:
            is_focused = False
            arah_angguk = "Bawah (Menunduk)" if pitch < 0 else "Atas"
            status_text = f"TIDAK FOKUS: Kepala Menghadap {arah_angguk}"
        elif gaze_dir != "Tengah (Layar)":
            is_focused = False
            status_text = f"TIDAK FOKUS: Mata {gaze_dir}"

        if is_focused:
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
        return (self.focused_frames / self.total_frames) * 100.0