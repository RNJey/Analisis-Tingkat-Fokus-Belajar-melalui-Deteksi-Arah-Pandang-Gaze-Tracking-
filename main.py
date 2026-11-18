import cv2
from modules.detector import FaceMeshDetector
from modules.gaze_analyzer import GazeAnalyzer


def main():
    detector = FaceMeshDetector(max_faces=1)
    analyzer = GazeAnalyzer()

    cap = cv2.VideoCapture(0)
    print("Tekan 'q' pada jendela kamera untuk keluar.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Cermin horizontal agar pergerakan kiri/kanan sesuai dengan layar
        frame = cv2.flip(frame, 1)

        # 1. Ekstraksi fitur dari modul Anggota 1
        features = detector.extract_features(frame)

        # 2. Analisis fokus dari modul Anggota 2
        metrics = analyzer.analyze(features)

        # 3. Gambar visualisasi kontur mata, iris, dan vektor arah hidung
        frame = detector.draw_eye_landmarks(frame, features)
        if metrics["nose_line"] is not None:
            p1, p2 = metrics["nose_line"]
            cv2.line(frame, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)

        # 4. Tampilkan panel informasi HUD (Heads-Up Display)
        status_color = (0, 220, 0) if metrics["is_focused"] else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (460, 155), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (460, 155), status_color, 2)

        cv2.putText(frame, metrics["status_text"], (22, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Skor Fokus Sesi : {metrics['focus_score']:.1f}%", (22, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR: {metrics['ear']:.2f} | Rasio Iris: {metrics['gaze_ratio']:.2f} ({metrics['gaze_direction']})",
                    (22, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Head Pose -> Pitch: {metrics['pitch']:.1f} deg | Yaw: {metrics['yaw']:.1f} deg",
                    (22, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Sistem Analisis Fokus Belajar - Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()