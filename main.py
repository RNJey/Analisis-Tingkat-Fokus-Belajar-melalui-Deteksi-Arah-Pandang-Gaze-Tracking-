import cv2
from modules.detector import FaceMeshDetector
from modules.gaze_analyzer import GazeAnalyzer
from modules.session_logger import SessionLogger


def main():
    detector = FaceMeshDetector(max_faces=1)
    analyzer = GazeAnalyzer()
    logger = SessionLogger(log_dir="logs", sample_interval_sec=0.5)

    cap = cv2.VideoCapture(0)
    print("Instruksi Kontrol:")
    print("- Tekan 'c' saat menatap lurus ke tengah layar untuk KALIBRASI posisi.")
    print("- Tekan 'q' untuk mengakhiri sesi dan menampilkan grafik laporan.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        features = detector.extract_features(frame)
        metrics = analyzer.analyze(features)

        # Catat data ke dalam logger
        logger.record(metrics)

        # Visualisasi mata dan garis arah hidung
        frame = detector.draw_eye_landmarks(frame, features)
        if metrics["nose_line"] is not None:
            p1, p2 = metrics["nose_line"]
            cv2.line(frame, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)

        # Panel HUD Informasi
        status_color = (0, 220, 0) if metrics["is_focused"] else (0, 0, 255)
        cv2.rectangle(frame, (10, 10), (485, 180), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (485, 180), status_color, 2)

        cv2.putText(frame, metrics["status_text"], (22, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Skor Fokus Sesi : {metrics['focus_score']:.1f}%", (22, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"EAR: {metrics['ear']:.2f} | Rasio Iris: {metrics['gaze_ratio']:.2f} ({metrics['gaze_direction']})",
                    (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Head Pose -> Pitch: {metrics['pitch']:.1f} deg | Yaw: {metrics['yaw']:.1f} deg",
                    (22, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        calib_text = "Status Kalibrasi: AKTIF (Tekan 'c' untuk reset)" if analyzer.is_calibrated else "Belum Kalibrasi (Tatap tengah & tekan 'c')"
        calib_color = (100, 255, 100) if analyzer.is_calibrated else (0, 215, 255)
        cv2.putText(frame, calib_text, (22, 156),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, calib_color, 1, cv2.LINE_AA)

        cv2.imshow("Sistem Analisis Fokus Belajar - Gaze Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if analyzer.calibrate(features):
                print("[INFO] Kalibrasi titik tengah berhasil diterapkan!")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Simpan CSV dan tampilkan grafik evaluasi setelah kamera ditutup
    logger.export_report()


if __name__ == "__main__":
    main()