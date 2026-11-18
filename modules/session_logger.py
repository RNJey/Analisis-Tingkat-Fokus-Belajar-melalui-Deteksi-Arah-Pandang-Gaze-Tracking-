import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


class SessionLogger:
    """
    Modul perekaman data time-series fokus belajar ke CSV
    dan pembuatan grafik analisis otomatis menggunakan Matplotlib.
    """

    def __init__(self, log_dir="logs", sample_interval_sec=0.5):
        self.log_dir = log_dir
        self.sample_interval = sample_interval_sec
        os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = time.time()
        self.last_record_time = 0.0
        self.records = []

    def record(self, metrics):
        """Mencatat data metrik pada interval waktu tertentu."""
        now = time.time()
        elapsed = now - self.start_time

        if (elapsed - self.last_record_time) < self.sample_interval:
            return

        self.last_record_time = elapsed
        self.records.append({
            "detik_ke": round(elapsed, 2),
            "is_focused": int(metrics["is_focused"]),
            "status": metrics["status_text"],
            "focus_score": round(metrics["focus_score"], 2),
            "ear": round(metrics["ear"], 3),
            "gaze_ratio": round(metrics["gaze_ratio"], 3),
            "pitch": round(metrics["pitch"], 2),
            "yaw": round(metrics["yaw"], 2)
        })

    def export_report(self):
        """Menyimpan data ke CSV dan menampilkan grafik ringkasan sesi."""
        if not self.records:
            print("Tidak ada data sesi yang tercatat.")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, f"log_fokus_{timestamp_str}.csv")
        img_path = os.path.join(self.log_dir, f"grafik_fokus_{timestamp_str}.png")

        df = pd.DataFrame(self.records)
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Data CSV berhasil disimpan di: {csv_path}")

        # Membuat visualisasi 3 panel grafik untuk laporan proyek
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Laporan Analisis Tingkat Fokus Belajar (Gaze & Head Pose Tracking)", fontsize=13, fontweight="bold")

        # Panel 1: Skor Fokus Kumulatif
        axes[0].plot(df["detik_ke"], df["focus_score"], color="#16a34a", linewidth=2, label="Skor Fokus Kumulatif (%)")
        axes[0].fill_between(df["detik_ke"], df["focus_score"], color="#bbf7d0", alpha=0.4)
        axes[0].set_ylabel("Fokus (%)")
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="lower right")

        # Panel 2: Sudut Orientasi Kepala (Pitch & Yaw)
        axes[1].plot(df["detik_ke"], df["yaw"], color="#2563eb", label="Yaw (Toleh Kiri/Kanan)")
        axes[1].plot(df["detik_ke"], df["pitch"], color="#dc2626", label="Pitch (Angguk Atas/Bawah)")
        axes[1].axhline(20, color="gray", linestyle=":", alpha=0.6)
        axes[1].axhline(-20, color="gray", linestyle=":", alpha=0.6)
        axes[1].set_ylabel("Sudut (Derajat)")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="upper right")

        # Panel 3: Rasio Lirikan Iris & Eye Aspect Ratio (EAR)
        axes[2].plot(df["detik_ke"], df["gaze_ratio"], color="#9333ea", label="Rasio Iris (0.5 = Tengah)")
        axes[2].plot(df["detik_ke"], df["ear"], color="#ca8a04", label="EAR (Bukaaan Kelopak Mata)")
        axes[2].axhline(0.38, color="#9333ea", linestyle=":", alpha=0.5)
        axes[2].axhline(0.62, color="#9333ea", linestyle=":", alpha=0.5)
        axes[2].set_xlabel("Waktu Sesi Berjalan (Detik)")
        axes[2].set_ylabel("Rasio (0.0 - 1.0)")
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        print(f"[INFO] Grafik laporan berhasil disimpan di: {img_path}")
        plt.show()