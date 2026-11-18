import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from modules.detector import FaceMeshDetector
from modules.gaze_analyzer import GazeAnalyzer

st.set_page_config(
    page_title="VisionFocus AI - Telemetry Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJEKSI CUSTOM CSS DARK GLASSMORPHISM ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #111827, #0b0f19 70%);
        color: #f8fafc;
    }
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
    }
    .hero-title {
        font-size: 1.45rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .badge-calib {
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .telemetry-card {
        background: rgba(17, 24, 39, 0.78);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .status-banner {
        border-radius: 12px;
        padding: 14px 18px;
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .status-focused {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34d399;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    .status-distracted {
        background: rgba(244, 63, 94, 0.18);
        border: 1px solid rgba(244, 63, 94, 0.6);
        color: #fb7185;
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.25);
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 14px;
    }
    .kpi-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 2px;
    }
    .gaze-track-wrap {
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .gaze-bar-bg {
        position: relative;
        height: 18px;
        background: linear-gradient(90deg, rgba(244,63,94,0.35) 0%, rgba(244,63,94,0.35) 38%, rgba(16,185,129,0.35) 38%, rgba(16,185,129,0.35) 62%, rgba(244,63,94,0.35) 62%, rgba(244,63,94,0.35) 100%);
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .gaze-dot {
        position: absolute;
        top: -3px;
        width: 22px;
        height: 22px;
        background: #38bdf8;
        border: 3px solid #ffffff;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 12px #38bdf8;
    }
    .pose-meter-bg {
        height: 8px;
        background: rgba(51, 65, 85, 0.7);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- INISIALISASI STATE ---
if "analyzer" not in st.session_state:
    st.session_state.analyzer = GazeAnalyzer()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "calibrate_req" not in st.session_state:
    st.session_state.calibrate_req = False


def render_telemetry_html(metrics, elapsed_sec):
    """Merender kartu telemetri interaktif secara real-time."""
    is_foc = metrics["is_focused"]
    status_cls = "status-focused" if is_foc else "status-distracted"
    status_icon = "🟢" if is_foc else "🚨"
    score = metrics["focus_score"]
    gaze_pct = float(np.clip(metrics["gaze_ratio"] * 100.0, 4.0, 96.0))
    ear_pct = float(np.clip((metrics["ear"] / 0.35) * 100.0, 0.0, 100.0))

    mins, secs = divmod(int(elapsed_sec), 60)
    time_str = f"{mins:02d}:{secs:02d}"

    return f"""
    <div class="status-banner {status_cls}">
        <span>{status_icon} {metrics['status_text']}</span>
        <span style="font-size:0.82rem; opacity:0.85;">⏱️ {time_str}</span>
    </div>

    <div class="kpi-grid">
        <div class="kpi-box">
            <div class="kpi-label">Skor Fokus Sesi</div>
            <div class="kpi-value" style="color: {'#34d399' if score >= 75 else '#fbbf24' if score >= 50 else '#fb7185'};">
                {score:.1f}%
            </div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Eye Aspect Ratio (EAR)</div>
            <div class="kpi-value" style="color: #38bdf8;">{metrics['ear']:.2f}</div>
            <div class="pose-meter-bg">
                <div style="width:{ear_pct:.0f}%; height:100%; background:#38bdf8;"></div>
            </div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Sudut Toleh (Yaw)</div>
            <div class="kpi-value">{metrics['yaw']:+.1f}°</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Sudut Angguk (Pitch)</div>
            <div class="kpi-value">{metrics['pitch']:+.1f}°</div>
        </div>
    </div>

    <div class="telemetry-card">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1; margin-bottom:6px;">
            <span>👁️ <b>Pelacak Posisi Iris Horizontal</b></span>
            <span style="color:#38bdf8; font-weight:700;">{metrics['gaze_direction']} ({metrics['gaze_ratio']:.2f})</span>
        </div>
        <div class="gaze-track-wrap">
            <div class="gaze-bar-bg">
                <div class="gaze-dot" style="left: {gaze_pct:.1f}%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#64748b; margin-top:6px;">
                <span>◀ Melirik Kiri (0.0)</span>
                <span style="color:#10b981; font-weight:600;">Zona Fokus Layar (0.38 - 0.62)</span>
                <span>Melirik Kanan (1.0) ▶</span>
            </div>
        </div>
    </div>
    """


# --- SIDEBAR KONTROL ---
with st.sidebar:
    st.markdown("### ⚙️ Kontrol Kamera & CV")
    run_camera = st.toggle("🚀 Aktifkan Kamera Sesi", value=False)
    use_clahe = st.checkbox("✨ Preprocessing CLAHE (Ruang Redup)", value=True)
    split_clahe_demo = st.checkbox("🌗 Mode Split-Screen CLAHE (Demo Dosen)", value=False)
    show_hud = st.checkbox("🎯 Tampilkan Reticle & Vektor 3D", value=True)

    st.markdown("---")
    st.markdown("### 🎯 Kalibrasi Posisi Duduk")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("📍 Kalibrasi Nol", use_container_width=True, type="primary"):
            st.session_state.calibrate_req = True
    with col_c2:
        if st.button("🔄 Reset Sesi", use_container_width=True):
            st.session_state.analyzer = GazeAnalyzer()
            st.session_state.logs = []
            st.session_state.start_time = time.time()
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎚️ Parameter Sensitivitas")
    ear_t = st.slider("Ambang Kantuk (EAR)", 0.12, 0.30, 0.20, 0.01)
    yaw_t = st.slider("Batas Toleh Kiri/Kanan (Yaw °)", 10.0, 35.0, 20.0, 1.0)
    pitch_t = st.slider("Batas Menunduk/Atas (Pitch °)", 10.0, 35.0, 18.0, 1.0)

    st.session_state.analyzer.ear_threshold = ear_t
    st.session_state.analyzer.yaw_threshold = yaw_t
    st.session_state.analyzer.pitch_threshold = pitch_t

# --- BANNER HEADER ---
is_cal = st.session_state.analyzer.is_calibrated
calib_badge = (
    '<span class="badge-calib" style="background:rgba(16,185,129,0.2); color:#34d399; border:1px solid #10b981;">✓ TERKALIBRASI</span>'
    if is_cal else
    '<span class="badge-calib" style="background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid #f59e0b;">⚠ BELUM KALIBRASI</span>'
)

st.markdown(f"""
<div class="hero-banner">
    <div>
        <div class="hero-title">VisionFocus AI — Gaze & Head Pose Telemetry</div>
        <div class="hero-sub">MediaPipe 478-Point Face Mesh • 3D solvePnP Pose Estimation • CLAHE Adaptive Enhancement</div>
    </div>
    <div>{calib_badge}</div>
</div>
""", unsafe_allow_html=True)

# --- TABS NAVIGASI ---
tab_live, tab_report = st.tabs(["📡 Live Telemetry & Vision Feed", "📊 Analitik Distraksi & Ekspor CSV"])

with tab_live:
    col_cam, col_tele = st.columns([1.35, 1], gap="large")

    with col_cam:
        video_slot = st.empty()

    with col_tele:
        telemetry_slot = st.empty()
        st.markdown("<div style='font-size:0.82rem; color:#94a3b8; font-weight:600; margin-bottom:4px;'>📈 Grafik Real-Time Skor Fokus & Sudut Kepala</div>", unsafe_allow_html=True)
        chart_slot = st.empty()

with tab_report:
    if st.session_state.logs:
        df_rep = pd.DataFrame(st.session_state.logs)
        c_dist, c_table = st.columns([1, 1.4])
        with c_dist:
            st.markdown("#### Distribusi Penyebab Distraksi")
            counts = st.session_state.analyzer.distraction_counts
            df_counts = pd.DataFrame(list(counts.items()), columns=["Kategori", "Jumlah Frame"]).set_index("Kategori")
            st.bar_chart(df_counts)
        with c_table:
            st.markdown("#### Riwayat Log Sesi Belajar")
            st.dataframe(df_rep.tail(25), use_container_width=True, height=260)
            csv_data = df_rep.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Unduh Laporan Lengkap (.CSV)",
                data=csv_data,
                file_name="laporan_fokus_vision.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.info("Mulai sesi kamera terlebih dahulu untuk melihat grafik distribusi distraksi dan mengunduh CSV.")

# --- LOOP KAMERA UTAMA ---
if run_camera:
    detector = FaceMeshDetector(max_faces=1)
    cap = cv2.VideoCapture(0)
    last_log_t = 0.0

    while cap.isOpened() and run_camera:
        ret, raw_frame = cap.read()
        if not ret:
            video_slot.error("Gagal mengakses webcam. Pastikan kamera tidak sedang dipakai aplikasi lain.")
            break

        raw_frame = cv2.flip(raw_frame, 1)
        features = detector.extract_features(raw_frame, use_clahe=use_clahe)

        if st.session_state.calibrate_req:
            if st.session_state.analyzer.calibrate(features):
                st.toast("Titik nol berhasil dikalibrasi!", icon="✅")
            st.session_state.calibrate_req = False

        metrics = st.session_state.analyzer.analyze(features)

        # Siapkan frame visualisasi (mendukung Split-Screen CLAHE untuk demo dosen)
        if split_clahe_demo:
            clahe_frame = detector.apply_clahe(raw_frame)
            mid_x = raw_frame.shape[1] // 2
            display_frame = raw_frame.copy()
            display_frame[:, mid_x:] = clahe_frame[:, mid_x:]
            cv2.line(display_frame, (mid_x, 0), (mid_x, raw_frame.shape[0]), (56, 189, 248), 2)
            cv2.putText(display_frame, "RAW (Asli)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(display_frame, "CLAHE Enhanced", (mid_x + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)
        else:
            display_frame = detector.apply_clahe(raw_frame) if use_clahe else raw_frame.copy()

        if show_hud:
            display_frame = detector.draw_sci_fi_hud(
                display_frame, features, metrics["is_focused"], metrics["nose_line"]
            )

        video_slot.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        now = time.time()
        elapsed = now - st.session_state.start_time
        telemetry_slot.markdown(render_telemetry_html(metrics, elapsed), unsafe_allow_html=True)

        if (now - last_log_t) >= 0.5:
            last_log_t = now
            st.session_state.logs.append({
                "Detik": round(elapsed, 1),
                "Skor Fokus (%)": round(metrics["focus_score"], 1),
                "Yaw (°)": round(metrics["yaw"], 1),
                "Pitch (°)": round(metrics["pitch"], 1),
                "Rasio Iris": round(metrics["gaze_ratio"], 2),
                "EAR": round(metrics["ear"], 2),
                "Status": metrics["status_text"]
            })
            df_live = pd.DataFrame(st.session_state.logs[-50:])
            chart_slot.line_chart(
                df_live.set_index("Detik")[["Skor Fokus (%)", "Yaw (°)", "Pitch (°)"]],
                height=195
            )

    cap.release()
else:
    with tab_live:
        video_slot.markdown("""
        <div class="telemetry-card" style="text-align:center; padding: 65px 20px;">
            <div style="font-size:2.8rem; margin-bottom:10px;">📷</div>
            <div style="font-size:1.15rem; font-weight:700; color:#e2e8f0;">Kamera Sesi Belum Aktif</div>
            <div style="font-size:0.88rem; color:#94a3b8; margin-top:6px;">
                Nyalakan sakelar <b>🚀 Aktifkan Kamera Sesi</b> pada panel sebelah kiri untuk memulai pelacakan.
            </div>
        </div>
        """, unsafe_allow_html=True)