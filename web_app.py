import io
import time
import threading
import cv2
import pandas as pd
from flask import Flask, Response, jsonify, request, render_template_string, send_file
from modules.detector import FaceMeshDetector
from modules.gaze_analyzer import GazeAnalyzer

app = Flask(__name__)

# --- STATE GLOBAL APLIKASI ---
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = FaceMeshDetector(max_faces=1)
        self.analyzer = GazeAnalyzer()
        self.camera_active = True
        self.use_clahe = True
        self.split_demo = False
        self.show_hud = True
        self.calibrate_req = False
        self.start_time = time.time()
        self.last_log_time = 0.0
        self.fps = 0.0
        self.logs = []
        self.latest_metrics = {
            "is_focused": True,
            "status_text": "Menunggu Kamera...",
            "gaze_direction": "Tengah (Layar)",
            "ear": 0.30,
            "gaze_ratio": 0.50,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "focus_score": 100.0
        }

state = AppState()


def generate_video_frames():
    """Generator streaming video MJPEG berkecepatan tinggi."""
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        if not state.camera_active:
            time.sleep(0.1)
            continue

        success, raw_frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            state.fps = (0.9 * state.fps) + (0.1 * (1.0 / dt))

        raw_frame = cv2.flip(raw_frame, 1)

        with state.lock:
            # CLAHE hanya diproses di balik layar agar video wajah tetap jernih & natural
            features = state.detector.extract_features(raw_frame, use_clahe=state.use_clahe)

            if state.calibrate_req:
                state.analyzer.calibrate(features)
                state.calibrate_req = False

            metrics = state.analyzer.analyze(features)
            state.latest_metrics = metrics

            # Catat log setiap 0.5 detik
            elapsed = round(now - state.start_time, 1)
            if (now - state.last_log_time) >= 0.5:
                state.last_log_time = now
                state.logs.append({
                    "Detik": elapsed,
                    "Skor_Fokus": round(metrics["focus_score"], 1),
                    "Yaw": round(metrics["yaw"], 1),
                    "Pitch": round(metrics["pitch"], 1),
                    "Rasio_Iris": round(metrics["gaze_ratio"], 2),
                    "EAR": round(metrics["ear"], 2),
                    "Status": metrics["status_text"]
                })

            # Visualisasi frame: Tampilkan wajah asli yang jernih kecuali saat Split Demo CLAHE aktif
            if state.split_demo:
                clahe_frame = state.detector.apply_clahe(raw_frame)
                mid_x = raw_frame.shape[1] // 2
                display_frame = raw_frame.copy()
                display_frame[:, mid_x:] = clahe_frame[:, mid_x:]
                cv2.line(display_frame, (mid_x, 0), (mid_x, raw_frame.shape[0]), (248, 189, 56), 2)
                cv2.putText(display_frame, "ORIGINAL RGB", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "CLAHE LAB CHANNEL", (mid_x + 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 189, 56), 2, cv2.LINE_AA)
            else:
                display_frame = raw_frame.copy()

            if state.show_hud:
                display_frame = state.detector.draw_sci_fi_hud(
                    display_frame, features, metrics["is_focused"], metrics["nose_line"]
                )

        ret, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# --- TEMPLATE FRONTEND REACT/TAILWIND BENTO UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionFocus AI — Real-Time Telemetry</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #09090b;
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.08) 0px, transparent 50%);
            color: #f4f4f5;
        }
        .font-mono-num { font-family: 'JetBrains Mono', monospace; }
        .bento-card {
            background: rgba(24, 24, 27, 0.65);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 1.25rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }
        input[type=range] {
            accent-color: #38bdf8;
        }
    </style>
</head>
<body class="min-h-screen p-4 md:p-6 select-none">

    <!-- TOP FLOATING NAVBAR -->
    <header class="max-w-[1440px] mx-auto mb-5 bento-card px-5 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3.5">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-400 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 font-bold text-lg">
                👁️
            </div>
            <div>
                <div class="flex items-center gap-2.5">
                    <h1 class="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
                        VisionFocus AI
                    </h1>
                    <span id="calib-badge" class="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30">
                        Belum Kalibrasi
                    </span>
                </div>
                <p class="text-xs text-zinc-400">MediaPipe 478-Point Mesh • 3D solvePnP Pose • Adaptive CLAHE</p>
            </div>
        </div>

        <!-- ACTION TOOLBAR -->
        <div class="flex items-center gap-2.5 flex-wrap">
            <div class="px-3 py-1.5 rounded-xl bg-zinc-900/90 border border-zinc-800 text-xs flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                <span class="text-zinc-400">FPS:</span>
                <span id="fps-val" class="font-mono-num font-bold text-white">0</span>
                <span class="text-zinc-600">|</span>
                <span id="timer-val" class="font-mono-num text-sky-400 font-bold">00:00</span>
            </div>

            <button onclick="sendAction('calibrate')" class="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 active:scale-95 transition-all text-zinc-950 font-bold text-xs shadow-lg shadow-sky-500/20 flex items-center gap-1.5">
                🎯 Kalibrasi Titik Nol
            </button>

            <button onclick="sendAction('reset')" class="px-3.5 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 active:scale-95 transition-all text-zinc-200 font-semibold text-xs border border-zinc-700">
                🔄 Reset Sesi
            </button>

            <a href="/api/export_csv" class="px-3.5 py-2 rounded-xl bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-300 border border-indigo-500/30 font-semibold text-xs transition-all">
                📥 Ekspor CSV
            </a>
        </div>
    </header>

    <!-- MAIN BENTO GRID -->
    <main class="max-w-[1440px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-5">

        <!-- KOLOM KIRI (7 KOLOM): VIDEO STREAM & PARAMETER DRAWER -->
        <div class="lg:col-span-7 flex flex-col gap-5">
            <!-- VIDEO CARD -->
            <div id="video-border" class="bento-card p-3 relative overflow-hidden transition-all duration-300 border-2 border-emerald-500/40">
                <!-- Floating Status Pill di atas Video -->
                <div class="absolute top-6 left-6 z-10 flex items-center gap-2.5 px-4 py-2 rounded-full bg-zinc-950/80 backdrop-blur-md border border-white/10 shadow-xl">
                    <span id="status-dot" class="w-3 h-3 rounded-full bg-emerald-400 shadow-[0_0_12px_#34d399]"></span>
                    <span id="status-text" class="text-xs font-bold tracking-wide text-emerald-300">FOKUS OPTIMAL (Menatap Layar)</span>
                </div>

                <!-- Feed Kamera -->
                <div class="rounded-xl overflow-hidden bg-zinc-950 aspect-video flex items-center justify-center">
                    <img src="/video_feed" class="w-full h-full object-cover" alt="Live Camera Stream" />
                </div>

                <!-- Quick Toggle Chips di bawah Video -->
                <div class="mt-3 pt-2 px-2 flex flex-wrap items-center justify-between gap-2 text-xs">
                    <div class="flex items-center gap-2 flex-wrap">
                        <label class="cursor-pointer px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 flex items-center gap-2">
                            <input type="checkbox" id="chk-hud" checked onchange="updateConfig()" class="rounded accent-sky-400">
                            <span>Reticle & Vektor 3D</span>
                        </label>
                        <label class="cursor-pointer px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 flex items-center gap-2">
                            <input type="checkbox" id="chk-clahe" checked onchange="updateConfig()" class="rounded accent-sky-400">
                            <span>Auto-CLAHE (Deteksi)</span>
                        </label>
                        <label class="cursor-pointer px-3 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/30 text-sky-300 flex items-center gap-2">
                            <input type="checkbox" id="chk-split" onchange="updateConfig()" class="rounded accent-sky-400">
                            <span>🌗 Split-Screen CLAHE (Demo Dosen)</span>
                        </label>
                    </div>
                </div>
            </div>

            <!-- PARAMETER SLIDERS BENTO CARD -->
            <div class="bento-card p-5">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-xs font-bold uppercase tracking-wider text-zinc-400">🎚️ Kalibrasi Sensitivitas Ambang Batas</h2>
                    <span class="text-[11px] text-zinc-500">Perubahan diterapkan secara real-time</span>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="p-3 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
                        <div class="flex justify-between text-xs mb-1.5">
                            <span class="text-zinc-400">Ambang Kantuk (EAR)</span>
                            <span id="lbl-ear" class="font-mono-num font-bold text-sky-400">0.20</span>
                        </div>
                        <input type="range" id="rng-ear" min="0.12" max="0.30" step="0.01" value="0.20" oninput="updateConfig()" class="w-full h-1.5 bg-zinc-800 rounded-lg cursor-pointer">
                    </div>

                    <div class="p-3 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
                        <div class="flex justify-between text-xs mb-1.5">
                            <span class="text-zinc-400">Batas Toleh (Yaw)</span>
                            <span id="lbl-yaw" class="font-mono-num font-bold text-sky-400">20°</span>
                        </div>
                        <input type="range" id="rng-yaw" min="10" max="35" step="1" value="20" oninput="updateConfig()" class="w-full h-1.5 bg-zinc-800 rounded-lg cursor-pointer">
                    </div>

                    <div class="p-3 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
                        <div class="flex justify-between text-xs mb-1.5">
                            <span class="text-zinc-400">Batas Angguk (Pitch)</span>
                            <span id="lbl-pitch" class="font-mono-num font-bold text-sky-400">18°</span>
                        </div>
                        <input type="range" id="rng-pitch" min="10" max="35" step="1" value="18" oninput="updateConfig()" class="w-full h-1.5 bg-zinc-800 rounded-lg cursor-pointer">
                    </div>
                </div>
            </div>
        </div>

        <!-- KOLOM KANAN (5 KOLOM): TELEMETRY BENTO CARDS -->
        <div class="lg:col-span-5 flex flex-col gap-5">

            <!-- BARIS ATAS: FOCUS RING & 4 KPI MINI -->
            <div class="grid grid-cols-1 sm:grid-cols-12 gap-4">
                <!-- Circular Focus Score -->
                <div class="sm:col-span-5 bento-card p-4 flex flex-col items-center justify-center relative">
                    <span class="text-[11px] font-bold uppercase tracking-wider text-zinc-400 mb-2">Skor Fokus Sesi</span>
                    <div class="relative w-28 h-28 flex items-center justify-center">
                        <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="40" stroke="rgba(255,255,255,0.07)" stroke-width="9" fill="transparent"/>
                            <circle id="score-ring" cx="50" cy="50" r="40" stroke="#34d399" stroke-width="9" fill="transparent"
                                stroke-dasharray="251.2" stroke-dashoffset="0" stroke-linecap="round" class="transition-all duration-300"/>
                        </svg>
                        <div class="absolute inset-0 flex flex-col items-center justify-center">
                            <span id="val-score" class="font-mono-num text-2xl font-extrabold text-white">100%</span>
                        </div>
                    </div>
                </div>

                <!-- 2D Head Pose Compass & EAR -->
                <div class="sm:col-span-7 bento-card p-4 flex flex-col justify-between">
                    <div class="flex items-center justify-between">
                        <span class="text-[11px] font-bold uppercase tracking-wider text-zinc-400">🧭 Radar Orientasi Kepala 3D</span>
                        <span id="val-pose" class="font-mono-num text-xs text-sky-400 font-bold">Y: 0.0° | P: 0.0°</span>
                    </div>

                    <!-- Radar Crosshair Interaktif -->
                    <div class="my-2 flex items-center gap-4">
                        <div class="relative w-24 h-24 rounded-full bg-zinc-950 border border-zinc-800 flex items-center justify-center overflow-hidden shrink-0">
                            <!-- Zona Aman Fokus -->
                            <div class="w-12 h-12 rounded-full border border-dashed border-emerald-500/40 bg-emerald-500/5"></div>
                            <!-- Garis Sumbu -->
                            <div class="absolute inset-x-0 h-[1px] bg-zinc-800"></div>
                            <div class="absolute inset-y-0 w-[1px] bg-zinc-800"></div>
                            <!-- Titik Posisi Kepala Real-time -->
                            <div id="radar-dot" class="absolute w-3.5 h-3.5 rounded-full bg-sky-400 shadow-[0_0_10px_#38bdf8] border-2 border-white transition-all duration-100"
                                 style="left: calc(50% - 7px); top: calc(50% - 7px);"></div>
                        </div>

                        <div class="flex-1 space-y-2">
                            <div class="p-2 rounded-lg bg-zinc-900/80 border border-zinc-800/80">
                                <div class="flex justify-between text-[11px]">
                                    <span class="text-zinc-400">Bukaan Mata (EAR)</span>
                                    <span id="val-ear" class="font-mono-num font-bold text-emerald-400">0.30</span>
                                </div>
                                <div class="w-full h-1.5 bg-zinc-800 rounded-full mt-1 overflow-hidden">
                                    <div id="bar-ear" class="h-full bg-gradient-to-r from-sky-400 to-emerald-400 transition-all duration-150" style="width: 85%;"></div>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-1.5 text-[11px]">
                                <div class="px-2.5 py-1.5 rounded-lg bg-zinc-900/80 border border-zinc-800/80 flex justify-between">
                                    <span class="text-zinc-500">Yaw</span>
                                    <span id="val-yaw" class="font-mono-num font-bold text-zinc-200">0.0°</span>
                                </div>
                                <div class="px-2.5 py-1.5 rounded-lg bg-zinc-900/80 border border-zinc-800/80 flex justify-between">
                                    <span class="text-zinc-500">Pitch</span>
                                    <span id="val-pitch" class="font-mono-num font-bold text-zinc-200">0.0°</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PELACAK POSISI IRIS HORIZONTAL -->
            <div class="bento-card p-4">
                <div class="flex justify-between items-center text-xs mb-2">
                    <span class="font-bold uppercase tracking-wider text-zinc-400">👁️ Pelacak Vektor Iris Horizontal</span>
                    <span id="val-gaze" class="font-mono-num font-bold text-sky-400">Tengah (0.50)</span>
                </div>
                <div class="relative h-5 rounded-full bg-zinc-950 border border-zinc-800 overflow-hidden p-0.5">
                    <!-- Zona Fokus Hijau di Tengah -->
                    <div class="absolute inset-y-0 left-[38%] right-[38%] bg-emerald-500/20 border-x border-emerald-500/40"></div>
                    <!-- Indikator Bola Mata -->
                    <div id="iris-dot" class="relative top-0.5 w-3.5 h-3.5 rounded-full bg-sky-400 border-2 border-white shadow-[0_0_12px_#38bdf8] transition-all duration-150"
                         style="left: 50%; transform: translateX(-50%);"></div>
                </div>
                <div class="flex justify-between text-[11px] text-zinc-500 mt-1.5 font-medium">
                    <span>◀ Melirik Kiri (0.0)</span>
                    <span class="text-emerald-400">Zona Fokus Layar (0.38 – 0.62)</span>
                    <span>Melirik Kanan (1.0) ▶</span>
                </div>
            </div>

            <!-- GRAFIK DUAL-AXIS REAL-TIME & COUNTER DISTRAKSI -->
            <div class="bento-card p-4 flex-1 flex flex-col justify-between">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs font-bold uppercase tracking-wider text-zinc-400">📈 Telemetri Fokus & Orientasi Kepala</span>
                    <span class="text-[11px] text-zinc-500">Dual-Axis (Skor % & Sudut °)</span>
                </div>
                <div class="h-44 w-full">
                    <canvas id="telemetryChart"></canvas>
                </div>

                <!-- Statistik Distraksi Pills -->
                <div class="grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-zinc-800/80 text-center">
                    <div class="p-1.5 rounded-lg bg-zinc-900/70">
                        <div class="text-[10px] text-zinc-500">Melirik</div>
                        <div id="cnt-gaze" class="font-mono-num text-xs font-bold text-rose-400">0</div>
                    </div>
                    <div class="p-1.5 rounded-lg bg-zinc-900/70">
                        <div class="text-[10px] text-zinc-500">Menoleh</div>
                        <div id="cnt-yaw" class="font-mono-num text-xs font-bold text-amber-400">0</div>
                    </div>
                    <div class="p-1.5 rounded-lg bg-zinc-900/70">
                        <div class="text-[10px] text-zinc-500">Menunduk</div>
                        <div id="cnt-pitch" class="font-mono-num text-xs font-bold text-purple-400">0</div>
                    </div>
                    <div class="p-1.5 rounded-lg bg-zinc-900/70">
                        <div class="text-[10px] text-zinc-500">Mengantuk</div>
                        <div id="cnt-drowsy" class="font-mono-num text-xs font-bold text-sky-400">0</div>
                    </div>
                </div>
            </div>

        </div>
    </main>

    <script>
        // Inisialisasi Grafik Dual-Axis (Agar Pitch/Yaw tidak gepeng tertimpa Skor 100%)
        const ctx = document.getElementById('telemetryChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Skor Fokus (%)',
                        data: [],
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(52, 211, 153, 0.08)',
                        yAxisID: 'yScore',
                        tension: 0.35,
                        fill: true,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Yaw (°)',
                        data: [],
                        borderColor: '#38bdf8',
                        yAxisID: 'yAngle',
                        tension: 0.35,
                        pointRadius: 0,
                        borderWidth: 1.5
                    },
                    {
                        label: 'Pitch (°)',
                        data: [],
                        borderColor: '#c084fc',
                        yAxisID: 'yAngle',
                        tension: 0.35,
                        pointRadius: 0,
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#a1a1aa', font: { size: 10 } } }
                },
                scales: {
                    x: { ticks: { color: '#71717a', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                    yScore: { type: 'linear', position: 'left', min: 0, max: 105, ticks: { color: '#34d399', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    yAngle: { type: 'linear', position: 'right', min: -35, max: 35, ticks: { color: '#38bdf8', font: { size: 10 } }, grid: { drawOnChartArea: false } }
                }
            }
        });

        async function sendAction(action) {
            await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            });
        }

        async function updateConfig() {
            const ear = parseFloat(document.getElementById('rng-ear').value);
            const yaw = parseFloat(document.getElementById('rng-yaw').value);
            const pitch = parseFloat(document.getElementById('rng-pitch').value);
            document.getElementById('lbl-ear').innerText = ear.toFixed(2);
            document.getElementById('lbl-yaw').innerText = yaw + '°';
            document.getElementById('lbl-pitch').innerText = pitch + '°';

            await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ear_threshold: ear,
                    yaw_threshold: yaw,
                    pitch_threshold: pitch,
                    show_hud: document.getElementById('chk-hud').checked,
                    use_clahe: document.getElementById('chk-clahe').checked,
                    split_demo: document.getElementById('chk-split').checked
                })
            });
        }

        // Polling telemetri cepat setiap 150ms
        setInterval(async () => {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                const m = data.metrics;

                // Update Header & Timer
                document.getElementById('fps-val').innerText = data.fps.toFixed(0);
                const mins = String(Math.floor(data.elapsed / 60)).padStart(2, '0');
                const secs = String(Math.floor(data.elapsed % 60)).padStart(2, '0');
                document.getElementById('timer-val').innerText = `${mins}:${secs}`;

                const badge = document.getElementById('calib-badge');
                if (data.is_calibrated) {
                    badge.innerText = '✓ Terkalibrasi';
                    badge.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
                } else {
                    badge.innerText = '⚠ Belum Kalibrasi';
                    badge.className = 'text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30';
                }

                // Update Status Banner & Video Border
                const vBorder = document.getElementById('video-border');
                const sDot = document.getElementById('status-dot');
                const sText = document.getElementById('status-text');
                sText.innerText = m.status_text;

                if (m.is_focused) {
                    vBorder.className = 'bento-card p-3 relative overflow-hidden transition-all duration-300 border-2 border-emerald-500/40';
                    sDot.className = 'w-3 h-3 rounded-full bg-emerald-400 shadow-[0_0_12px_#34d399]';
                    sText.className = 'text-xs font-bold tracking-wide text-emerald-300';
                } else {
                    vBorder.className = 'bento-card p-3 relative overflow-hidden transition-all duration-300 border-2 border-rose-500/70 shadow-[0_0_30px_rgba(244,63,94,0.2)]';
                    sDot.className = 'w-3 h-3 rounded-full bg-rose-500 shadow-[0_0_12px_#f43f5e] animate-ping';
                    sText.className = 'text-xs font-bold tracking-wide text-rose-400';
                }

                // Update Circular Focus Ring
                const score = m.focus_score;
                document.getElementById('val-score').innerText = score.toFixed(1) + '%';
                const ring = document.getElementById('score-ring');
                const offset = 251.2 - (251.2 * score) / 100.0;
                ring.style.strokeDashoffset = offset;
                ring.style.stroke = score >= 75 ? '#34d399' : (score >= 50 ? '#fbbf24' : '#fb7185');

                // Update Radar 2D & EAR
                document.getElementById('val-ear').innerText = m.ear.toFixed(2);
                document.getElementById('bar-ear').style.width = Math.min(100, (m.ear / 0.35) * 100) + '%';
                document.getElementById('val-yaw').innerText = (m.yaw >= 0 ? '+' : '') + m.yaw.toFixed(1) + '°';
                document.getElementById('val-pitch').innerText = (m.pitch >= 0 ? '+' : '') + m.pitch.toFixed(1) + '°';
                document.getElementById('val-pose').innerText = `Y: ${m.yaw.toFixed(0)}° | P: ${m.pitch.toFixed(0)}°`;

                const radarX = Math.max(-40, Math.min(40, (m.yaw / 35.0) * 40));
                const radarY = Math.max(-40, Math.min(40, (-m.pitch / 35.0) * 40));
                const radarDot = document.getElementById('radar-dot');
                radarDot.style.transform = `translate(${radarX}px, ${radarY}px)`;

                // Update Iris Horizontal Glider
                const gazePct = Math.max(4, Math.min(96, m.gaze_ratio * 100));
                document.getElementById('iris-dot').style.left = gazePct + '%';
                document.getElementById('val-gaze').innerText = `${m.gaze_direction} (${m.gaze_ratio.toFixed(2)})`;

                // Update Counter Distraksi
                document.getElementById('cnt-gaze').innerText = data.counts['Melirik'] || 0;
                document.getElementById('cnt-yaw').innerText = data.counts['Menoleh'] || 0;
                document.getElementById('cnt-pitch').innerText = data.counts['Menunduk/Atas'] || 0;
                document.getElementById('cnt-drowsy').innerText = data.counts['Mengantuk'] || 0;

                // Update Chart.js
                if (data.history.length > 0) {
                    chart.data.labels = data.history.map(d => d.Detik + 's');
                    chart.data.datasets[0].data = data.history.map(d => d.Skor_Fokus);
                    chart.data.datasets[1].data = data.history.map(d => d.Yaw);
                    chart.data.datasets[2].data = data.history.map(d => d.Pitch);
                    chart.update();
                }
            } catch (e) {}
        }, 150);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    return Response(generate_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def api_telemetry():
    with state.lock:
        elapsed = time.time() - state.start_time
        return jsonify({
            "metrics": {
                "is_focused": bool(state.latest_metrics["is_focused"]),
                "status_text": state.latest_metrics["status_text"],
                "gaze_direction": state.latest_metrics["gaze_direction"],
                "ear": float(state.latest_metrics["ear"]),
                "gaze_ratio": float(state.latest_metrics["gaze_ratio"]),
                "pitch": float(state.latest_metrics["pitch"]),
                "yaw": float(state.latest_metrics["yaw"]),
                "focus_score": float(state.latest_metrics["focus_score"])
            },
            "fps": round(state.fps, 1),
            "elapsed": round(elapsed, 1),
            "is_calibrated": state.analyzer.is_calibrated,
            "counts": state.analyzer.distraction_counts,
            "history": state.logs[-45:]
        })


@app.route("/api/action", methods=["POST"])
def api_action():
    data = request.get_json() or {}
    act = data.get("action")
    with state.lock:
        if act == "calibrate":
            state.calibrate_req = True
        elif act == "reset":
            state.analyzer = GazeAnalyzer()
            state.logs = []
            state.start_time = time.time()
    return jsonify({"ok": True})


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json() or {}
    with state.lock:
        state.analyzer.ear_threshold = float(data.get("ear_threshold", 0.20))
        state.analyzer.yaw_threshold = float(data.get("yaw_threshold", 20.0))
        state.analyzer.pitch_threshold = float(data.get("pitch_threshold", 18.0))
        state.show_hud = bool(data.get("show_hud", True))
        state.use_clahe = bool(data.get("use_clahe", True))
        state.split_demo = bool(data.get("split_demo", False))
    return jsonify({"ok": True})


@app.route("/api/export_csv")
def api_export_csv():
    with state.lock:
        df = pd.DataFrame(state.logs) if state.logs else pd.DataFrame(columns=["Detik", "Skor_Fokus", "Yaw", "Pitch", "Rasio_Iris", "EAR", "Status"])
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="laporan_fokus_vision.csv")


if __name__ == "__main__":
    print("Dashboard aktif! Buka di browser: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)