# 🎓 AI Proctoring System

A real-time, vision-based exam proctoring backend built with Python, FastAPI, MediaPipe, and YOLOv8. This is my first computer vision project — built to learn and explore the domain of real-time facial analysis, gaze estimation, and behavioral risk detection.

---

## 📌 What It Does

The system processes a live webcam feed through a WebSocket connection, analyzes the student's behavior frame by frame, and produces a real-time risk score alongside detailed telemetry. A built-in browser dashboard lets you observe everything live.

---

## ✨ Features

### 👁️ Face & Head Analysis
- **Face detection** via MediaPipe Face Mesh (up to 5 simultaneous faces)
- **Head pose estimation** using OpenCV `solvePnP` (yaw, pitch, roll)
- **Eye/gaze tracking** — horizontal and vertical iris position relative to a personal baseline
- **Pitch ratio** — chin-to-nose-to-forehead ratio for vertical head tilt
- **Face frontalness & eye symmetry** scores for confidence gating

### 🎯 Attention Model
- Per-face attention state machine: `FOCUSED → TRANSITIONING → LOOKING_AWAY`
- Three independent sub-signals: horizontal head motion, gaze deviation, face orientation
- EMA-smoothed baselines that adapt to natural head movement
- Fast-trigger paths for sudden large yaw/pitch/gaze jumps
- Calibration-personalized noise floors applied to all thresholds

### 📦 Object Detection (YOLOv8)
- **Phone detection** — primary (`cell phone`) and surrogate classes (`remote`, `laptop`, `book`, `mouse`, `tv`)
- **Person detection** — multi-person awareness with pose-skeleton validation
- **Audio device detection** — earbuds and headsets via optional gadget model
- Probe mode for low-confidence re-scan when primary pass misses

### 🤝 Hand-Face Interaction
- MediaPipe Hands detects when a hand covers the face
- Configurable overlap threshold and bounding box expansion

### 🔴 Risk Engine
- 11 risk rules: `multiple_persons`, `multiple_faces`, `phone_detected`, `wearing_audio_device`, `face_missing`, `face_not_facing_camera`, `extreme_gaze`, `vertical_head_motion`, `horizontal_head_motion`, `high_movement`, `hand_on_face`
- Temporal activation/release gates per rule (no flickering)
- EMA + moving-average score smoothing
- Streak-based repetition multipliers

### 🏆 Three-Tier Violation System
| Tier | Description |
|------|-------------|
| **Tier 1** | Single brief trigger — logged silently for post-exam review |
| **Tier 2** | Same rule fires 3+ times within a 120-second window — flagged for human review |
| **Tier 3** | Dangerous rule combination or sustained high-severity violation — immediate alert |

Tier 3 combos include: `phone + gaze away`, `phone + face hidden`, `extra person + facing away`, and more.

### 🎛️ 8-Stage Calibration System
Guides the user through a personalized setup before the exam:
1. Neutral baseline
2. Gaze center
3. Gaze left edge
4. Gaze right edge
5. Gaze top edge
6. Gaze bottom edge
7. Head vertical movement
8. Hand-to-face gesture

After calibration, personalized thresholds for gaze boundaries, pitch ratio, background motion zones, and head motion are applied automatically.

### 🖥️ Hardware-Adaptive Pipeline
Automatically detects hardware at startup and selects one of three modes:

| Mode | Frame Stride | Resolution | YOLO FPS |
|------|-------------|------------|----------|
| `CPU_MODE` | Every 4th frame | ~480p | 3.0 |
| `MID_GPU_MODE` | Every 2nd frame | ~720p | 8.0 |
| `HIGH_GPU_MODE` | Every frame | Up to full | 12.0 |

Dynamic stride boost kicks in under CPU/latency pressure. Automatic GPU→CPU fallback on CUDA errors.

### 📊 Performance Reporting
- Per-frame metrics: FPS, frame time, per-stage timings, dropped frames
- System resources: CPU%, RAM, GPU utilization, temperature, power, VRAM
- Session timeline + threshold-crossing event log
- JSON export for cross-device comparison

### 🔧 Live Threshold Tuner
Sliders in the browser dashboard let you adjust any detection threshold in real time without restarting the server — gaze sensitivity, head motion gates, phone confidence, eye visibility cutoff, and more.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.10+, FastAPI, WebSocket |
| Face/Hand | MediaPipe Face Mesh, MediaPipe Hands |
| Object Detection | YOLOv8 (Ultralytics) |
| Head Pose | OpenCV `solvePnP` |
| Tracking | Custom centroid tracker + OpenCV KCF/CSRT |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Inference | PyTorch (CPU or CUDA) |

---

## 🚀 Quick Start

### 1. Install Python 3.10+
```bash
python --version
```

### 2. Install PyTorch (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
> For CPU-only: `pip install torch torchvision torchaudio`

### 3. Install dependencies
```bash
pip install fastapi "uvicorn[standard]" opencv-python ultralytics mediapipe==0.10.9 numpy scipy websockets psutil
```

### 4. Run the server
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Open the dashboard
```
http://127.0.0.1:8000
```

Click **Run Calibration** before starting monitoring for best accuracy.

---

## 📁 Project Structure

```
├── main.py                    # FastAPI app + WebSocket handler + ProctoringSession
├── config.py                  # All settings in one frozen dataclass
├── frontend/index.html        # Live browser dashboard
├── vision/
│   ├── risk_engine.py         # Risk scoring + attention model
│   ├── attention_model.py     # Per-face attention state machine
│   ├── tier_engine.py         # 3-tier violation classifier
│   ├── calibration.py         # 8-stage calibration manager
│   ├── eye_tracker.py         # Iris-based gaze estimation
│   ├── head_pose.py           # solvePnP head pose estimator
│   ├── face_detector.py       # MediaPipe face detection + temporal filter
│   ├── face_tracker.py        # Centroid-based face tracker
│   ├── object_detector.py     # YOLOv8 phone/person/audio detection
│   ├── hand_face_detector.py  # MediaPipe hands
│   ├── movement_analyzer.py   # Per-face movement score
│   ├── scene_analyzer.py      # Background subtraction + motion zones
│   ├── person_filter.py       # Skeleton + flow validation for persons
│   ├── frame_quality.py       # Low light / dirty camera / blocked view
│   └── person_pose_validator.py
├── tracking/
│   ├── face_tracker.py        # AdaptiveFaceTracker (detector + OpenCV tracker)
│   ├── person_tracker.py      # Centroid person tracker
│   └── face_embedding.py      # Lightweight re-ID descriptor
├── pipeline/
│   ├── scheduler.py           # Hardware-aware frame/YOLO scheduling
│   ├── frame_capture.py       # Dedicated capture thread
│   └── inference_engine.py    # Threaded inference with bounded queues
├── hardware/
│   └── hardware_detector.py   # CPU/MID_GPU/HIGH_GPU profile selection
├── utils/
│   ├── performance.py         # Rolling FPS counter
│   ├── performance_report.py  # Session metric aggregator + JSON export
│   ├── resource_monitor.py    # CPU/RAM/GPU resource sampling
│   └── smoothing.py           # Moving average
└── models/
    ├── schemas.py             # Pydantic request/response models
    ├── face_detector.py
    ├── eye_tracker.py
    └── pose_model.py
```

---

## ⚙️ WebSocket API

Send frames as base64 JPEG:
```json
{ "frame": "<base64>", "client": { "viewportWidth": 1920, "viewportHeight": 1080, "devicePixelRatio": 1 } }
```

Available commands (send instead of a frame):
```json
{ "command": "start_calibration" }
{ "command": "cancel_calibration" }
{ "command": "calibration_status" }
{ "command": "reset_stats" }
{ "command": "set_thresholds", "thresholds": { "gaze_extreme_horizontal_threshold": 0.07 } }
{ "command": "get_thresholds" }
{ "command": "performance_report" }
{ "command": "performance_report_export" }
{ "command": "performance_report_reset" }
```

---

## 🔮 What I Want to Learn Next

This is my first serious computer vision project and there is a lot more to explore:

- **Better gaze estimation** — replace iris ratio heuristics with a proper regression model trained on gaze datasets
- **Face recognition** — add identity verification so the system can detect student substitution
- **Audio anomaly detection** — flag suspicious audio events alongside video signals
- **ONNX/TensorRT export** — optimize inference latency on edge devices
- **Multi-camera support** — cross-view triangulation for a more robust proctoring setup
- **Deeper understanding of MediaPipe internals** — understand how landmark confidence propagates

---

## 📄 License

This project is for learning purposes. Feel free to explore, fork, and build on it.