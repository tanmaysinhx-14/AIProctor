# 🚀 AI Model Setup Guide

This guide walks you through installing dependencies, setting up a virtual environment, and running the AI model.

---

## 1️⃣ Install Python (Version 3.10+)

Download and install Python from:  
https://www.python.org/downloads/

Verify installation:

```bash
python --version
```

---

## 2️⃣ Upgrade PIP

Make sure `pip` is installed and up to date:

```bash
python -m pip install --upgrade pip
```

---

## 3️⃣ Install PyTorch (with CUDA 12.1)

Install Torch, Torchvision, and Torchaudio:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ Ensure your system supports CUDA 12.1.  
> If you do not have a compatible GPU, install the CPU version instead.

---

## 4️⃣ Install Required Python Libraries

```bash
pip install fastapi "uvicorn[standard]" opencv-python ultralytics mediapipe==0.10.9 numpy scipy websockets
```

---

## 5️⃣ Create a Virtual Environment

Create a Python 3.11 virtual environment:

```bash
py -3.11 -m venv .venv
```

---

## 6️⃣ Activate the Virtual Environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 7️⃣ Run the AI Model

Start the FastAPI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Once running, open your browser and visit:

```
http://127.0.0.1:8000
```

---

# ✅ Setup Complete

Your AI model should now be running locally.

If you encounter issues:
- Verify Python version
- Ensure the virtual environment is activated
- Check CUDA compatibility (if using GPU)

---

## Performance Report Module

This backend now records dynamic runtime metrics into a session-level report that is stable for analysis and LLM usage.

What it captures:
- Throughput: `fps`, `totalTimeMs`, dropped frames
- CPU/RAM: system + process utilization and memory
- GPU: utilization, memory pressure, temperature, power
- Efficiency: `fpsPerCpuProcess`, `fpsPerCpuSystem`, `frameMsPerCpuProcess`, `frameMsPerCpuSystem`
- Device context: hostname, OS, CPU, RAM, GPU(s), Torch/CUDA versions, device fingerprint
- Important events: spikes/pressure/recovery markers with timestamps

### WebSocket Commands

Send these JSON messages to `/ws`:

```json
{"command":"performance_report_status"}
```

```json
{"command":"performance_report"}
```

```json
{"command":"performance_report_export"}
```

```json
{"command":"performance_report_reset"}
```

Notes:
- `performance_report` returns full structured report JSON (LLM-ready fields included).
- `performance_report_export` writes a `.json` file in `reports/` (configurable).
- Report auto-export can happen when the WebSocket session closes.

### Cross-Laptop Comparison

For comparing the same model/scenario on different laptops, use:
- `report.device` for hardware/software context
- `report.comparisonVector` for direct comparable numbers
- `report.observations` + `report.eventLog` for important behavior

## Adaptive Runtime Pipeline (Hardware-Aware)

The backend now auto-selects runtime mode at startup:
- `CPU_MODE`
- `MID_GPU_MODE`
- `HIGH_GPU_MODE`

Behavior adapts by mode:
- Dynamic frame stride (CPU: every 4th, mid GPU: every 2nd, high GPU: every frame)
- Face detection interval (CPU: every 10 frames, GPU: every 5 frames)
- Dynamic resolution target (CPU ~480p, mid GPU ~720p, high GPU up to full input)
- YOLO inference frequency tuned per mode
- Added horizontal-head-motion risk signal for excessive lateral movement patterns
- Calibration now learns eye-gaze sensitivity and persistent background-motion zones, then applies those zones to face/object filtering after calibration

### Threaded Pipeline Layout

Modules added:
- `hardware/hardware_detector.py`
- `pipeline/frame_capture.py`
- `pipeline/inference_engine.py`
- `pipeline/scheduler.py`
- `tracking/face_tracker.py`
- `ui/renderer.py`

Runtime flow (websocket mode):
1. Frame ingest loop (capture)
2. Inference worker (AI processing)
3. Render worker (non-blocking send of latest result)

### GPU Fallback Safety

If GPU inference fails at runtime, the session automatically switches to CPU profile and continues processing.
A `GPU_FALLBACK` warning is emitted in telemetry.
