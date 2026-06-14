import cv2
import argparse
import os
import time
import threading
import zipfile
import io
from datetime import datetime, timedelta
from flask import Flask, Response, request, send_file
from functools import wraps

# =========================
# CLI ARGUMENTS
# =========================
parser = argparse.ArgumentParser(
    description="PocketBeagle webcam streaming server"
)

parser.add_argument(
    "--detector",
    choices=["none", "hog", "dnn"],
    default="none",
    help="Detection backend to use"
)

parser.add_argument(
    "--confidence",
    type=float,
    default=0.3,
    help="Confidence threshold for DNN detection"
)

args = parser.parse_args()

DETECTOR = args.detector.lower()

# =========================
# CONFIG
# =========================
USERNAME = "admin"
PASSWORD = "password123"

MODEL_PROTOTXT = "dnn_model/deploy.prototxt"
MODEL_WEIGHTS  = "dnn_model/mobilenet_iter_73000.caffemodel"

PERSON_CLASS_ID      = 15
RECORDS_DIR          = "records"
NO_PERSON_TIMEOUT    = 3.0   # seconds before stopping recording
CLEANUP_INTERVAL     = 24 * 3600  # 24 hours in seconds
MAX_RECORDING_AGE    = 72 * 3600  # 72 hours in seconds

os.makedirs(RECORDS_DIR, exist_ok=True)

app = Flask(__name__)

# =========================
# AUTHENTICATION
# =========================
def check_auth(username, password):
    return username == USERNAME and password == PASSWORD


def authenticate():
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'}
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth:
            return authenticate()
        if not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# =========================
# LOAD DETECTORS
# =========================
hog = None
net = None

if DETECTOR == "hog":
    print("[INFO] Initializing HOG detector...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

elif DETECTOR == "dnn":
    print("[INFO] Initializing DNN detector...")
    net = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_WEIGHTS)

else:
    print("[INFO] Detection disabled")

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# =========================
# RECORDING STATE
# =========================
recorder        = None   # cv2.VideoWriter or None
recorder_lock   = threading.Lock()
last_seen_time  = None   # timestamp when person was last detected
is_recording    = False


def start_recording():
    global recorder, is_recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(RECORDS_DIR, f"{timestamp}.avi")
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    recorder  = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
    is_recording = True
    print(f"[REC] Started recording: {filepath}")


def stop_recording():
    global recorder, is_recording
    if recorder:
        recorder.release()
        recorder = None
    is_recording = False
    print("[REC] Stopped recording")


def update_recording(person_detected, frame):
    global last_seen_time, is_recording

    now = time.time()

    if person_detected:
        last_seen_time = now
        if not is_recording:
            start_recording()

    if is_recording:
        # stop if person absent for more than NO_PERSON_TIMEOUT
        if last_seen_time is not None and (now - last_seen_time) > NO_PERSON_TIMEOUT:
            stop_recording()
        else:
            with recorder_lock:
                if recorder:
                    recorder.write(frame)

# =========================
# CLEANUP THREAD
# =========================
def cleanup_old_recordings():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        cutoff = time.time() - MAX_RECORDING_AGE
        for fname in os.listdir(RECORDS_DIR):
            fpath = os.path.join(RECORDS_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                print(f"[CLEANUP] Deleted old recording: {fname}")


cleanup_thread = threading.Thread(target=cleanup_old_recordings, daemon=True)
cleanup_thread.start()

# =========================
# HOG DETECTOR
# =========================
def detect_hog(frame):
    boxes, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )
    person_detected = len(boxes) > 0
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person (HOG)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, person_detected

# =========================
# DNN DETECTOR
# =========================
def detect_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )
    net.setInput(blob)
    detections = net.forward()

    person_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < args.confidence:
            continue
        class_id = int(detections[0, 0, i, 1])
        if class_id != PERSON_CLASS_ID:
            continue
        person_detected = True
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype("int")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame, person_detected

# =========================
# UNIFIED DETECTOR
# =========================
def detect_people(frame):
    if DETECTOR == "hog":
        return detect_hog(frame)
    if DETECTOR == "dnn":
        return detect_dnn(frame)
    return frame, False

# =========================
# VIDEO STREAM
# =========================
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame, person_detected = detect_people(frame)

        # overlay REC indicator
        if is_recording:
            cv2.circle(frame, (620, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (628, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        update_recording(person_detected, frame.copy())

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# =========================
# WEB ROUTES
# =========================
@app.route("/")
@requires_auth
def index():
    recordings = sorted(os.listdir(RECORDS_DIR), reverse=True)
    rec_rows = ""
    for fname in recordings:
        fpath = os.path.join(RECORDS_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
        rec_rows += f"""
        <tr>
            <td>{fname}</td>
            <td>{mtime}</td>
            <td>{size_mb:.1f} MB</td>
        </tr>"""

    rec_count = len(recordings)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PocketBeagle Cam</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Courier New', monospace;
            background: #0d0f12;
            color: #c8d0d8;
            min-height: 100vh;
            padding: 24px;
        }}
        header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            border-bottom: 1px solid #1e2530;
            padding-bottom: 16px;
        }}
        header h1 {{
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #e0e6ee;
        }}
        .badge {{
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 3px;
            background: #1a2a1a;
            color: #4caf50;
            border: 1px solid #2a4a2a;
            letter-spacing: 0.08em;
        }}
        .badge.dnn {{ background: #1a1a2a; color: #7c9fec; border-color: #2a2a4a; }}
        .badge.none {{ background: #1e2530; color: #7a8899; border-color: #2a3545; }}
        .layout {{
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
            align-items: start;
        }}
        @media (max-width: 900px) {{
            .layout {{ grid-template-columns: 1fr; }}
        }}
        .feed-wrap {{
            background: #111418;
            border: 1px solid #1e2530;
            border-radius: 6px;
            overflow: hidden;
        }}
        .feed-wrap img {{
            width: 100%;
            display: block;
        }}
        .feed-label {{
            padding: 8px 14px;
            font-size: 0.7rem;
            color: #4a5a6a;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}
        .panel {{
            background: #111418;
            border: 1px solid #1e2530;
            border-radius: 6px;
            overflow: hidden;
        }}
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid #1e2530;
        }}
        .panel-header h2 {{
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #8a9aaa;
        }}
        .count {{
            font-size: 0.7rem;
            color: #4a5a6a;
        }}
        .btn-download {{
            display: block;
            width: calc(100% - 32px);
            margin: 14px 16px;
            padding: 10px;
            background: #1a2a3a;
            color: #7cb8e0;
            border: 1px solid #2a3a4a;
            border-radius: 4px;
            font-family: inherit;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            transition: background 0.15s;
        }}
        .btn-download:hover {{ background: #223040; }}
        .btn-download:disabled {{
            opacity: 0.35;
            cursor: not-allowed;
        }}
        .rec-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.72rem;
        }}
        .rec-table th {{
            padding: 7px 16px;
            text-align: left;
            color: #4a5a6a;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 400;
            border-bottom: 1px solid #1a2230;
        }}
        .rec-table td {{
            padding: 7px 16px;
            border-bottom: 1px solid #161c24;
            color: #8a9aaa;
        }}
        .rec-table tr:last-child td {{ border-bottom: none; }}
        .rec-table tr:hover td {{ background: #141920; }}
        .empty {{
            padding: 28px 16px;
            text-align: center;
            color: #3a4a5a;
            font-size: 0.78rem;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>
    <header>
        <h1>PocketBeagle Cam</h1>
        <span class="badge {'dnn' if DETECTOR == 'dnn' else 'none' if DETECTOR == 'none' else ''}">{DETECTOR.upper()}</span>
    </header>

    <div class="layout">
        <div class="feed-wrap">
            <img src="/video_feed" alt="Live feed">
            <div class="feed-label">Live — 640 &times; 480</div>
        </div>

        <div class="panel">
            <div class="panel-header">
                <h2>Recordings</h2>
                <span class="count">{rec_count} file{'s' if rec_count != 1 else ''}</span>
            </div>

            {'<a href="/download_all" class="btn-download">&#8595; Download all as ZIP</a>' if rec_count > 0 else '<span class="btn-download" style="opacity:0.3;cursor:default;">No recordings yet</span>'}

            {'<table class="rec-table"><thead><tr><th>File</th><th>Recorded</th><th>Size</th></tr></thead><tbody>' + rec_rows + '</tbody></table>' if rec_count > 0 else '<div class="empty">No recordings yet.<br>Recordings start automatically<br>when a person is detected.</div>'}
        </div>
    </div>
</body>
</html>"""


@app.route("/video_feed")
@requires_auth
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/download_all")
@requires_auth
def download_all():
    files = [f for f in os.listdir(RECORDS_DIR)
             if os.path.isfile(os.path.join(RECORDS_DIR, f))]

    if not files:
        return Response("No recordings available", status=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in files:
            zf.write(os.path.join(RECORDS_DIR, fname), fname)
    buf.seek(0)

    zip_name = f"recordings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name=zip_name)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print(f"[INFO] Detector mode: {DETECTOR}")
    print(f"[INFO] Recordings saved to: {os.path.abspath(RECORDS_DIR)}")
    app.run(host="0.0.0.0", port=5000, threaded=True)
