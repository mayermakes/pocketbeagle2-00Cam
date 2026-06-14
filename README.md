# PocketBeagle Webcam Server

A lightweight Flask-based webcam streaming server with optional person detection and automatic recording. Designed to run on a PocketBeagle 2 (or any Linux SBC with a USB webcam).

---

## Features

- **Live MJPEG stream** accessible from any browser on the local network
- **Person detection** via HOG (CPU-only) or DNN/MobileNet (more accurate)
- **Automatic recording** — starts when a person is detected, stops 3 seconds after they leave
- **Auto-cleanup** — recordings older than 72 hours are deleted every 24 hours
- **Download all recordings** as a ZIP from the web interface
- HTTP Basic Auth protecting all routes

---

## Requirements

- PocketBeagle 2 running Debian/Ubuntu
- Python 3.8+
- USB webcam (detected as `/dev/video0`)
- Network access (wired or USB-network)

---

## Installation

### 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0
```

> `libgl1` and `libglib2.0-0` are needed by OpenCV on headless systems.

### 2. Clone or copy the project files

```
stream_server.py
requirements.txt
```

Optionally, for DNN detection, also include:

```
dnn_model/
  deploy.prototxt
  mobilenet_iter_73000.caffemodel
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** On the PocketBeagle 2 (ARM), `opencv-contrib-python-headless` may take several minutes to install. If it fails, try the system package instead:
> ```bash
> sudo apt install -y python3-opencv
> ```
> Then install only Flask inside the venv:
> ```bash
> pip install flask
> ```

---

## Usage

### Start with no detection (stream only)

```bash
python stream_server.py
```

### Start with HOG person detection

```bash
python stream_server.py --detector hog
```

### Start with DNN person detection

```bash
python stream_server.py --detector dnn
```

### Adjust DNN confidence threshold (default: 0.3)

```bash
python stream_server.py --detector dnn --confidence 0.5
```

---

## Accessing the interface

Once running, open a browser and navigate to:

```
http://<pocketbeagle-ip>:5000
```

You will be prompted for credentials:

| Field    | Value         |
|----------|---------------|
| Username | `admin`       |
| Password | `password123` |

> To change credentials, edit the `USERNAME` and `PASSWORD` constants near the top of `stream_server.py`.

---

## Recording behaviour

| Event | Action |
|---|---|
| Person detected | Recording starts immediately |
| Person leaves frame | 3-second grace period begins |
| No person for 3 seconds | Recording stops and file is saved |
| File older than 72 hours | Automatically deleted (checked every 24 h) |

Recordings are saved to the `records/` folder (created automatically) as `.avi` files named by timestamp, e.g. `20250614_153201.avi`.

To download all recordings at once, click **Download all as ZIP** in the web interface.

---

## Run on boot (optional)

Create a systemd service to start the server automatically:

```bash
sudo nano /etc/systemd/system/webcam-server.service
```

```ini
[Unit]
Description=PocketBeagle Webcam Server
After=network.target

[Service]
User=debian
WorkingDirectory=/home/debian/webcam
ExecStart=/home/debian/webcam/venv/bin/python stream_server.py --detector hog
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable webcam-server
sudo systemctl start webcam-server
```

Check logs:

```bash
sudo journalctl -u webcam-server -f
```

---

## Detector comparison

| Mode | Speed | Accuracy | Notes |
|------|-------|----------|-------|
| `none` | Fastest | No detection | Stream only, no recording trigger |
| `hog` | Fast | Moderate | CPU-only, good for well-lit scenes |
| `dnn` | Slower | Better | Requires MobileNet model files |

> On the PocketBeagle 2, `hog` is recommended. DNN may cause frame rate drops due to limited CPU resources.

---

## DNN model files

If using `--detector dnn`, download the MobileNet SSD model:

```bash
mkdir dnn_model && cd dnn_model

wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt

wget https://drive.google.com/uc?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc \
     -O mobilenet_iter_73000.caffemodel
```

---

## Troubleshooting

**Webcam not found**
```
RuntimeError: Could not open webcam
```
Check that your webcam is connected and visible:
```bash
ls /dev/video*
```

**Low frame rate with DNN**
Switch to `--detector hog` or `--detector none`.

**Port already in use**
Kill the existing process:
```bash
sudo lsof -i :5000
kill <PID>
```
