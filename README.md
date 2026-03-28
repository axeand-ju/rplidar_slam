# RPLidar C1 — Live SLAM Demo

A self-contained demo for the **RPLidar C1** that visualizes live scans and performs
incremental scan-matching SLAM (ICP-based), building a persistent map as you move
the sensor around the room.

## Architecture

```
RPLidar C1 (USB) ─► Python backend (rplidarc1 + ICP SLAM) ─► WebSocket ─► Browser dashboard
```

- **Backend** (`slam_server.py`): Reads scans from the lidar, runs ICP scan matching
  between consecutive frames, accumulates a global map, and streams everything to
  connected browsers via WebSocket.
- **Frontend** (`index.html`): A single-file dashboard that renders the live scan,
  the accumulated map, and the estimated trajectory in real time.

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) — or pip

Dependencies (`rplidarc1`, `websockets`, `numpy`, `scipy`) are declared
in `pyproject.toml` and in the PEP 723 inline metadata inside `slam_server.py`,
so `uv` handles everything automatically.

## Quick Start

1. Plug in the RPLidar C1 via USB.
2. Find the serial port:
   - **Linux**: typically `/dev/ttyUSB0` (check `ls /dev/ttyUSB*`)
   - **Windows**: `COM3`, `COM4`, etc. (check Device Manager → Ports)
   - **macOS**: `/dev/tty.usbserial-*` (check `ls /dev/tty.usb*`)
3. Start the server — `uv` will create a venv and install dependencies on first run:

```bash
# Linux (default port is /dev/ttyUSB0)
uv run slam_server.py

# Windows
uv run slam_server.py --port COM3

# Test without hardware on any OS
uv run slam_server.py --simulate
```

Alternatively, set up as a project:

```bash
uv sync
uv run slam_server.py --simulate
```

Or without uv:

```bash
pip install rplidarc1 websockets numpy scipy
python slam_server.py --simulate
```

4. Open `http://localhost:8765` in a browser (or just open `index.html` and it
   will connect to `ws://localhost:8765`).

### Command-line options

| Flag            | Default         | Description                        |
|-----------------|-----------------|------------------------------------|
| `--port`        | *auto-detected* | Serial port of the RPLidar         |
| `--baudrate`    | `460800`        | Serial baud rate (C1 default)      |
| `--ws-host`     | `0.0.0.0`       | WebSocket bind address             |
| `--ws-port`     | `8765`          | WebSocket port                     |
| `--max-map-pts` | `50000`         | Downsample map when exceeding this |

## How the SLAM works

This is a **teaching demo**, not a production SLAM stack. The approach:

1. **Scan acquisition**: Each 360° scan from the C1 yields a polar point cloud
   (angle, distance). We convert to Cartesian (x, y) in the sensor frame.
2. **ICP scan matching**: We align the new scan against the previous scan using
   Iterative Closest Point (ICP) with a KD-tree for nearest-neighbor queries.
   This gives us a relative transform (dx, dy, dθ) between frames.
3. **Pose accumulation**: The relative transforms are chained to maintain a global
   pose estimate. Drift will accumulate — this is expected in a simple demo
   without loop closure.
4. **Map building**: Each scan is transformed into global coordinates and appended
   to a growing map point cloud (with voxel downsampling to keep things fast).

Students can observe:
- How ICP converges (or fails with large motions)
- How drift accumulates without loop closure
- The difference between local scan and global map
- The effect of environment geometry on matching quality
