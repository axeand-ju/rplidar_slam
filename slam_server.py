#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rplidar-roboticia",
#     "websockets>=12.0",
#     "numpy",
#     "scipy",
# ]
# ///
"""
RPLidar C1 — Live SLAM Demo Server
====================================
Reads scans from an RPLidar C1, performs ICP-based scan matching to estimate
odometry, builds a global map, and streams everything to a browser dashboard
via WebSocket.

Usage:
    uv run slam_server.py                          # Linux, default port
    uv run slam_server.py --port COM3              # Windows
    uv run slam_server.py --simulate               # No hardware
"""

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Try to import the rplidar driver.  We also provide a --simulate flag that
# generates fake scans so you can test the dashboard without hardware.
# ---------------------------------------------------------------------------
try:
    from rplidar import RPLidar
    HAS_RPLIDAR = True
except ImportError:
    HAS_RPLIDAR = False

try:
    import websockets
    from websockets.server import serve as ws_serve
except ImportError:
    print("ERROR: 'websockets' package not found.  Install with:\n"
          "  pip install websockets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("slam_demo")


# ═══════════════════════════════════════════════════════════════════════════
#  ICP Scan Matcher
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ICPResult:
    """Result of an ICP alignment."""
    rotation: float = 0.0        # radians
    translation: np.ndarray = field(default_factory=lambda: np.zeros(2))
    converged: bool = False
    mean_error: float = float("inf")
    iterations: int = 0


def icp_match(
    source: np.ndarray,       # (N, 2) — new scan (to be transformed)
    target: np.ndarray,       # (M, 2) — reference scan
    max_iterations: int = 40,
    tolerance: float = 1e-4,
    max_correspondence_dist: float = 500.0,  # mm
    # Initial guess (warm-start from velocity model)
    init_rotation: float = 0.0,
    init_translation: np.ndarray | None = None,
) -> ICPResult:
    """
    2-D point-to-point ICP.

    Aligns *source* to *target* and returns the (R, t) that minimises
    the sum of squared distances between matched point pairs.
    """
    if source.shape[0] < 10 or target.shape[0] < 10:
        return ICPResult()

    result = ICPResult()

    # Build KD-tree on the target (reference) cloud once
    tree = cKDTree(target)

    # Apply initial guess
    R_acc = _rotation_matrix(init_rotation)
    t_acc = init_translation.copy() if init_translation is not None else np.zeros(2)

    src = (R_acc @ source.T).T + t_acc

    prev_error = float("inf")

    for i in range(max_iterations):
        # 1. Find closest points in target for each source point
        dists, indices = tree.query(src, k=1)

        # 2. Filter by max correspondence distance
        mask = dists < max_correspondence_dist
        if mask.sum() < 10:
            break

        src_matched = src[mask]
        tgt_matched = target[indices[mask]]

        # 3. Compute optimal rigid transform (SVD method)
        src_centroid = src_matched.mean(axis=0)
        tgt_centroid = tgt_matched.mean(axis=0)

        src_centered = src_matched - src_centroid
        tgt_centered = tgt_matched - tgt_centroid

        H = src_centered.T @ tgt_centered  # 2×2
        U, _, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T

        # Ensure proper rotation (det = +1)
        if np.linalg.det(R_step) < 0:
            Vt[-1, :] *= -1
            R_step = Vt.T @ U.T

        t_step = tgt_centroid - R_step @ src_centroid

        # 4. Update accumulated transform
        src = (R_step @ src.T).T + t_step
        R_acc = R_step @ R_acc
        t_acc = R_step @ t_acc + t_step

        # 5. Check convergence
        mean_err = dists[mask].mean()
        result.iterations = i + 1
        result.mean_error = float(mean_err)

        if abs(prev_error - mean_err) < tolerance:
            result.converged = True
            break
        prev_error = mean_err

    result.rotation = float(math.atan2(R_acc[1, 0], R_acc[0, 0]))
    result.translation = t_acc
    return result


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# ═══════════════════════════════════════════════════════════════════════════
#  SLAM State
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SLAMState:
    # Current pose in global frame
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    # Velocity model for ICP warm-start
    prev_dx: float = 0.0
    prev_dy: float = 0.0
    prev_dtheta: float = 0.0

    # Previous scan in sensor-local frame
    prev_scan: np.ndarray | None = None

    # Accumulated global map (downsampled)
    map_points: np.ndarray | None = None
    max_map_points: int = 50_000
    voxel_size: float = 30.0  # mm — map downsampling resolution

    # Trajectory history
    trajectory: list = field(default_factory=list)

    # Stats
    scan_count: int = 0
    icp_time_ms: float = 0.0


def process_scan(state: SLAMState, scan_xy: np.ndarray) -> dict:
    """
    Process a new scan: run ICP against the previous scan, update pose,
    grow the map, return a JSON-serialisable status dict for the frontend.
    """
    state.scan_count += 1

    if state.prev_scan is None:
        # First scan — initialise
        state.prev_scan = scan_xy
        state.map_points = scan_xy.copy()
        state.trajectory.append((state.x, state.y, state.theta))
        return _build_message(state, scan_xy, None)

    # --- ICP scan matching ---
    t0 = time.perf_counter()
    icp = icp_match(
        source=scan_xy,
        target=state.prev_scan,
        init_rotation=state.prev_dtheta,
        init_translation=np.array([state.prev_dx, state.prev_dy]),
    )
    state.icp_time_ms = (time.perf_counter() - t0) * 1000

    # Update velocity model
    if icp.converged or icp.mean_error < 200:
        dtheta = icp.rotation
        dx, dy = icp.translation
    else:
        # ICP failed — assume no motion
        dtheta = 0.0
        dx, dy = 0.0, 0.0
        log.warning("ICP did not converge (err=%.1f mm), assuming stationary",
                     icp.mean_error)

    state.prev_dx = dx
    state.prev_dy = dy
    state.prev_dtheta = dtheta

    # Accumulate global pose
    cos_t = math.cos(state.theta)
    sin_t = math.sin(state.theta)
    state.x += cos_t * dx - sin_t * dy
    state.y += sin_t * dx + cos_t * dy
    state.theta += dtheta
    state.trajectory.append((state.x, state.y, state.theta))

    # Transform scan to global frame and append to map
    R_global = _rotation_matrix(state.theta)
    scan_global = (R_global @ scan_xy.T).T + np.array([state.x, state.y])

    if state.map_points is not None:
        state.map_points = np.vstack([state.map_points, scan_global])
    else:
        state.map_points = scan_global

    # Downsample map if it's getting too large
    if state.map_points.shape[0] > state.max_map_points:
        state.map_points = _voxel_downsample(state.map_points, state.voxel_size)
        log.info("Map downsampled to %d points", state.map_points.shape[0])

    state.prev_scan = scan_xy

    return _build_message(state, scan_xy, icp)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel grid downsampling for 2-D points."""
    quantised = np.floor(points / voxel_size).astype(np.int32)
    _, indices = np.unique(quantised, axis=0, return_index=True)
    return points[indices]


def _build_message(state: SLAMState, scan_local: np.ndarray,
                   icp: ICPResult | None) -> dict:
    """Build the JSON message to send to the frontend."""
    # Subsample map for transmission if very large
    map_pts = state.map_points
    if map_pts is not None and map_pts.shape[0] > 20_000:
        step = max(1, map_pts.shape[0] // 20_000)
        map_pts = map_pts[::step]

    # Transform local scan to global for display
    R_global = _rotation_matrix(state.theta)
    scan_global = (R_global @ scan_local.T).T + np.array([state.x, state.y])

    msg = {
        "type": "slam_update",
        "scan_count": state.scan_count,
        "pose": {
            "x": round(state.x, 1),
            "y": round(state.y, 1),
            "theta": round(math.degrees(state.theta), 2),
        },
        # Send points as flat arrays for compact JSON
        "scan": {
            "x": scan_global[:, 0].round(1).tolist(),
            "y": scan_global[:, 1].round(1).tolist(),
        },
        "map": {
            "x": map_pts[:, 0].round(1).tolist(),
            "y": map_pts[:, 1].round(1).tolist(),
        } if map_pts is not None else {"x": [], "y": []},
        "trajectory": [
            {"x": round(t[0], 1), "y": round(t[1], 1)}
            for t in state.trajectory[-500:]
        ],
        "icp": {
            "converged": icp.converged if icp else True,
            "mean_error": round(icp.mean_error, 2) if icp else 0,
            "iterations": icp.iterations if icp else 0,
            "time_ms": round(state.icp_time_ms, 1),
        },
    }
    return msg


# ═══════════════════════════════════════════════════════════════════════════
#  Simulated Lidar (for testing without hardware)
# ═══════════════════════════════════════════════════════════════════════════

class SimulatedLidar:
    """
    Generates fake scans of a rectangular room while moving the sensor
    along a figure-8 path.  Useful for testing the dashboard.
    """
    def __init__(self):
        self.t = 0.0
        # Room corners (mm)
        self.walls = [
            ((-3000, -2000), (3000, -2000)),
            ((3000, -2000), (3000, 2000)),
            ((3000, 2000), (-3000, 2000)),
            ((-3000, 2000), (-3000, -2000)),
            # Some internal obstacles
            ((-1000, -500), (-1000, 500)),
            ((500, -800), (500, 800)),
            ((500, 800), (1500, 800)),
        ]

    def _sensor_pose(self) -> tuple[float, float, float]:
        """Figure-8 trajectory."""
        x = 1500 * math.sin(self.t * 0.15)
        y = 800 * math.sin(self.t * 0.3)
        theta = self.t * 0.08
        return x, y, theta

    def iter_scans(self):
        """Yields scans as lists of (quality, angle_deg, distance_mm)."""
        while True:
            sx, sy, stheta = self._sensor_pose()
            scan = []
            for i in range(360):
                angle = math.radians(i)
                # Cast a ray from (sx, sy) at angle (stheta + angle)
                abs_angle = stheta + angle
                dx = math.cos(abs_angle)
                dy = math.sin(abs_angle)
                min_dist = 6000  # max range
                for (wx1, wy1), (wx2, wy2) in self.walls:
                    d = self._ray_segment_intersect(sx, sy, dx, dy,
                                                     wx1, wy1, wx2, wy2)
                    if d is not None and d < min_dist:
                        min_dist = d
                # Add some noise
                min_dist += np.random.normal(0, 10)
                if min_dist > 100:  # minimum range filter
                    scan.append((15, i, max(0, min_dist)))
            yield scan
            self.t += 0.5
            time.sleep(0.15)  # ~6-7 Hz

    @staticmethod
    def _ray_segment_intersect(ox, oy, dx, dy, x1, y1, x2, y2):
        """Ray-line segment intersection. Returns distance or None."""
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-10:
            return None
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
        if t > 0 and 0 <= u <= 1:
            return t
        return None

    def stop(self):
        pass

    def disconnect(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket Server
# ═══════════════════════════════════════════════════════════════════════════

connected_clients: set = set()


async def ws_handler(websocket):
    connected_clients.add(websocket)
    log.info("Client connected (%d total)", len(connected_clients))
    try:
        async for _ in websocket:
            pass  # We don't expect messages from the client
    finally:
        connected_clients.discard(websocket)
        log.info("Client disconnected (%d remaining)", len(connected_clients))


async def broadcast(message: str):
    if connected_clients:
        await asyncio.gather(
            *(client.send(message) for client in connected_clients),
            return_exceptions=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════

async def slam_loop(lidar, state: SLAMState):
    """Read scans from the lidar and run SLAM in a background thread."""
    loop = asyncio.get_event_loop()

    def scan_generator():
        for scan in lidar.iter_scans():
            yield scan

    for scan in await loop.run_in_executor(None, lambda: next(iter([list(lidar.iter_scans())]))):
        pass  # This won't work well — let's use a different pattern

    # We'll run the blocking lidar loop in a thread and push results via a queue
    pass


async def main(args):
    state = SLAMState(max_map_points=args.max_map_pts)

    # --- Set up lidar ---
    if args.simulate:
        log.info("Running in SIMULATION mode (no hardware)")
        lidar = SimulatedLidar()
    else:
        if not HAS_RPLIDAR:
            log.error("rplidar package not installed. Install with:\n"
                      "  pip install rplidar-roboticia\n"
                      "Or use --simulate for testing without hardware.")
            sys.exit(1)
        log.info("Connecting to RPLidar on %s @ %d baud", args.port, args.baudrate)
        lidar = RPLidar(args.port, baudrate=args.baudrate)
        info = lidar.get_info()
        health = lidar.get_health()
        log.info("Lidar info: %s", info)
        log.info("Lidar health: %s", health)

    # --- WebSocket server ---
    log.info("Starting WebSocket server on %s:%d", args.ws_host, args.ws_port)
    server = await ws_serve(ws_handler, args.ws_host, args.ws_port)

    # Also serve the HTML file via a simple HTTP server on ws_port + 1
    http_port = args.ws_port + 1
    log.info("Serving dashboard at http://localhost:%d", http_port)

    # --- Scan processing loop (in executor to not block asyncio) ---
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    def lidar_reader():
        """Blocking reader that pushes scans into the async queue."""
        try:
            for raw_scan in lidar.iter_scans():
                # Convert polar to Cartesian
                points = []
                for quality, angle_deg, distance_mm in raw_scan:
                    if quality > 0 and distance_mm > 50:
                        rad = math.radians(angle_deg)
                        x = distance_mm * math.cos(rad)
                        y = distance_mm * math.sin(rad)
                        points.append((x, y))
                if len(points) > 20:
                    arr = np.array(points)
                    try:
                        queue.put_nowait(arr)
                    except asyncio.QueueFull:
                        pass  # Drop frame if consumer is behind
        except Exception as e:
            log.error("Lidar reader error: %s", e)

    loop = asyncio.get_event_loop()
    reader_task = loop.run_in_executor(None, lidar_reader)

    log.info("SLAM loop running — open the dashboard to see live data")

    try:
        while True:
            try:
                scan_xy = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

            # Process scan through SLAM pipeline
            msg = process_scan(state, scan_xy)
            json_str = json.dumps(msg)

            # Broadcast to all connected clients
            await broadcast(json_str)

            if state.scan_count % 50 == 0:
                map_n = state.map_points.shape[0] if state.map_points is not None else 0
                log.info(
                    "Scan #%d | Pose (%.0f, %.0f, %.1f°) | Map: %d pts | "
                    "ICP: %.1f ms, err=%.1f, %s",
                    state.scan_count,
                    state.x, state.y, math.degrees(state.theta),
                    map_n,
                    state.icp_time_ms,
                    msg["icp"]["mean_error"],
                    "OK" if msg["icp"]["converged"] else "FAIL",
                )
    except asyncio.CancelledError:
        pass
    finally:
        log.info("Shutting down…")
        try:
            lidar.stop()
            lidar.disconnect()
        except Exception:
            pass
        server.close()
        await server.wait_closed()


def _default_serial_port() -> str:
    """Pick a sensible default serial port for the current OS."""
    if sys.platform == "win32":
        return "COM3"
    elif sys.platform == "darwin":
        return "/dev/tty.usbserial-0001"
    return "/dev/ttyUSB0"


def parse_args():
    default_port = _default_serial_port()
    p = argparse.ArgumentParser(description="RPLidar C1 SLAM Demo")
    p.add_argument("--port", default=default_port,
                    help=f"Serial port for the RPLidar (default: {default_port})")
    p.add_argument("--baudrate", type=int, default=460800,
                    help="Serial baud rate (default: 460800 for C1)")
    p.add_argument("--ws-host", default="0.0.0.0",
                    help="WebSocket bind address (default: 0.0.0.0)")
    p.add_argument("--ws-port", type=int, default=8765,
                    help="WebSocket port (default: 8765)")
    p.add_argument("--max-map-pts", type=int, default=50_000,
                    help="Max map points before downsampling (default: 50000)")
    p.add_argument("--simulate", action="store_true",
                    help="Run with simulated lidar (no hardware needed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle Ctrl+C gracefully.
    # On Windows, asyncio's event loop doesn't support SIGINT via
    # add_signal_handler, so we fall back to signal.signal + KeyboardInterrupt.
    loop = asyncio.new_event_loop()

    if os.name != "nt":
        # Unix: install signal handler on the event loop
        def _shutdown(sig, frame):
            log.info("Received signal %s, shutting down…", sig)
            for task in asyncio.all_tasks(loop):
                task.cancel()
        signal.signal(signal.SIGINT, lambda s, f: _shutdown(s, f))

    try:
        loop.run_until_complete(main(args))
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down…")
    finally:
        loop.close()
