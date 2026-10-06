"""A minimal `Jarvis.app`, so macOS shows "Jarvis" instead of "python".

macOS names a process after its executable, and names apps (Dock, menu bar, Login Items,
the microphone permission) after the bundle around it. The bundle holds a copy of the
virtual environment's Python as `Contents/MacOS/Jarvis`, an Info.plist and an icon, ad hoc
signed so the name is bound to it. Python finds its standard library and the project's
packages through PYTHONHOME and PYTHONPATH, set by whoever starts it (the LaunchAgent,
or `voice` when it opens the HUD window).
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import shutil
import site
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

from backend.config import Settings

log = logging.getLogger(__name__)

BUNDLE_ID = "local.my-jarvis"
LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework"
    "/Support/lsregister"
)


def app_path(s: Settings) -> Path:
    return s.data_path / f"{s.assistant_name}.app"


def executable(s: Settings) -> Path:
    return app_path(s) / "Contents" / "MacOS" / _exe_name(s)


def _exe_name(s: Settings) -> str:
    return re.sub(r"[^\w.-]", "", s.assistant_name) or "Jarvis"


def python_env() -> dict[str, str]:
    """What the bundled Python needs to find the standard library and our packages.

    Inside the bundle (no venv there), the variables it was started with are passed on.
    """
    if os.environ.get("PYTHONHOME"):
        keys = ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE")
        return {k: os.environ[k] for k in keys if k in os.environ}
    packages = [p for p in site.getsitepackages() if p.startswith(sys.prefix)]
    return {
        "PYTHONHOME": os.path.realpath(sys.base_prefix),
        "PYTHONPATH": os.pathsep.join(packages),
        "PYTHONNOUSERSITE": "1",
    }


def launcher(s: Settings) -> tuple[str | None, dict[str, str] | None]:
    """Program and environment to start a Python child as the app, if it was built."""
    exe = executable(s)
    if not exe.exists():
        return None, None
    return str(exe), {**os.environ, **python_env()}


def info_plist(s: Settings) -> dict:
    return {
        "CFBundleName": s.assistant_name,
        "CFBundleDisplayName": s.assistant_name,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": _exe_name(s),
        "CFBundleIconFile": "icon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        # Required: without it, macOS ends a bundled app that opens the microphone.
        "NSMicrophoneUsageDescription": (
            f'{s.assistant_name} listens for "Hey {s.assistant_name}" and for your requests, '
            "and transcribes them on this Mac."
        ),
    }


def build(s: Settings) -> Path:
    """(Re)create the bundle from the current Python. Returns its path."""
    app = app_path(s)
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    exe = macos / _exe_name(s)
    shutil.copy2(os.path.realpath(sys.executable), exe)  # a copy: a link keeps "python"
    with (app / "Contents" / "Info.plist").open("wb") as f:
        plistlib.dump(info_plist(s), f)
    try:
        write_icon(resources / "icon.icns")
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("App icon not created: %s", exc)
    subprocess.run(
        ["codesign", "--force", "--sign", "-", "--identifier", BUNDLE_ID, str(app)],
        check=True,
        capture_output=True,
    )
    # So Login Items and the permission prompts find the name and icon.
    if Path(LSREGISTER).exists():
        subprocess.run([LSREGISTER, "-f", str(app)], capture_output=True)
    return app


def remove(s: Settings) -> None:
    app = app_path(s)
    if app.exists():
        if Path(LSREGISTER).exists():
            subprocess.run([LSREGISTER, "-u", str(app)], capture_output=True)
        shutil.rmtree(app)


# Icon: the HUD's rings, drawn with numpy (no imaging library needed)


def icon_pixels(size: int = 1024) -> np.ndarray:
    """RGBA image: dark rounded square, cyan rings, a glowing core."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    dx, dy = (x - c) / size, (y - c) / size
    r = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    aa = 1.5 / size

    def band(radius: float, width: float) -> np.ndarray:
        return np.clip((width / 2 - np.abs(r - radius)) / aa + 0.5, 0, 1)

    # Rounded square (macOS icon grid: ~80% of the canvas, ~22% corner radius).
    half, corner = 0.40, 0.09
    qx, qy = np.abs(dx) - (half - corner), np.abs(dy) - (half - corner)
    dist = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    shape = np.clip((corner - dist) / aa + 0.5, 0, 1)

    bg = (
        np.array([2, 5, 10], np.float32)
        + np.array([8, 20, 32], np.float32) * np.clip(1 - r / 0.45, 0, 1)[..., None]
    )
    cyan = np.array([0, 229, 255], np.float32)
    light = np.zeros_like(r)
    light += band(0.30, 0.008) * 0.55
    segments = (np.mod(angle * 18 / np.pi, 2) < 1.4).astype(np.float32)
    light += band(0.255, 0.024) * segments * 0.9
    arc = ((angle > -2.6) & (angle < -0.4)).astype(np.float32)
    light += band(0.205, 0.016) * arc
    light += band(0.165, 0.006) * 0.6
    light += np.exp(-((r / 0.075) ** 2)) * 1.1 + np.exp(-((r / 0.16) ** 2)) * 0.25
    light = np.clip(light, 0, 1.4)

    rgb = bg + cyan * np.clip(light, 0, 1)[..., None]
    rgb += 255 * np.clip(light - 1, 0, 1)[..., None]  # white-hot centre
    rgba = np.concatenate([np.clip(rgb, 0, 255), shape[..., None] * 255], axis=-1)
    return rgba.astype(np.uint8)


def png_bytes(pixels: np.ndarray) -> bytes:
    h, w, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def write_icon(path: Path) -> None:
    """An .icns from an iconset drawn at each size (`sips` fails to write .icns on macOS 26)."""
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for points in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                suffix = "@2x" if scale == 2 else ""
                png = png_bytes(icon_pixels(points * scale))
                (iconset / f"icon_{points}x{points}{suffix}.png").write_bytes(png)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(path)],
            check=True,
            capture_output=True,
        )
