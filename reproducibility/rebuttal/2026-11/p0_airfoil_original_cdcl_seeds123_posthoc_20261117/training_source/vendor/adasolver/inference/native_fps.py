"""Small compiled FPS loop; retains NumPy's float64 distances and tie breaking."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np

FLAGS = ("-std=c99", "-O3", "-fPIC", "-shared", "-ffp-contract=off")


@lru_cache(maxsize=1)
def load_native_fps():
    source = Path(__file__).with_name("_fps_native.c")
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError("native FPS requires a C compiler; select the numpy backend if unavailable")
    identity = hashlib.sha256(source.read_bytes() + repr((FLAGS, platform.machine())).encode()).hexdigest()[:20]
    root = Path(os.environ.get("ADASOLVER_NATIVE_CACHE", str(source.parents[2]/"temp/native_fps")))
    root.mkdir(parents=True, exist_ok=True)
    library = root/f"fps_{identity}.so"
    if not library.exists():
        temporary = root/f".fps_{identity}_{os.getpid()}.so"
        result = subprocess.run([compiler, *FLAGS, str(source), "-o", str(temporary), "-lm"],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"native FPS compilation failed: {result.stderr}")
        os.replace(temporary, library)
    dll = ctypes.CDLL(str(library.resolve()))
    function = dll.adasolver_fps
    function.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
                        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")]
    function.restype = ctypes.c_int
    return function


def native_fps(points, nearest, take, *, first_if_infinite=0):
    points = np.asarray(points)
    if points.dtype != np.float64 or points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("native FPS requires finite float64 point coordinates")
    if not 0 <= take <= len(points):
        raise ValueError("invalid FPS point count")
    distances = np.array(nearest, dtype=np.float64, order="C", copy=True)
    if distances.shape != (len(points),) or np.isnan(distances).any() or np.any(distances < 0):
        raise ValueError("invalid initial FPS distances")
    selected = np.empty(take, dtype=np.int64)
    code = load_native_fps()(np.ascontiguousarray(points), distances, len(points),
                             points.shape[1], take, first_if_infinite, selected)
    if code:
        raise ValueError(f"native FPS rejected its inputs (status {code})")
    return selected
