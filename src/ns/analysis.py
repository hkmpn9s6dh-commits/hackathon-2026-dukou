"""频谱与动态的分析：给"混得好不好"一个能算的口径。

八度带电平相对于 1 kHz。目标曲线 TARGET_BANDS 写在下面，
它是"一首电影感氛围曲在流媒体上应该长什么样"的口径，
不是行业标准——但它是**可判定**的，这就够了。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import dsp

FS = dsp.FS

BANDS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

# 目标：以 1 kHz 为 0，向下小幅抬、向上缓降。
# 参照的是电影配乐/氛围类的常见长期平均谱（低频有托、3 kHz 之后别硬）。
TARGET_BANDS = np.array([-7.0, -3.0, -0.5, 1.0, 1.5, 0.0, -1.5, -4.5, -9.0, -16.0])


def octave_bands(x: np.ndarray, bands: list[float] | None = None) -> np.ndarray:
    """各八度带的能量（dB），相对 1 kHz。"""
    bands = bands or BANDS
    y = x if x.ndim == 1 else x.mean(axis=1)
    f, p = signal.welch(y, FS, nperseg=16384)
    p = np.maximum(p, 1e-20)
    out = []
    for fc in bands:
        lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(max(np.sum(p[m]), 1e-20)))
    out = np.array(out)
    ref = out[bands.index(1000.0)]
    return out - ref


def band_error(x: np.ndarray) -> np.ndarray:
    return octave_bands(x) - TARGET_BANDS


def section_lufs(x: np.ndarray, edges: list[float]) -> list[float]:
    """按给定时间点切段，量每段的集成响度。"""
    out = []
    for a, b in zip(edges, edges[1:]):
        seg = x[int(a * FS) : int(b * FS)]
        if len(seg) < FS:
            out.append(float("nan"))
            continue
        out.append(round(dsp.integrated_lufs(seg), 2))
    return out


def short_term(x: np.ndarray, window_s: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """短时响度曲线（LUFS），用于画动态曲线。"""
    y = x if x.ndim == 2 else x[:, None]
    kw = dsp.k_weighting(y)
    n = len(kw)
    w = int(window_s * FS)
    hop = int(1.0 * FS)
    ts, vs = [], []
    for s in range(0, max(1, n - w), hop):
        seg = kw[s : s + w]
        ms = np.mean(seg ** 2, axis=0)
        lj = -0.691 + 10 * np.log10(max(np.sum(ms), 1e-12))
        ts.append(s / FS)
        vs.append(lj)
    return np.array(ts), np.array(vs)
