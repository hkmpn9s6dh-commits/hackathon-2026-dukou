"""单源：一块被敲响的钢板。

整首《渡口》里所有声音都由这一个函数产出的脉冲派生。
没有采样、没有外部素材、没有预置音色。

脉冲 = 模态合成（36 个非谐分音）+ 3 ms 激励噪声。
模态频率取自矩形板的近似 `f ∝ (m/a)² + (n/b)²`，
衰减时间随频率下降（高频先走），这是金属声的关键。
"""

from __future__ import annotations

import numpy as np

from . import dsp

FS = dsp.FS

# 板材比例（长宽比），决定非谐程度。1.37 ≈ 一块偏长的钢板。
PLATE_RATIO = 1.37
BASE_HZ = 82.4069  # E2


def mode_table(n_max: int = 6) -> list[tuple[float, float, float]]:
    """返回 [(ratio, amp, t60_weight), ...]，ratio 相对最低阶。"""
    raw = []
    for m in range(1, n_max + 1):
        for n in range(1, n_max + 1):
            f = (m / PLATE_RATIO) ** 2 + (n / 1.0) ** 2
            amp = 1.0 / (m * n) ** 0.55
            raw.append((f, amp))
    raw.sort()
    f_min = raw[0][0]
    out = []
    for f, amp in raw:
        r = f / f_min
        if r > 42:          # 20 kHz 以上丢掉
            continue
        out.append((r, amp))
    return out


def steel_impulse(seconds: float = 4.2, strength: float = 0.6, seed: int = 20261201,
                  base_hz: float = BASE_HZ, decay_scale: float = 1.0) -> np.ndarray:
    """敲一下钢板。

    strength 0..1：敲得越重，高阶分音相对越强（非线性激励的近似）。
    base_hz：同一块板换成这个基频出声（模态比例不变，还是同一块板）。
    decay_scale：整体衰减倍率；板越小，响得越短。
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    y = np.zeros(n, dtype=np.float64)

    for i, (ratio, amp) in enumerate(mode_table()):
        f = base_hz * ratio
        if f >= FS / 2 * 0.9:
            continue
        # 高阶分音对力度更敏感
        a = amp * (0.08 + 0.92 * strength) ** (1.0 + 0.09 * ratio)
        # 衰减：低频长、高频短；同一阶左右各一，产生轻微拍频
        t60 = (2.0 + 4.0 / (1.0 + 0.30 * ratio)) * decay_scale
        tau = t60 / 6.9078
        det = rng.normal(0, 0.18)
        phase = rng.uniform(0, 2 * np.pi)
        y += a * np.exp(-t / tau) * np.sin(2 * np.pi * (f + det) * t + phase)
        y += 0.55 * a * np.exp(-t / (tau * 0.92)) * np.sin(
            2 * np.pi * (f - det * 1.7) * t + phase * 1.31
        )

    # 3 ms 激励噪声：敲击点的"咔"
    nb = int(0.003 * FS)
    noise = rng.normal(0, 1, nb) * np.exp(-np.arange(nb) / (0.0009 * FS))
    noise = dsp.biquad(noise,
                       [0.2029, 0.0, -0.2029], [1.0, -1.5866, 0.6058])  # ~1.2 kHz 带通
    y[:nb] += noise * 1.6 * strength ** 1.5

    # 轻微软削波，模拟激励瞬间的非线性
    y = dsp.soft_clip(y, drive=1.4, ceil=1.6)
    peak = np.max(np.abs(y))
    return y / peak


def impulse_spectrum(imp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.fft.rfftfreq(len(imp), 1 / FS)
    m = np.abs(np.fft.rfft(imp))
    return f, m
