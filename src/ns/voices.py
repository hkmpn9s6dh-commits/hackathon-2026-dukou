"""声部：每一样乐器都是那块钢板的一次变形。

不许出现外部采样。每个函数第一个参数都是 IMP（钢板的脉冲），
`render.py` 会把调用记录成一条来源账，验收时断言全部指向同一份脉冲。
"""

from __future__ import annotations

import numpy as np

from . import dsp
from .source import steel_impulse

FS = dsp.FS
SRC_TAG = "steel_impulse@seed20261201"

IMP = steel_impulse()


# ------------------------------------------------------------------ 工具

def _env(n: int, attack_s: float, release_s: float, sustain: float = 1.0) -> np.ndarray:
    """梯形包络（线性），两端各留一点余弦以避免咔哒。"""
    a = max(1, int(attack_s * FS))
    r = max(1, int(release_s * FS))
    a = min(a, n)
    r = min(r, max(1, n - a))
    e = np.ones(n) * sustain
    e[:a] = np.linspace(0, 1, a, endpoint=False) ** 1.4
    e[n - r :] *= np.linspace(1, 0, r) ** 1.6
    return e


def _place(buf: np.ndarray, x: np.ndarray, at_s: float, gain: float = 1.0) -> None:
    i = int(at_s * FS)
    if i >= len(buf):
        return
    n = min(len(x), len(buf) - i)
    buf[i : i + n] += gain * x[:n]


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


# ------------------------------------------------------------------ 钟 / 锣

def gong(midi: float, seconds: float, vel: float = 1.0) -> np.ndarray:
    """同一块板，换基频再敲一次。

    不用重采样：重采样会把音高和时长绑死，高音钟会短到听不见。
    这里改 `base_hz`，模态比例与敲击点都不变——还是那块板，
    只是"这一块"被换成了另一个尺寸。板越小，响得越短。
    """
    f = _hz(midi)
    scale = (82.4069 / f) ** 0.30
    y = steel_impulse(seconds=seconds, strength=0.30 + 0.70 * vel,
                      base_hz=f, decay_scale=scale)
    return y[: int(seconds * FS)] * vel


def bell_buoy(midi: float, seconds: float, vel: float = 0.8) -> np.ndarray:
    """浮标钟：在钢板的高阶分音上取一个更"筒状"的音，加一点拍频。"""
    base = gong(midi, seconds, vel)
    n = len(base)
    t = np.arange(n) / FS
    wob = 1.0 + 0.004 * np.sin(2 * np.pi * 0.37 * t)
    return base * wob


# ------------------------------------------------------------------ 水

def water_stereo(seconds: float, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """水：同一份源，右声道用不同滤波与微小延迟——单声道折叠不吃亏。"""
    l = water_bed(seconds, seed=seed)
    r = dsp.lowpass(np.roll(l, int(0.006 * FS)), 7000, 2)
    r = dsp.highpass(r, 180, 2) * 0.9 + dsp.highpass(l, 900, 2) * 0.35
    return l, r


def water_bed(seconds: float, seed: int = 11) -> np.ndarray:
    """水声：钢板噪声尾被颗粒化，外加一层缓慢起伏的滤波噪声。"""
    rng = np.random.default_rng(seed)
    # 取脉冲的噪声残差（高频部分），它本身就是"敲击的空气声"
    noise = IMP - dsp.lowpass(IMP, 900, order=4)
    n = int(seconds * FS)
    out = np.zeros(n)

    # 底噪：极缓慢起伏的宽带
    w = rng.normal(0, 1, n)
    w = dsp.highpass(w, 220, 2)
    w = dsp.lowpass(w, 5200, 2)
    lfo = 0.55 + 0.45 * np.sin(2 * np.pi * 0.071 * np.arange(n) / FS + 0.4)
    lfo *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.023 * np.arange(n) / FS + 1.9)
    out += w * lfo * 0.055

    # 颗粒：短促的"啪嗒"
    n_grain = int(0.020 * FS)
    pos = 0
    while pos < n - n_grain:
        i = int(rng.uniform(0, max(1, len(noise) - n_grain * 4)))
        seg = noise[i : i + n_grain].copy()
        seg *= np.exp(-np.arange(n_grain) / (0.004 * FS))
        r = float(rng.uniform(0.75, 1.6))
        seg = dsp.resample(seg, r)
        g = float(rng.uniform(0.06, 0.32))
        _place(out, seg, pos / FS, g)
        pos += int(rng.exponential(0.085) * FS) + n_grain // 3
    return out


def lapping(seconds: float, seed: int = 12) -> np.ndarray:
    """船舷的拍水：低频冲击 + 宽带溅声，每 2.3 秒左右一次。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    out = np.zeros(n)
    t = 0.4
    while t < seconds - 1.0:
        body = dsp.lowpass(IMP[: int(0.55 * FS)], 260, 2) * np.exp(
            -np.arange(int(0.55 * FS)) / (0.09 * FS)
        )
        _place(out, body, t, float(rng.uniform(0.25, 0.6)))
        splash = dsp.highpass(IMP[: int(0.30 * FS)], 1500, 2) * np.exp(
            -np.arange(int(0.30 * FS)) / (0.035 * FS)
        )
        _place(out, splash, t + 0.02, float(rng.uniform(0.10, 0.28)))
        t += float(rng.uniform(1.7, 3.1))
    return out


# ------------------------------------------------------------------ 引擎

def _loop_crossfade(x: np.ndarray, n: int, xfade_s: float = 0.25) -> np.ndarray:
    """把 x 循环铺满 n 个样本，接缝处做余弦交叉淡化，避免周期性的咔哒。"""
    if len(x) >= n:
        return x[:n]
    xf = max(1, int(xfade_s * FS))
    out = np.zeros(n + xf)
    pos = 0
    ramp = np.linspace(0, 1, xf) ** 0.5
    while pos < n:
        seg = x.copy()
        if pos > 0:
            seg = seg.copy()
            seg[: min(xf, len(seg))] *= ramp[: min(xf, len(seg))]
        end = min(pos + len(seg), n + xf)
        out[pos:end] += seg[: end - pos]
        pos += len(seg) - xf
    return out[:n]


def engine(seconds: float, rpm_hz: float = 7.6, level: float = 1.0,
           seed: int = 13) -> np.ndarray:
    """柴油机：钢板低通后压到 20 Hz 附近，再按点火率调幅。"""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    body = dsp.lowpass(IMP, 140, 4)
    body = dsp.resample(body, 0.78)                     # 一个 3.3 s 的"冲程"循环
    body = _loop_crossfade(body, n)

    # 点火：每转一次，带一点不匀（机械磨损）
    jitter = 1.0 + 0.012 * np.sin(2 * np.pi * 0.19 * t) + 0.006 * np.sin(2 * np.pi * 0.07 * t)
    phase = 2 * np.pi * rpm_hz * np.cumsum(jitter) / FS
    fire = (np.sin(phase) * 0.5 + 0.5) ** 6              # 窄脉冲串
    fire = dsp.lowpass(fire, 900, 2)

    y = body * (0.35 + 0.65 * fire)
    y += 0.18 * dsp.lowpass(np.sin(phase * 0.5) * fire, 300, 2)  # 半速的"突"
    return y * level


def hull_knock(vel: float = 1.0) -> np.ndarray:
    """船体的闷响——全曲的拍点。"""
    n = int(0.42 * FS)
    y = dsp.lowpass(IMP[:n], 95, 4)
    y = y * np.exp(-np.arange(n) / (0.055 * FS))
    return dsp.soft_clip(y * 2.2 * vel, drive=1.1, ceil=1.0)


# ------------------------------------------------------------------ 弦垫

def _plate_amp(freqs: np.ndarray) -> np.ndarray:
    """把钢板谱当成一条"音色曲线"来查：给定频率，返回该处的相对幅度。"""
    from .source import impulse_spectrum
    f_grid, mag = impulse_spectrum(IMP)
    mag = mag / max(np.max(mag), 1e-9)
    out = np.interp(freqs, f_grid, mag, left=0.0, right=0.0)
    return out ** 0.7


def pad(notes: list[float], seconds: float, seed: int = 21,
        bright: float = 0.5) -> np.ndarray:
    """弦垫：把钢板谱"冻结"成一条谐波列，再加合唱式的微失谐。

    做法与箫同一个原理——板的模态谱被拉成整数倍谐波，
    所以垫子的音色就是这块板的音色。用频域直接放分音再 IFFT，
    比逐个正弦快两个数量级，相位也天然随机。
    """
    n = int(seconds * FS)
    N = 1 << (n - 1).bit_length()
    rng = np.random.default_rng(seed)
    spec = np.zeros(N // 2 + 1, dtype=complex)
    df = FS / N
    f_max = 900.0 + 5200.0 * bright

    for i, m in enumerate(notes):
        f0 = _hz(m)
        weight = 1.0 / (i + 1.4) ** 0.25
        k = 1
        while f0 * k < min(f_max, 12000):
            amp = float(_plate_amp(np.array([f0 * k]))[0]) * (k ** -1.02) * weight
            if amp > 1e-5:
                # 三个微失谐的副本，叠出合唱的厚度
                for det in (-0.45, 0.0, 0.55):
                    idx = int(round((f0 * k + det) / df))
                    if 0 < idx < len(spec):
                        spec[idx] += amp * np.exp(1j * rng.uniform(0, 2 * np.pi))
            k += 1

    y = np.fft.irfft(spec, N)[:n]
    y = y / max(np.max(np.abs(y)), 1e-9)

    # 运弓的呼吸：两个极慢的起伏，加一点微颤
    t = np.arange(n) / FS
    breathe = 1.0 + 0.14 * np.sin(2 * np.pi * 0.061 * t + 0.7)                   + 0.08 * np.sin(2 * np.pi * 0.023 * t + 2.1)
    y = y * breathe
    # 起势与收势的柔化，避免方块感
    edge = int(0.35 * FS)
    if n > 2 * edge:
        y[:edge] *= np.linspace(0, 1, edge) ** 0.6
        y[-edge:] *= np.linspace(1, 0, edge) ** 0.6
    return y * 1.6


def air(seconds: float, seed: int = 22) -> np.ndarray:
    """空气层：钢板极高频的残渣，拉长，只做空间用。"""
    n = int(seconds * FS)
    hf = IMP - dsp.lowpass(IMP, 4200, 4)
    y = dsp.time_stretch_grain(hf, factor=max(1.0, (seconds * 1.05) / (len(IMP) / FS)),
                               grain_s=0.22, hop_s=0.06, seed=seed)
    return _loop_crossfade(y, n)


# ------------------------------------------------------------------ 箫 / 簧

def reed(midi: float, seconds: float, vel: float = 0.8, seed: int = 31,
         vibrato: float = 1.0) -> np.ndarray:
    """箫音：把钢板的模态谱"冻结"成一条谐波列。

    具体做法：对第 k 次谐波，去钢板谱里找离 k*f 最近的那个模态，
    取它的幅度。于是这个音的音色就是这块板的音色，只是被拉成了整数倍。
    """
    from .source import impulse_spectrum

    f0 = _hz(midi)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    f_grid, mag = impulse_spectrum(IMP)
    mag = mag / max(np.max(mag), 1e-9)

    y = np.zeros(n)
    k = 1
    while f0 * k < FS / 2 * 0.8:
        target = f0 * k
        idx = np.argmin(np.abs(f_grid - target))
        a = float(mag[idx]) ** 0.75
        if a < 0.004 and k > 3:
            k += 1
            continue
        # 高次谐波弱一些，接近管乐
        a *= 1.0 / (1.0 + 0.055 * (k - 1))
        # 颤音：频率轻微摆动 + 起音时的上滑
        vib = 1.0 + 0.0045 * vibrato * np.sin(2 * np.pi * 5.1 * t + 0.3 * k)
        scoop = 1.0 + 0.030 * np.exp(-t / 0.10)
        ph = 2 * np.pi * np.cumsum(target * vib * scoop) / FS
        y += a * np.sin(ph)
        k += 1

    # 气声：钢板的高频残差，跟着包络走
    breath = IMP - dsp.lowpass(IMP, 2600, 4)
    breath = dsp.time_stretch_grain(breath, factor=max(1.0, (seconds * 1.1) / (len(IMP) / FS)),
                                    grain_s=0.09, hop_s=0.03, seed=seed)
    breath = _loop_crossfade(breath, n)
    env = _env(n, 0.09, 0.28)
    y = y * env + breath * env * 0.05

    # 管体的共振：在 1.1 kHz 附近加一点
    y = dsp.peak_eq(y, 1150, 3.0, q=1.6)
    return y * vel / max(np.max(np.abs(y)), 1e-9)


# ------------------------------------------------------------------ 房间

def room_ir(seconds: float = 3.6, seed: int = 41, size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """混响脉冲：钢板自己的尾巴就是房间。

    拉伸 → 指数衰减 → 谱相位随机化（扩散）→ 拉平频谱倾斜。
    拉平这一步是必须的：不拉平的话钢板谱的低频会把整首歌泡在泥里。
    低频（<300 Hz）左右共用同一个相位，保住单声道兼容。
    """
    n = int(seconds * FS)
    base = dsp.time_stretch_grain(IMP, factor=max(1.0, (seconds * 1.6) / (len(IMP) / FS)),
                                  grain_s=0.25, hop_s=0.06, seed=seed)
    base = _loop_crossfade(base, n)
    base = base / max(np.max(np.abs(base)), 1e-9)

    # 目标频谱：−0.9 dB/oct 的柔和倾斜 + 4 kHz 以上按空气吸收滚降
    f = np.fft.rfftfreq(n, 1 / FS)
    f_safe = np.maximum(f, 20.0)
    tilt = (f_safe / 200.0) ** (-0.45)
    air_abs = 1.0 / (1.0 + (f / 5200.0) ** 2)
    target = tilt * air_abs

    t = np.arange(n) / FS
    decay = (0.72 * np.exp(-t / (0.42 * size)) + 0.28 * np.exp(-t / (1.9 * size)))
    decay *= np.minimum(1.0, t / 0.008)

    outs = []
    for ch, s_ in enumerate((seed, seed + 977)):
        Y = np.fft.rfft(base)
        phase_rand = np.random.default_rng(s_).uniform(0, 2 * np.pi, len(Y))
        phase = np.angle(Y)
        # 300 Hz 以下保留原相位（左右相关），以上随机
        mask = np.clip((f - 150.0) / 150.0, 0.0, 1.0)
        ph = (1 - mask) * phase + mask * phase_rand
        Y2 = target * np.exp(1j * ph)
        y = np.fft.irfft(Y2, n) * decay
        outs.append(y / max(np.sqrt(np.sum(y ** 2)), 1e-9))
    return outs[0], outs[1]


def apply_room(x: np.ndarray, wet: float = 0.35, size: float = 1.0) -> np.ndarray:
    """x: (n,) 或 (n,2)。返回加了混响的立体声 (n,2)。"""
    y = x if x.ndim == 2 else np.stack([x, x], axis=1)
    ir_l, ir_r = room_ir(size=size)
    w = dsp.fft_conv_st(y, ir_l, ir_r)
    # 混响的低频并回单声道，保住单声道兼容
    low_w = dsp.lowpass(w, 220, 2)
    mono_low = (low_w[:, 0] + low_w[:, 1]) * 0.5
    w = w - low_w + np.stack([mono_low, mono_low], axis=1)
    return y * (1 - wet) + w * wet
