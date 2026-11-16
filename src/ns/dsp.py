"""基础 DSP 与计量。

这里只放与"音乐内容"无关的东西：滤波器、卷积、重采样、限幅，
以及 ITU-R BS.1770-4 / EBU R128 的响度计量。

计量部分的写法刻意不依赖任何音频库（只 numpy + scipy.signal），
以便与 ffmpeg 的 ebur128 交叉核对时是两条独立实现。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

FS = 48_000


# ---------------------------------------------------------------- 滤波器

def biquad(x: np.ndarray, b, a) -> np.ndarray:
    """单次二阶节。x 可以是 (n,) 或 (n, ch)。"""
    return signal.lfilter(b, a, x, axis=0)


def k_weighting(x: np.ndarray) -> np.ndarray:
    """BS.1770-4 的两级 K 加权（系数为 48 kHz 标准值）。"""
    # Stage 1：高架（模拟头部声学）
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    # Stage 2：RLB 高通
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return biquad(biquad(x, b1, a1), b2, a2)


def highpass(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, fc / (FS / 2), btype="highpass", output="sos")
    return signal.sosfilt(sos, x, axis=0)


def lowpass(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, min(fc, FS / 2 * 0.98) / (FS / 2), btype="lowpass", output="sos")
    return signal.sosfilt(sos, x, axis=0)


def peak_eq(x: np.ndarray, fc: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """RBJ 峰值均衡。"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / FS
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return biquad(x, np.array(b) / a[0], np.array(a) / a[0])


def shelf(x: np.ndarray, fc: float, gain_db: float, high: bool = True) -> np.ndarray:
    """一阶搁架（RBJ 简化式），用于整体的明暗微调。"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / FS
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / np.sqrt(2)
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return biquad(x, np.array(b) / a[0], np.array(a) / a[0])


def comb(x: np.ndarray, f0: float, decay: float = 0.72, n_harm: int = 24) -> np.ndarray:
    """谐波梳状共振：把一个宽带信号变成有明确音高的东西。

    用一串短延迟加反馈的近似——这里直接做频域上的谐波增强，
    等价于把 x 在每个 k*f0 上做窄带提升，再多加一点原始信号。
    """
    out = x.copy()
    for k in range(1, n_harm + 1):
        f = f0 * k
        if f > FS / 2 * 0.45:
            break
        out = out + (decay ** (k - 1)) * peak_eq(x, f, 11.0, q=max(5.0, f / 50.0))
    return out


# ---------------------------------------------------------------- 卷积 / 重采样

def fft_conv(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """单声道 FFT 卷积，返回与 x 等长的结果（尾部截断）。"""
    from scipy.signal import oaconvolve
    y = oaconvolve(x, ir, mode="full")[: len(x)]
    return y


def fft_conv_st(x: np.ndarray, ir_l: np.ndarray, ir_r: np.ndarray) -> np.ndarray:
    """立体声：左右各用一条 IR。x 为 (n, 2)。"""
    out = np.empty_like(x)
    out[:, 0] = fft_conv(x[:, 0], ir_l)
    out[:, 1] = fft_conv(x[:, 1], ir_r)
    return out


def resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """按比例重采样（ratio > 1 = 变长/降调）。用多相滤波，抗混叠。"""
    if abs(ratio - 1.0) < 1e-9:
        return x
    # 用有理近似，分母上限 1000
    from fractions import Fraction
    fr = Fraction(ratio).limit_denominator(1000)
    y = signal.resample_poly(x, fr.numerator, fr.denominator, axis=0)
    return y


def time_stretch_grain(
    x: np.ndarray, factor: float, grain_s: float = 0.12, hop_s: float = 0.03, seed: int = 7
) -> np.ndarray:
    """颗粒化时间拉伸：适合把一段瞬态拉成一片持续音。

    factor > 1 变长。grain_s 是颗粒长度，hop_s 是输出步进。
    """
    rng = np.random.default_rng(seed)
    n_in = len(x)
    n_out = int(n_in * factor)
    n_grain = int(min(grain_s * FS, n_in))
    hop = int(hop_s * FS)
    win = np.hanning(n_grain)
    out = np.zeros(n_out + n_grain)
    pos = 0
    src = 0.0
    while pos < n_out:
        i = int(src + rng.normal(0, n_grain * 0.05))
        i = max(0, min(n_in - n_grain, i))
        seg = x[i : i + n_grain] * win
        out[pos : pos + n_grain] += seg
        pos += hop
        src += hop / factor
        if src > n_in - n_grain:
            src = max(0, n_in - n_grain)
            if rng.random() < 0.4:
                src = rng.uniform(0, max(0, n_in - n_grain))
    return out[:n_out]


# ---------------------------------------------------------------- 动态

def soft_clip(x: np.ndarray, drive: float = 1.0, ceil: float = 1.0) -> np.ndarray:
    """tanh 软削波，保留电平但压掉尖峰。"""
    return ceil * np.tanh(drive * x / ceil) / np.tanh(drive)


def _peak_env(x: np.ndarray, oversample: int = 4, block: int = 1 << 20) -> np.ndarray:
    """每个基率样本点上的真峰估计：分块过采样，避免一次吃掉几 GB。

    返回 (n,) 的线性峰值（>= 样本峰）。
    """
    y = x if x.ndim == 2 else x[:, None]
    n = len(y)
    out = np.empty(n)
    step = block - 8
    for s in range(0, n, step):
        e = min(n, s + block)
        seg = y[s:e]
        up = signal.resample_poly(seg, oversample, 1, axis=0)
        m = np.max(np.abs(up), axis=1)
        # resample_poly 的群延迟约为 (taps-1)/2；用等长裁剪近似对齐
        m = m[: len(seg) * oversample]
        m = np.pad(m, (0, max(0, len(seg) * oversample - len(m))), mode="edge")
        out[s:e] = m.reshape(len(seg), oversample).max(axis=1)
        if e >= n:
            break
    return out


def limiter(x: np.ndarray, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
            release_ms: float = 120.0, oversample: int = 4) -> np.ndarray:
    """前瞻峰值限幅（全向量化）。

    1) 分块过采样求真峰包络；2) 前视取未来最大值；3) 取目标增益；
    4) 用滑动最小值实现"保持 release 毫秒不回升"；5) 短滑动平均抹平台阶。
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

    ceil = 10 ** (ceiling_db / 20)
    y = x if x.ndim == 2 else x[:, None]
    n = len(y)
    la = max(1, int(lookahead_ms * 1e-3 * FS))

    env = _peak_env(y, oversample=oversample)
    fut = maximum_filter1d(env, size=la * 2 + 1, mode="nearest")
    target = np.minimum(1.0, ceil / np.maximum(fut, 1e-9))

    rel = max(1, int(release_ms * 1e-3 * FS))
    g = minimum_filter1d(target, size=rel, mode="nearest")
    g = uniform_filter1d(g, size=max(1, la // 2), mode="nearest")
    g = np.minimum(g, target)          # 平滑后不许比目标更松
    return y * g[:, None]


# ---------------------------------------------------------------- 计量

BLOCK_S = 0.400
STEP_S = 0.100


def _block_ms(x: np.ndarray, block_s: float, step_s: float) -> np.ndarray:
    """返回 (blocks, ch) 的均方值。窗口维度在最后，必须对 axis=2 求均值。"""
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    blk = int(block_s * FS)
    stp = int(step_s * FS)
    if n < blk:
        return np.mean(x ** 2, axis=0, keepdims=True)
    frames = np.lib.stride_tricks.sliding_window_view(x, blk, axis=0)[::stp]  # (b, ch, blk)
    return np.mean(frames ** 2, axis=2)


def integrated_lufs(x: np.ndarray) -> float:
    """BS.1770-4 集成响度（LUFS）。"""
    x = x if x.ndim == 2 else x[:, None]
    kw = k_weighting(x)
    z = _block_ms(kw, BLOCK_S, STEP_S)          # (blocks, ch)
    w = np.ones(z.shape[1])
    lj = -0.691 + 10 * np.log10(np.maximum(np.sum(z * w, axis=1), 1e-12))
    g1 = lj > -70.0
    if not np.any(g1):
        return -np.inf
    lbar = -0.691 + 10 * np.log10(np.mean(np.sum(z[g1] * w, axis=1)))
    g2 = g1 & (lj > lbar - 10.0)
    if not np.any(g2):
        return float(lbar)
    return float(-0.691 + 10 * np.log10(np.mean(np.sum(z[g2] * w, axis=1))))


def true_peak_dbtp(x: np.ndarray, oversample: int = 4) -> float:
    """真峰值（dBTP）。4 倍过采样。"""
    y = x if x.ndim == 2 else x[:, None]
    up = signal.resample_poly(y, oversample, 1, axis=0)
    return float(20 * np.log10(max(np.max(np.abs(up)), 1e-12)))


def lra(x: np.ndarray) -> float:
    """EBU R128 LRA（LU）。3 s 短时窗，66.7% 重叠，绝对门 -70、相对门 -20。"""
    y = x if x.ndim == 2 else x[:, None]
    kw = k_weighting(y)
    blk, stp = 3.0, 1.0
    n = len(kw)
    if n < int(blk * FS):
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(kw, int(blk * FS), axis=0)[:: int(stp * FS)]
    z = np.mean(frames ** 2, axis=1)
    w = np.ones(z.shape[1])
    lj = -0.691 + 10 * np.log10(np.maximum(np.sum(z * w, axis=1), 1e-12))
    g1 = lj > -70.0
    if not np.any(g1):
        return 0.0
    lbar = -0.691 + 10 * np.log10(np.mean(np.sum(z[g1] * w, axis=1)))
    g2 = g1 & (lj > lbar - 20.0)
    if not np.any(g2):
        return 0.0
    v = lj[g2]
    return float(np.percentile(v, 95) - np.percentile(v, 10))


def stereo_correlation(x: np.ndarray) -> float:
    y = x if x.ndim == 2 else np.stack([x, x], axis=1)
    a, b = y[:, 0], y[:, 1]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def mono_fold_excess_db(x: np.ndarray) -> float:
    """单声道折叠后的"额外"损失（dB）。

    立体声折成单声道本来就会掉 3.01 dB（两个声道功率合成一个）。
    这里减掉这 3.01，只看有没有因为相位相消而额外损失；
    绝对值 > 3 dB 说明单声道兼容有问题。
    """
    y = x if x.ndim == 2 else np.stack([x, x], axis=1)
    st = integrated_lufs(y)
    mono = (y[:, 0] + y[:, 1]) * 0.5
    return float(st - 3.0103 - integrated_lufs(mono[:, None]))


def dc_offset(x: np.ndarray) -> float:
    y = x if x.ndim == 2 else x[:, None]
    return float(max(abs(np.mean(y[:, c])) for c in range(y.shape[1])))


def spectral_centroid(x: np.ndarray) -> float:
    y = x if x.ndim == 1 else x.mean(axis=1)
    f, p = signal.welch(y, FS, nperseg=8192)
    p = np.maximum(p, 0)
    return float(np.sum(f * p) / max(np.sum(p), 1e-12))


def low_ratio(x: np.ndarray, split_hz: float = 120.0) -> float:
    """低频（<split_hz）能量占比。"""
    y = x if x.ndim == 1 else x.mean(axis=1)
    f, p = signal.welch(y, FS, nperseg=8192)
    p = np.maximum(p, 0)
    return float(np.sum(p[f < split_hz]) / max(np.sum(p), 1e-12))


def onset_density(x: np.ndarray, thresh_db: float = -30.0) -> float:
    """每秒的显著起音数（谱通量正部超过阈值）。"""
    y = x if x.ndim == 1 else x.mean(axis=1)
    hop, nfft = 512, 2048
    f, t, Z = signal.stft(y, FS, nperseg=nfft, noverlap=nfft - hop)
    mag = np.abs(Z)
    flux = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
    if flux.size == 0:
        return 0.0
    thr = np.percentile(flux, 100 + thresh_db)  # thresh_db=-30 → 70 分位以上算起音
    thr = max(thr, flux.mean() * 1.5)
    return float(np.sum(flux > thr) / (len(y) / FS))


def peak_db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-12)))


# ---------------------------------------------------------------- 频谱匹配

def minphase_ir(mag: np.ndarray, n_fft: int = 16384, tail_ms: float = 60.0) -> np.ndarray:
    """由幅度谱构造最小相位 FIR（倒谱法），没有预振铃。

    mag 长度必须是 n_fft//2+1。
    """
    mag = np.maximum(mag, 1e-6)
    log_mag = np.log(mag)
    cep = np.fft.irfft(log_mag, n_fft)          # 实倒谱
    half = n_fft // 2
    c_min = np.zeros(n_fft)
    c_min[0] = cep[0]
    c_min[1:half] = 2.0 * cep[1:half]
    c_min[half] = cep[half]
    h = np.fft.irfft(np.exp(np.fft.rfft(c_min)), n_fft)
    n_out = int(tail_ms * 1e-3 * FS)
    h = h[:n_out].copy()
    # 尾巴做半汉宁收口，避免截断造成的高频波纹
    taper = int(n_out * 0.4)
    h[-taper:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper)))
    return h


def match_eq(x: np.ndarray, measured_db: np.ndarray, target_db: np.ndarray,
             bands: list[float], max_db: float = 9.0) -> np.ndarray:
    """把长期平均谱往目标曲线上拉（有上限、平滑过的）。

    measured_db / target_db 都是相对 1 kHz 的八度带电平。
    """
    from scipy.interpolate import PchipInterpolator

    corr = np.clip(target_db - measured_db, -max_db, max_db)
    freqs = np.log2(np.array(bands, dtype=float))
    n_fft = 16384
    f = np.fft.rfftfreq(n_fft, 1 / FS)
    lf = np.log2(np.maximum(f, 20.0))
    interp = PchipInterpolator(freqs, corr, extrapolate=True)
    curve = interp(np.clip(lf, freqs[0] - 1.0, freqs[-1] + 1.0))
    # 两端收回到 0 dB，免得把次声与超声抬起来
    curve[lf < freqs[0] - 1.0] = 0.0
    curve[lf > freqs[-1] + 1.0] = 0.0
    mag = 10 ** (curve / 20.0)
    h = minphase_ir(mag, n_fft)
    y = x if x.ndim == 2 else x[:, None]
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        out[:, c] = oaconv(y[:, c], h)[: len(y)]
    return out


from scipy.signal import oaconvolve as oaconv  # noqa: E402
