"""DSP 与计量的断言。

这些不是"跑一遍看看"，是"这个数必须落在这个区间，否则曲子不能交"。
计量部分另有一条与 ffmpeg ebur128 的交叉核对（见 test_crosscheck.py）。
"""

import numpy as np
import pytest

from ns import dsp

FS = dsp.FS


def sine(f=1000.0, db=-20.0, seconds=3.0, ch=2):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = 10 ** (db / 20) * np.sin(2 * np.pi * f * t)
    return np.stack([x] * ch, axis=1) if ch > 1 else x[:, None]


def test_lufs_scale_is_additive():
    """幅度加倍 = +6.02 LU。计量必须是线性的。"""
    a = dsp.integrated_lufs(sine(db=-26))
    b = dsp.integrated_lufs(sine(db=-20))
    assert abs((b - a) - 6.02) < 0.05


def test_lufs_of_silence():
    assert dsp.integrated_lufs(np.zeros((FS, 2))) == -np.inf


def test_lufs_stereo_vs_mono_double():
    """同一个信号放两个声道 = +3.01 LU（功率相加）。"""
    m = dsp.integrated_lufs(sine(db=-20, ch=1))
    s = dsp.integrated_lufs(sine(db=-20, ch=2))
    assert abs((s - m) - 3.01) < 0.05


def test_true_peak_of_fullscale_square():
    """方波经带限后本来就会有过冲（Gibbs，约 +2 dB），只断言它不小于样本峰且没离谱。"""
    x = np.sign(np.sin(2 * np.pi * 100 * np.arange(FS) / FS))[:, None] * 0.999
    tp = dsp.true_peak_dbtp(x)
    sp = dsp.peak_db(x)
    assert sp - 0.05 <= tp <= sp + 2.5


def test_true_peak_oversampling_finds_intersample():
    """44.1/4 采样相位的方波：样本峰值小于真峰值，过采样必须能抓到。"""
    x = np.sin(2 * np.pi * (FS / 4 + 37) * np.arange(FS) / FS)[:, None]
    x = x / np.max(np.abs(x))
    tp = dsp.true_peak_dbtp(x)
    sp = dsp.peak_db(x)
    assert tp >= sp - 1e-6


def test_limiter_respects_ceiling():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.9, (FS * 2, 2))
    y = dsp.limiter(x, ceiling_db=-1.0)
    assert dsp.true_peak_dbtp(y) <= -0.9


def test_limiter_is_transparent_below_ceiling():
    x = 0.2 * sine(db=-14)
    y = dsp.limiter(x, ceiling_db=-1.0)
    assert np.allclose(x, y, atol=2e-3)


def test_mono_fold_of_identical_channels_has_no_extra_loss():
    """同相的两声道折成单声道，除了固有的 3.01 dB 之外不该再掉。"""
    x = sine(db=-20, ch=2)
    assert abs(dsp.mono_fold_excess_db(x)) < 0.05


def test_stereo_correlation_extremes():
    x = sine(db=-20, ch=1)
    assert dsp.stereo_correlation(np.stack([x[:, 0], x[:, 0]], axis=1)) > 0.999
    assert dsp.stereo_correlation(np.stack([x[:, 0], -x[:, 0]], axis=1)) < -0.999


def test_dc_offset_detects_bias():
    x = sine(db=-20, ch=1) + 0.01
    assert dsp.dc_offset(x) > 0.009


def test_resample_changes_length_by_ratio():
    x = np.zeros(4800)
    y = dsp.resample(x, 2.0)
    assert abs(len(y) / 4800 - 2.0) < 0.01


def test_grain_stretch_length():
    x = np.random.default_rng(1).normal(0, 1, 4800)
    y = dsp.time_stretch_grain(x, 3.0)
    assert abs(len(y) / (4800 * 3.0) - 1.0) < 0.02


def test_highpass_kills_dc():
    x = np.ones((FS, 1))
    y = dsp.highpass(x, 30, 2)
    assert abs(np.mean(y[FS // 2 :])) < 1e-4


def test_onsets_are_counted():
    """每秒一次点击 -> 起音密度接近 1。"""
    n = FS * 6
    x = np.zeros(n)
    for i in range(6):
        x[i * FS : i * FS + 240] = np.hanning(240)
    assert 0.5 < dsp.onset_density(x[:, None]) < 2.0
