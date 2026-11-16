"""聆听表：把一段音频画成评委能读的一张图。

频谱图 + 波形 + 短时响度，三段共用一条时间轴，段落用虚线分开。
这不是给耳朵听的东西，是给不看/听不到的人的替代物——报告里会写清楚这一点。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import analysis, dsp

# 表上的标签用英文：这张图是给不看中文的评委看的，也免了跨机字体依赖
SECTIONS = [(0.0, 60.0, "I  before light"), (60.0, 120.0, "II  first grey"),
            (120.0, 180.0, "III  sunrise"), (180.0, 252.0, "IV  far shore")]

def sheet(x: np.ndarray, path: Path, title: str = "") -> None:
    """一张图里放三样：频谱图、波形（带段落分隔）、短时响度。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy import signal

    mono = x.mean(axis=1)
    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 1.0, 1.0], hspace=0.18)

    ax1 = fig.add_subplot(gs[0])
    f, t, S = signal.spectrogram(mono, dsp.FS, nperseg=4096, noverlap=3072, mode="magnitude")
    Sdb = 20 * np.log10(np.maximum(S, 1e-9))
    vmax = Sdb.max()
    im = ax1.pcolormesh(t, f, Sdb, shading="auto", cmap="magma",
                        vmin=vmax - 78, vmax=vmax)
    ax1.set_yscale("symlog", linthresh=200)
    ax1.set_ylim(30, 18000)
    ax1.set_ylabel("Hz")
    ax1.set_title(title or "spectrogram (log frequency)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.02, label="dB")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    tt = np.arange(len(mono)) / dsp.FS
    step = 240
    m = (len(mono) // step) * step
    env = np.max(np.abs(mono[:m]).reshape(-1, step), axis=1)
    tt2 = np.arange(len(env)) * step / dsp.FS
    ax2.fill_between(tt2, -env * 100, env * 100, color="#c9c9c9", lw=0)
    ax2.plot(tt2, env * 100, color="#555", lw=0.6)
    ax2.set_ylim(-105, 105)
    ax2.set_ylabel("peak %")
    ax2.set_title("waveform", loc="left", fontsize=10)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ts, vs = analysis.short_term(x)
    ax3.plot(ts, vs, color="#b3402f", lw=1.2)
    ax3.set_ylim(-42, -6)
    ax3.set_ylabel("LUFS (3 s)")
    ax3.set_xlabel("seconds")
    ax3.set_title("short-term loudness", loc="left", fontsize=10)

    for ax in (ax1, ax2, ax3):
        for a, _, lbl in SECTIONS:
            ax.axvline(a, color="w" if ax is ax1 else "#999", lw=0.9,
                       ls="--" if ax is not ax1 else "-", alpha=0.7)
        for a, b, lbl in SECTIONS:
            ax.text(a + 1.0, ax.get_ylim()[1] * 0.92 if ax is not ax1 else 15000,
                    lbl, color="w" if ax is ax1 else "#333", fontsize=9, va="top")
    ax1.set_xlim(0, len(mono) / dsp.FS)

    fig.suptitle("Crossing at First Light — listening sheet  (4:12, Cinematic)",
                 x=0.01, ha="left", fontsize=13)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


