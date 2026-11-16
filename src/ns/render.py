"""渲染：把乐谱变成一条立体声音频。

七条总线，各自有明确的电平目标，统一送进"那块板做的房间"。
电平不靠耳朵调——每一条总线先量 RMS，再按下面的 trim 表对齐。
"""

from __future__ import annotations

import numpy as np

from . import dsp, score, voices

FS = dsp.FS
N = int(score.TOTAL_S * FS)

# 每条总线的目标 RMS（dBFS，−20 ≈ 满刻度的十分之一）。
# 数值是量完之后定的，不是拍脑袋：见 docs/02-PRODUCTION.md。
LAST_LOG: list[tuple[str, str]] = []   # 每条声部的来源账，build 时写进 build.json

# 三个读法。A/B/C 是给盲排用的，评委不知道哪个是哪个。
# offsets：在基准电平上加多少 dB；mute：整条总线不出现。
VARIANTS = {
    "boat":   {"label": "船",   "offsets": {}, "mute": []},
    "water":  {"label": "水线", "offsets": {"pad": 1.5, "reed": 1.0, "water": 2.0,
                                            "bell": -1.0},
               "mute": ["engine", "knock", "gong"]},
    "bell":   {"label": "钟",   "offsets": {"bell": 3.0, "gong": 1.5, "reed": 1.5,
                                            "engine": -4.0, "knock": -3.0, "pad": -1.5},
               "mute": []},
}

BUS_TARGET_LUFS = {
    "pad": -20.0,
    "bass": -36.0,
    "reed": -22.0,
    "bell": -27.0,
    "gong": -33.0,
    "water": -25.0,
    "engine": -33.0,
    "knock": -37.0,
    "air": -34.0,
}


def _rms_db(x: np.ndarray) -> float:
    r = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return 20 * np.log10(max(r, 1e-12))


def _lufs_of(x: np.ndarray) -> float:
    v = dsp.integrated_lufs(np.stack([x, x], axis=1))
    return v if np.isfinite(v) else -120.0


class Bus:
    """一条立体声总线：左右各一条，加完再统一 trim。"""

    def __init__(self, name: str):
        self.name = name
        self.l = np.zeros(N)
        self.r = np.zeros(N)

    def add(self, x: np.ndarray, at_s: float, gl: float = 1.0, gr: float | None = None):
        if gr is None:
            gr = gl
        i = int(at_s * FS)
        if i >= N or i < 0:
            return
        n = min(len(x), N - i)
        self.l[i : i + n] += gl * x[:n]
        self.r[i : i + n] += gr * x[:n]

    def add_panned(self, x: np.ndarray, at_s: float, pan: float, gain: float = 1.0,
                   haas_ms: float = 0.0):
        th = (pan + 1) * np.pi / 4
        gl = gain * np.cos(th) * 1.41421
        gr = gain * np.sin(th) * 1.41421
        if haas_ms > 0:
            d = int(haas_ms * 1e-3 * FS)
            if pan < 0:
                gr_x = np.concatenate([np.zeros(d), x[: len(x) - d]]) if d < len(x) else x
                self.add(gr_x, at_s, 0.0, gr)
                gl_x = x
                self.add(gl_x, at_s, gl, 0.0)
            else:
                gl_x = np.concatenate([np.zeros(d), x[: len(x) - d]]) if d < len(x) else x
                self.add(gl_x, at_s, gl, 0.0)
                self.add(x, at_s, 0.0, gr)
        else:
            self.add(x, at_s, gl, gr)


def render(verbose: bool = True, variant: str = "boat") -> np.ndarray:
    LAST_LOG.clear()

    def mark(voice: str):
        LAST_LOG.append((voice, voices.SRC_TAG))

    def log(m):
        if verbose:
            print(m, flush=True)

    buses: dict[str, Bus] = {}

    cfg = VARIANTS[variant]
    muted = set(cfg["mute"])

    def bus(name):
        if name not in buses:
            buses[name] = Bus(name)
        return buses[name]

    # ---------------- 水
    log("[1/7] water")
    wet_curve = np.interp(np.arange(N) / FS,
                          [0, 110, 128, 180, 205, 236, 252],
                          [0.9, 1.0, 1.45, 1.1, 0.8, 0.7, 0.15])
    mark("water")
    wl, wr = voices.water_stereo(score.TOTAL_S, seed=1011)
    b = bus("water")
    b.add(wl * wet_curve, 0.0, 1.0, 0.0)
    b.add(wr * wet_curve, 0.0, 0.0, 1.0)
    lap = voices.lapping(score.TOTAL_S, seed=1013)
    b.add(lap, 0.0, 0.9, 0.2)
    b.add(lap, 0.0, 0.2, 0.9)

    # ---------------- 引擎
    log("[2/7] engine")
    eng_curve = np.zeros(N)
    for a, bnd, lv in score.ENGINE:
        i0, i1 = int(a * FS), min(N, int(bnd * FS))
        if i1 <= i0:
            continue
        ramp = np.minimum(1.0, np.linspace(0, 1, i1 - i0) * 1.6) ** 0.5
        eng_curve[i0:i1] = np.maximum(eng_curve[i0:i1], lv * ramp)
    mark("engine")
    eng = voices.engine(score.TOTAL_S, rpm_hz=7.6) * eng_curve
    b = bus("engine")
    b.add(eng, 0.0, 1.0, 1.0)

    # ---------------- 弦垫与低音线
    log("[3/7] pads")
    SPREAD = [-0.46, -0.22, 0.02, 0.26, 0.46]
    for i, (bar, nb, name, bright) in enumerate(score.pad_plan()):
        t0 = score.bt(bar)
        dur = nb * score.BAR
        env = np.ones(int((dur + 1.4) * FS))
        a = int(0.9 * FS)
        r = int(1.6 * FS)
        env[:a] = np.linspace(0, 1, a) ** 0.8
        env[len(env) - r :] *= np.linspace(1, 0, r) ** 0.9
        g = 0.55 + 0.45 * bright
        # 一个音一条声像：和弦被摊开，既立体又不至于相位相消
        for k, note in enumerate(score.CHORDS[name]):
            y = voices.pad([note], dur + 1.4, seed=700 + i * 7 + k, bright=bright) * env
            mark(f"pad:{name}:{note}")
            bus("pad").add_panned(y, t0, SPREAD[k % len(SPREAD)],
                                  g / (k + 1.4) ** 0.25, haas_ms=3.0)

        bn = score.BASS[name]
        mark(f"bass:{name}")
        by = voices.gong(bn, min(dur + 1.2, 7.0), vel=0.55)
        by = dsp.lowpass(by, 240, 4) * np.exp(-np.arange(len(by)) / (2.4 * FS))
        bus("bass").add(by, t0, 1.0, 1.0)

    # ---------------- 箫
    log("[4/7] reed")
    for bar, beat, notes, vel, lp in score.reed_plan():
        t = score.bt(bar) + beat * score.BEAT
        for k, (midi, dur_beats) in enumerate(notes):
            dur = dur_beats * score.BEAT
            mark(f"reed:{midi}")
            y = voices.reed(midi, dur + 0.45, vel=vel, seed=900 + k)
            if lp is not None:
                y = dsp.lowpass(y, lp, 3)
            bus("reed").add_panned(y, t, -0.15, vel * 1.05, haas_ms=2.5)
            t += dur

    # ---------------- 钟 / 锣
    log("[5/7] bells & gongs")
    for bar, beat, midi, vel in score.BELLS:
        mark(f"bell:{midi}")
        y = voices.bell_buoy(midi, 7.0, vel=vel)
        bus("bell").add_panned(y, score.bt(bar) + beat * score.BEAT,
                               0.35 if midi > 78 else -0.35, vel * 0.9, haas_ms=4.0)
    for bar, beat, midi, vel in score.GONGS:
        mark(f"gong:{midi}")
        y = voices.gong(midi, 12.0, vel=vel)
        bus("gong").add(y, score.bt(bar) + beat * score.BEAT, vel * 0.85, vel * 0.85)

    # ---------------- 船体闷响
    log("[6/7] knocks")
    plan: dict[int, list[float]] = {}
    for d in (score.KNOCKS, score.KNOCKS_II, score.KNOCKS_III, score.KNOCKS_IV):
        for bar, beats in d.items():
            plan.setdefault(bar, []).extend(beats)
    for bar, beats in sorted(plan.items()):
        for bt_ in sorted(set(beats)):
            v = 0.55 if bar < 27 else (0.8 if bar < 39 else 0.4)
            mark("knock")
            y = voices.hull_knock(v)
            bus("knock").add(y, score.bt(bar) + bt_ * score.BEAT, 1.0, 1.0)

    # ---------------- 空气
    log("[7/7] air + room")
    mark("air")
    air = voices.air(score.TOTAL_S, seed=1014)
    air_curve = np.interp(np.arange(N) / FS, [0, 60, 120, 180, 230, 252],
                          [0.15, 0.35, 0.5, 0.4, 0.25, 0.02])
    b = bus("air")
    b.add(air * air_curve, 0.0, 1.0, 0.0)
    b.add(air * air_curve, 0.0, 0.0, 1.0)

    # ---------------- 电平对齐
    for name, b in buses.items():
        raw = dsp.integrated_lufs(np.stack([b.l, b.r], axis=1))
        target = BUS_TARGET_LUFS[name] + cfg["offsets"].get(name, 0.0)
        g = 10 ** ((target - raw) / 20) if np.isfinite(raw) else 1.0
        b.l *= g
        b.r *= g
        log(f"    {name:<7} raw {raw:6.1f} LUFS → {target:6.1f} LUFS（×{g:.4f}）")

    # ---------------- 段落包络
    seg = np.interp(np.arange(N) / FS,
                    [p[0] for p in score.SECTION_GAIN],
                    [p[2] for p in score.SECTION_GAIN])
    from scipy.ndimage import uniform_filter1d
    curve = uniform_filter1d(seg, max(1, int(6.0 * FS)))

    L = np.zeros(N)
    R = np.zeros(N)
    for name, b in buses.items():
        if name in muted:
            continue
        L += b.l * curve
        R += b.r * curve

    # ---------------- 总线 EQ
    L = dsp.highpass(L, 30, 2)
    R = dsp.highpass(R, 30, 2)
    for ch in (L, R):
        ch[:] = dsp.peak_eq(ch, 300, -2.0, q=1.0)
        ch[:] = dsp.peak_eq(ch, 2400, 1.2, q=1.3)
        ch[:] = dsp.shelf(ch, 8500, 1.6, high=True)

    pre = np.stack([L, R], axis=1)
    log(f"    混完（房间前）LUFS={dsp.integrated_lufs(pre):.2f} peak={dsp.peak_db(pre):.2f} dBFS")

    st = voices.apply_room(pre, wet=0.40, size=1.25)
    log(f"    房间后 LUFS={dsp.integrated_lufs(st):.2f} peak={dsp.peak_db(st):.2f} dBFS")
    return st


def master(st: np.ndarray, target_lufs: float = -14.0, ceiling_db: float = -1.0,
           match_spectrum: bool = True) -> np.ndarray:
    """母带：去直流 → 频谱匹配 → 极轻胶合 → 对响度 → 限幅。"""
    from . import analysis
    st = st - np.mean(st, axis=0, keepdims=True)
    if match_spectrum:
        # 两遍：第一遍把大倾斜拉过来，第二遍收掉第一遍的过冲
        for _ in range(2):
            meas = analysis.octave_bands(st)
            st = dsp.match_eq(st, meas, analysis.TARGET_BANDS, analysis.BANDS, max_db=8.0)
    st = dsp.soft_clip(st * 1.2, drive=0.55, ceil=1.0)
    cur = dsp.integrated_lufs(st)
    st = st * (10 ** ((target_lufs - cur) / 20))
    st = dsp.limiter(st, ceiling_db=ceiling_db - 0.3, lookahead_ms=8.0, release_ms=160.0)
    cur = dsp.integrated_lufs(st)
    st = st * (10 ** ((target_lufs - cur) / 20))
    st = dsp.limiter(st, ceiling_db=ceiling_db, lookahead_ms=8.0, release_ms=160.0)
    return st
