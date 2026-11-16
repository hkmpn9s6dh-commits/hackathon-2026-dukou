"""动机能不能被听成旋律？

评委（GLM-5.3F）在盲排里点出的唯一盲区是：
「五音动机到底成不成旋律——谱上只能确认那几条能量线在，听不出它们站不站得住。」

这一条其实是可以量的：在每一个动机音的**起音窗口**里，
看它的基频在局部背景上高出去多少（谱突出度）。
如果每个音都从背景里冒出来，它在混音里就是"一条旋律线"；
如果冒不出来，它就只是垫子的一部分。

    python scripts/motif_check.py

判据：中位谱突出度 ≥ 6 dB，且 80% 以上的音 ≥ 3 dB。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ns import dsp, score, wavio  # noqa: E402

OUT = ROOT / "out"


def prominence_db(x: np.ndarray, at_s: float, f0: float, win_s: float = 0.55) -> float:
    """在 at_s 附近的窗口里，f0 处的能量比它左右各 1/3 八度的中位数高多少 dB。"""
    i0 = int(max(0, (at_s + 0.06) * dsp.FS))          # 跳过起音的 60 ms
    n = int(win_s * dsp.FS)
    seg = x[i0 : i0 + n].mean(axis=1) if x.ndim == 2 else x[i0 : i0 + n]
    if len(seg) < n // 2:
        return float("nan")
    f, p = signal.welch(seg, dsp.FS, nperseg=min(16384, len(seg)))
    p = np.maximum(p, 1e-20)
    # ±2% 以内算"这个音"；两侧各 1/3 八度（含垫子的谐波）算"它要冒出来的背景"
    near = (f > f0 / 1.02) & (f < f0 * 1.02)
    side = ((f > f0 / 1.34) & (f < f0 / 1.045)) | ((f > f0 * 1.045) & (f < f0 * 1.34))
    if not near.any() or not side.any():
        return float("nan")
    return float(10 * np.log10(p[near].max() / np.median(p[side])))


def main() -> int:
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "muxi.wav"
    x = wavio.read_wav24(wav)
    rows = []
    for bar, beat, notes, vel, lp in score.reed_plan():
        t = score.bt(bar) + beat * score.BEAT
        for midi, dur_beats in notes:
            f0 = 440.0 * 2 ** ((midi - 69) / 12)
            rows.append({
                "bar": bar, "t": round(t, 2), "midi": midi, "hz": round(f0, 1),
                "vel": vel, "lp": lp,
                "prom_db": round(prominence_db(x, t, f0), 2),
            })
            t += dur_beats * score.BEAT

    vals = np.array([r["prom_db"] for r in rows], dtype=float)
    good = vals[~np.isnan(vals)]
    med = float(np.median(good))
    frac = float(np.mean(good >= 3.0))
    buried = [r for r in rows if not np.isnan(r["prom_db"]) and r["prom_db"] < 3.0]

    print(f"{'bar':>4} {'t(s)':>7} {'midi':>5} {'Hz':>7} {'vel':>5} {'LP':>6} {'突出度':>7}")
    for r in rows:
        print(f"{r['bar']:>4} {r['t']:>7.2f} {r['midi']:>5} {r['hz']:>7.1f} "
              f"{r['vel']:>5.2f} {str(r['lp'] or '-'):>6} {r['prom_db']:>+7.2f}")
    print()
    print(f"中位谱突出度 {med:+.2f} dB（判据 ≥ +6.0）")
    print(f"≥ +3 dB 的比例 {frac * 100:.0f}%（判据 ≥ 80%）")
    if buried:
        print(f"被埋的音 {len(buried)} 个：" +
              ", ".join(f"bar{r['bar']} midi{r['midi']} ({r['prom_db']:+.1f})" for r in buried))
    ok = med >= 6.0 and frac >= 0.8
    print("结论：" + ("PASS —— 动机在混音里是一条听得出的线" if ok else "FAIL —— 动机只是垫子的一部分"))

    (ROOT / "proof" / "motif.json").write_text(
        json.dumps({"median_prom_db": round(med, 2), "frac_above_3db": round(frac, 3),
                    "pass": ok, "notes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
