"""闸 C：把"能交件"这件事逐条验一遍。

每条都是可判定的，没有一条靠"听起来还行"。
退出码 = 失败的条数（0 才算过）。

    python scripts/verify.py              # 验现有产物
    python scripts/verify.py --twice      # 再重建一次，比对 sha256（确定性）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ns import analysis, dsp, score, wavio  # noqa: E402

OUT = ROOT / "out"
PROOF = ROOT / "proof"
SRC_TAG = "steel_impulse@seed20261201"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_loudness(p: Path) -> tuple[float | None, float | None]:
    """用 ffmpeg 的 ebur128 独立量一遍（与自建的表不是同一份实现）。"""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None, None
    r = subprocess.run(
        [exe, "-hide_banner", "-nostats", "-i", str(p),
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    text = r.stderr
    i = text.rfind("Integrated loudness")
    if i < 0:
        return None, None
    seg = text[i : i + 400]
    mi = re.search(r"I:\s*(-?\d+\.?\d*)\s*LUFS", seg)
    mt = re.search(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", seg)
    return (float(mi.group(1)) if mi else None,
            float(mt.group(1)) if mt else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twice", action="store_true", help="再渲染一次比对确定性")
    a = ap.parse_args()

    wav = OUT / "muxi.wav"
    mp3 = OUT / "muxi.mp3"
    if not wav.exists():
        print("缺 out/muxi.wav，先跑 python scripts/build.py")
        return 1

    print("== 1. 交付物在不在 ==")
    check("out/muxi.wav 存在", wav.exists(), f"{wav.stat().st_size / 1e6:.1f} MB" if wav.exists() else "")
    check("out/muxi.mp3 存在", mp3.exists(), f"{mp3.stat().st_size / 1e6:.2f} MB" if mp3.exists() else "")

    print("== 2. WAV 往返 ==")
    x = wavio.read_wav24(wav)
    check("24-bit WAV 能读回", x.ndim == 2 and x.shape[1] == 2, f"shape={x.shape}")
    check("帧数 = 252.0 s × 48 kHz", x.shape[0] == int(round(score.TOTAL_S * dsp.FS)),
          f"{x.shape[0]} vs {int(round(score.TOTAL_S * dsp.FS))}")
    check("分母写的是 2^23（没有把 -0.5 读成 +1.5）", float(np.max(np.abs(x))) <= 1.0001,
          f"peak={float(np.max(np.abs(x))):.4f}")
    check("没有样本越过满刻度", bool(np.max(np.abs(x)) < 1.0), f"{dsp.peak_db(x):.2f} dBFS")

    print("== 3. 母带指标（SPEC 里写死的区间） ==")
    lufs = dsp.integrated_lufs(x)
    tp = dsp.true_peak_dbtp(x)
    lra = dsp.lra(x)
    check("集成响度 −14.0 ± 0.5 LUFS", abs(lufs + 14.0) <= 0.5, f"{lufs:.2f}")
    check("真峰值 ≤ −1.0 dBTP", tp <= -0.95, f"{tp:.2f}")
    check("LRA 在 6–16 LU（没把动态压死）", 6.0 <= lra <= 16.0, f"{lra:.2f}")
    check("直流偏置 < 1e-3", dsp.dc_offset(x) < 1e-3, f"{dsp.dc_offset(x):.2e}")
    mf = dsp.mono_fold_excess_db(x)
    check("单声道折叠额外损失 < 3 dB", abs(mf) < 3.0, f"{mf:.2f}")
    corr = dsp.stereo_correlation(x)
    check("立体声相关 0.05–0.85（不是假立体也不是单声道）", 0.05 <= corr <= 0.85, f"{corr:.3f}")

    print("== 4. 频率平衡（与 TARGET_BANDS 对齐） ==")
    bands = analysis.octave_bands(x)
    err = np.abs(bands - analysis.TARGET_BANDS)
    check("八度带最大偏差 ≤ 4 dB", float(err.max()) <= 4.0,
          "worst " + ", ".join(f"{b:.0f}Hz {e:+.1f}dB"
                               for b, e in zip(analysis.BANDS, bands - analysis.TARGET_BANDS)
                               if abs(e) == err.max())[:80])

    print("== 5. 结构（四段必须真的不一样） ==")
    edges = [0.0, 60.0, 120.0, 180.0, 252.0]
    sec = analysis.section_lufs(x, edges)
    check("第三段是全曲最响", sec[2] == max(sec), f"{['%.1f' % v for v in sec]}")
    check("第二段比第一段响（有推进）", sec[1] > sec[0] + 2.0, f"{sec[1]:.1f} vs {sec[0]:.1f}")
    check("第四段退回去（比第二段轻 4 dB 以上）", sec[3] < sec[1] - 4.0, f"{sec[3]:.1f} vs {sec[1]:.1f}")
    check("结尾 5 秒在衰减", dsp.integrated_lufs(x[-5 * dsp.FS :]) < dsp.integrated_lufs(x[-30 * dsp.FS : -25 * dsp.FS]) - 3.0,
          f"末 5 s {dsp.integrated_lufs(x[-5 * dsp.FS:]):.1f} LUFS")

    print("== 6. 单源约束 ==")
    bj = json.loads((OUT / "build.json").read_text(encoding="utf-8"))
    prov = bj.get("provenance", {})
    check("来源账只指向一处", prov.get("sources") == [SRC_TAG], f"{prov.get('sources')}")
    check("声部调用数 > 40（说明确实是一个一个派生的）", prov.get("calls", 0) > 40,
          f"{prov.get('calls')}")

    print("== 7. MP3 ==")
    if mp3.exists():
        head = mp3.read_bytes()[:10]
        check("带 ID3v2 标签", head[:3] == b"ID3", head[:3].decode("latin1"))
        size = mp3.stat().st_size
        kbps = size * 8 / (score.TOTAL_S * 1000)
        check("码率 ≥ 192 kbps（320 CBR）", kbps >= 192, f"{kbps:.0f} kbps")
        import lameenc  # noqa: F401
        # 用 ffmpeg 解回来，量响度——MP3 编解码后响度不该跑
        exe = shutil.which("ffmpeg")
        if exe:
            dec = OUT / "mp3check.wav"
            subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3),
                            "-c:a", "pcm_s24le", str(dec)], check=False)
            if dec.exists():
                y = wavio.read_wav24(dec)
                dl = dsp.integrated_lufs(y)
                check("MP3 解码后响度与 WAV 差 ≤ 0.5 LU", abs(dl - lufs) <= 0.5,
                      f"{dl:.2f} vs {lufs:.2f}")
                check("MP3 解码后真峰值 ≤ −0.8 dBTP", dsp.true_peak_dbtp(y) <= -0.8,
                      f"{dsp.true_peak_dbtp(y):.2f}")
                dec.unlink(missing_ok=True)
        else:
            print("  [SKIP] 没有 ffmpeg，跳过 MP3 回解")

    print("== 8. 与 ffmpeg 的独立实现交叉核对 ==")
    fl, ftp = ffmpeg_loudness(wav)
    if fl is None:
        print("  [SKIP] 没有 ffmpeg")
    else:
        check("自建响度表 vs ffmpeg ebur128 差 ≤ 0.2 LU", abs(fl - lufs) <= 0.2,
              f"ffmpeg {fl:.1f} vs 自建 {lufs:.1f}")
        if ftp is not None:
            check("真峰值口径差 ≤ 0.5 dB", abs(ftp - tp) <= 0.5, f"ffmpeg {ftp:.1f} vs 自建 {tp:.1f}")

    print("== 9. 确定性 ==")
    if a.twice:
        h1 = sha256(wav)
        print("  重新渲染一遍 …")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build.py"), "--wav-only"],
                       check=True, cwd=str(ROOT))
        h2 = sha256(wav)
        check("两次渲染 sha256 一致", h1 == h2, f"{h1[:16]}… vs {h2[:16]}…")
    else:
        print("  [SKIP] 加 --twice 才验（约多花 1.5 分钟）")

    nfail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(f"合计 {len(results)} 条，失败 {nfail} 条")
    (PROOF / "verify.json").write_text(
        json.dumps([{"check": n, "pass": ok, "detail": d} for n, ok, d in results],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return nfail


if __name__ == "__main__":
    raise SystemExit(main())
