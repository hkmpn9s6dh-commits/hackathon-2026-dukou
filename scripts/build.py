"""一条命令从零重建交付物。

用法：
    python scripts/build.py            # 全量：wav + mp3
    python scripts/build.py --wav-only # 只出 wav（调试用）

产出：
    out/muxi.wav   48 kHz / 24-bit PCM / 立体声
    out/muxi.mp3   48 kHz / 320 kbps CBR / 带 ID3
    out/build.json 本次渲染的全部计量值
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ns import dsp, score, wavio  # noqa: E402
from ns import render as R  # noqa: E402

OUT = ROOT / "out"

TITLE = "渡口"
SUBTITLE = "Crossing at First Light"
ARTIST = "CB 主驾（CodeBuddy / WorkBuddy agent）"
COMMENT = (
    "100% synthesized from a single 4.2 s modal-synthesis steel-plate impulse. "
    "No samples, no presets, no external audio. "
    "AI tools used: an LLM agent wrote the DSP engine, the score and the mix automation "
    "and rendered the master; no generative-audio model was used."
)


def write_mp3(path: Path, x: np.ndarray, bitrate: int = 320) -> None:
    import lameenc

    pcm = np.clip(x, -1.0, 1.0)
    ints = np.round(pcm * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(dsp.FS)
    enc.set_channels(2)
    enc.set_quality(2)
    data = enc.encode(ints.tobytes())
    data += enc.flush()
    path.write_bytes(bytes(data))


def id3_tags(path: Path) -> None:
    """写 ID3v2.3 标签（手写，避免额外依赖）。"""
    def frame(fid: bytes, text: str) -> bytes:
        body = b"\x00" + text.encode("utf-16")   # 带 BOM 的 UTF-16
        import struct as st
        return fid + st.pack(">I", len(body)) + b"\x00\x00" + body

    frames = b"".join([
        frame(b"TIT2", f"{TITLE} / {SUBTITLE}"),
        frame(b"TPE1", ARTIST),
        frame(b"TALB", "NEXT SOUND Awards 2026 · Cinematic Single"),
        frame(b"TCON", "Cinematic"),
        frame(b"TDRC", "2026"),
        frame(b"COMM", COMMENT),
    ])
    import struct as st
    header = b"ID3\x03\x00\x00" + st.pack(">I", len(frames))
    path.write_bytes(header + frames + path.read_bytes())


def measure(x: np.ndarray) -> dict:
    return {
        "duration_s": round(len(x) / dsp.FS, 3),
        "integrated_lufs": round(dsp.integrated_lufs(x), 3),
        "true_peak_dbtp": round(dsp.true_peak_dbtp(x), 3),
        "sample_peak_db": round(dsp.peak_db(x), 3),
        "lra_lu": round(dsp.lra(x), 3),
        "dc_offset": round(dsp.dc_offset(x), 8),
        "stereo_corr": round(dsp.stereo_correlation(x), 4),
        "mono_fold_excess_db": round(dsp.mono_fold_excess_db(x), 3),
        "spectral_centroid_hz": round(dsp.spectral_centroid(x), 1),
        "low_ratio_lt120": round(dsp.low_ratio(x, 120), 4),
        "onset_density_per_s": round(dsp.onset_density(x), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-only", action="store_true")
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"render … 目标 {score.TOTAL_S:.1f} s / {dsp.FS} Hz")
    st = R.render()
    print(f"  render 用时 {time.time() - t0:.1f} s，未母带 LUFS={dsp.integrated_lufs(st):.2f}")
    # 留 0.2 dB 余量：MP3 编解码会让真峰值略微上冲
    st = R.master(st, target_lufs=a.target_lufs, ceiling_db=-1.2)
    print(f"  master 用时 {time.time() - t0:.1f} s")

    m = measure(st)
    print("  " + json.dumps(m, ensure_ascii=False))

    wavio.write_wav24(OUT / "muxi.wav", st)
    print(f"  wav  {(OUT / 'muxi.wav').stat().st_size / 1e6:.1f} MB")
    if not a.wav_only:
        write_mp3(OUT / "muxi.mp3", st)
        id3_tags(OUT / "muxi.mp3")
        print(f"  mp3  {(OUT / 'muxi.mp3').stat().st_size / 1e6:.2f} MB")

    prov = {
        "calls": len(R.LAST_LOG),
        "sources": sorted({s for _, s in R.LAST_LOG}),
    }
    (OUT / "build.json").write_text(
        json.dumps({"measures": m, "provenance": prov,
                    "seconds": round(time.time() - t0, 1)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
