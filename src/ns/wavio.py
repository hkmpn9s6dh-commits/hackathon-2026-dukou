"""WAV 的读写（24-bit PCM）。

不用标准库 wave：它对 24-bit 的 readframes 会少读，读出来的长度对不上。
这里直接按 chunk 解析字节，读回来的长度必须等于写进去的帧数——这是闸 C 的一条。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import dsp


def write_wav24(path: Path, x: np.ndarray, fs: int | None = None) -> int:
    fs = fs or dsp.FS
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    n, ch = x.shape
    ints = np.round(x * (2 ** 23 - 1)).astype("<i4")
    b = ints.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    byte_rate = fs * ch * 3
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(b)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, ch, fs, byte_rate, ch * 3, 24))
        f.write(b"data")
        f.write(struct.pack("<I", len(b)))
        f.write(b)
    return n


def read_wav24(path: Path) -> np.ndarray:
    """返回 (frames, ch) 的 float64，范围 [-1, 1)。"""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path} 不是 RIFF/WAVE")
    pos = 12
    ch = bits = fs = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            _fmt, ch, fs = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if data is None or bits != 24:
        raise ValueError(f"{path}: 期望 24-bit data chunk，拿到 bits={bits}")
    if len(data) % (ch * 3):
        raise ValueError(f"{path}: data 长度 {len(data)} 不是帧长的整数倍")
    u8 = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    i32 = np.zeros((u8.shape[0], 4), dtype=np.uint8)
    i32[:, :3] = u8
    # 24-bit 是二补数：最高位为 1 要补成负号，否则 -0.5 会读成 +1.5
    i32[:, 3] = np.where(u8[:, 2] >= 128, 255, 0)
    x = i32.view("<i4").reshape(-1, ch).astype(np.float64) / float(2 ** 23)
    return x
