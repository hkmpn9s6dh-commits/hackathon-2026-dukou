"""把"听觉"换成可以看的东西：频谱图、短时响度曲线、频段表。

产出 proof/sheet.png（给盲排评委看的那张"聆听表"）与 proof/audit.json。
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ns import analysis, dsp, plot, score, wavio  # noqa: E402

OUT = ROOT / "out"
PROOF = ROOT / "proof"
# 表上的标签用英文：这张图是给不看中文的评委看的，也免了跨机字体依赖
SECTIONS = [(0.0, 60.0, "I  before light"), (60.0, 120.0, "II  first grey"),
            (120.0, 180.0, "III  sunrise"), (180.0, 252.0, "IV  far shore")]


def html_escape_free_name(path: Path) -> str:
    return path.name


def main() -> int:
    PROOF.mkdir(exist_ok=True)
    x = wavio.read_wav24(OUT / "muxi.wav")
    bands = analysis.octave_bands(x)
    err = bands - analysis.TARGET_BANDS
    edges = [s[0] for s in SECTIONS] + [SECTIONS[-1][1]]
    sec = analysis.section_lufs(x, edges)
    audit = {
        "bands_db_rel_1k": {str(int(b)): round(float(v), 1)
                            for b, v in zip(analysis.BANDS, bands)},
        "band_error_db": {str(int(b)): round(float(v), 1)
                          for b, v in zip(analysis.BANDS, err)},
        "max_abs_band_error_db": round(float(np.max(np.abs(err))), 2),
        "section_lufs": {lbl: v for (_, _, lbl), v in zip(SECTIONS, sec)},
        "measures": json.loads((OUT / "build.json").read_text(encoding="utf-8"))["measures"],
    }
    (PROOF / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    plot.sheet(x, PROOF / "sheet.png")
    print("wrote proof/sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
