"""渲染三个读法，各出一张聆听表。标签用甲乙丙，映射另存 proof/variant-map.json。

    python scripts/variants.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ns import plot, render as R, wavio  # noqa: E402

OUT = ROOT / "out"
PROOF = ROOT / "proof"

ORDER = ["water", "boat", "bell"]      # 甲乙丙 → 内部名（只在本地这份文件里）
LABELS = ["甲", "乙", "丙"]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    PROOF.mkdir(exist_ok=True)
    mapping = {}
    for label, name in zip(LABELS, ORDER):
        print(f"=== {label} ({name}) ===")
        st = R.render(variant=name)
        st = R.master(st, target_lufs=-14.0, ceiling_db=-1.2)
        p = OUT / f"variant-{label}.wav"
        wavio.write_wav24(p, st)
        plot.sheet(st, PROOF / f"sheet-{label}.png")
        from ns import dsp
        mapping[label] = {
            "internal": name,
            "lufs": round(dsp.integrated_lufs(st), 2),
            "lra": round(dsp.lra(st), 2),
            "tp": round(dsp.true_peak_dbtp(st), 2),
            "corr": round(dsp.stereo_correlation(st), 3),
            "mono_excess": round(dsp.mono_fold_excess_db(st), 2),
        }
        print("   ", mapping[label])
    (PROOF / "variant-map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote proof/variant-map.json（这份不能给评委）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
