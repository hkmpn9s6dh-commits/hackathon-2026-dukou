"""单源脉冲的断言。整首曲子能不能成立，全看这一个函数。"""

import numpy as np

from ns import dsp
from ns.source import BASE_HZ, mode_table, steel_impulse, impulse_spectrum

FS = dsp.FS


def test_impulse_is_normalised_and_finite():
    x = steel_impulse()
    assert np.all(np.isfinite(x))
    assert abs(np.max(np.abs(x)) - 1.0) < 1e-9


def test_impulse_is_long_and_still_ringing_at_two_seconds():
    """钢板要能响够久——弦垫全靠这条尾巴。"""
    x = steel_impulse(seconds=4.2)
    tail = np.sqrt(np.mean(x[int(2.0 * FS) :] ** 2))
    assert tail > 1e-3, "两秒后已经没有能量，拉不出弦垫"


def test_impulse_decays():
    x = steel_impulse()
    early = np.sqrt(np.mean(x[: FS // 10] ** 2))
    late = np.sqrt(np.mean(x[int(3.5 * FS) :] ** 2))
    assert late < early * 0.05


def test_mode_table_is_inharmonic():
    """模态不能是整数倍——否则听起来像风琴，不像钢板。"""
    ratios = np.array([r for r, _ in mode_table()])
    assert len(ratios) > 20
    assert np.all(np.diff(ratios) > 0)
    # 前 20 个模态里，至少一半不是整数比
    first = ratios[:20]
    nonint = np.sum(np.abs(first - np.round(first)) > 0.03)
    assert nonint >= 10


def test_fundamental_matches_base():
    x = steel_impulse(seconds=3.0)
    f, m = impulse_spectrum(x)
    band = (f > BASE_HZ * 0.6) & (f < BASE_HZ * 1.6)
    assert band.any()
    peak = f[band][np.argmax(m[band])]
    assert abs(peak - BASE_HZ) < 6.0, f"基频落在 {peak:.1f} Hz"


def test_velocity_changes_timbre_not_just_level():
    """敲得轻的版本，高频占比必须更低（不是简单缩放）。"""
    soft = steel_impulse(strength=0.2)
    hard = steel_impulse(strength=1.0)
    cs, ch = dsp.spectral_centroid(soft), dsp.spectral_centroid(hard)
    assert cs < ch * 0.92


def test_deterministic():
    a = steel_impulse()
    b = steel_impulse()
    assert np.array_equal(a, b)
