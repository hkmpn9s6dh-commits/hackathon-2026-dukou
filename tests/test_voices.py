"""声部与乐谱的断言。

声部：长度、有限性、音高是否真的落在目标上。
乐谱：调式、音域、同时发声的音之间不许有难听的撞音。
"""

import numpy as np
import pytest

from ns import dsp, score, voices
from ns.source import impulse_spectrum

FS = dsp.FS


# ------------------------------------------------------------------ 声部

def test_all_voices_finite_and_nonempty():
    got = {
        "gong": voices.gong(62, 2.0),
        "bell": voices.bell_buoy(74, 2.0),
        "water": voices.water_bed(1.5),
        "lap": voices.lapping(4.0),
        "engine": voices.engine(1.5),
        "knock": voices.hull_knock(),
        "pad": voices.pad([38, 45, 52], 3.0),
        "air": voices.air(2.0),
        "reed": voices.reed(74, 2.0),
    }
    for k, v in got.items():
        assert np.all(np.isfinite(v)), k
        assert len(v) > 0, k
        assert np.max(np.abs(v)) > 1e-6, f"{k} 全是静音"


def test_engine_is_a_pulse_train_not_a_drone():
    """引擎要有 7–9 Hz 的调幅，否则听起来是嗡嗡声不是柴油机。"""
    en = voices.engine(6.0, rpm_hz=7.6)
    env = np.abs(dsp.lowpass(en, 300, 2))
    from scipy.signal import welch
    f, p = welch(env, FS, nperseg=65536)     # 分辨率 0.73 Hz，否则分不清 7.6 与 5.86
    band = (f > 5) & (f < 11)
    peak = f[band][np.argmax(p[band])]
    assert 6.5 < peak < 9.0, f"调幅落在 {peak:.2f} Hz"


def test_gong_pitch_is_where_we_asked():
    for midi in (38, 50, 62, 74):
        target = 440.0 * 2 ** ((midi - 69) / 12)
        x = voices.gong(midi, 3.0)
        f, m = impulse_spectrum(x)
        band = (f > target * 0.97) & (f < target * 1.03)
        assert band.any()
        local = m[band].max()
        outer = m[(f > target * 0.7) & (f < target * 2.5) & ~band].max()
        assert local > 0.2 * outer, f"MIDI {midi} 的基频被埋住了"


def test_pad_has_the_requested_pitch_classes():
    """弦垫必须真的在那个和弦上，而不是一片噪声。"""
    notes = score.CHORDS["Dm9"]
    y = voices.pad(notes, 6.0)
    f, m = impulse_spectrum(y)
    m = m / max(m.max(), 1e-9)
    for midi in notes[1:]:
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        band = (f > hz * 0.985) & (f < hz * 1.015)
        assert band.any()
        assert m[band].max() > 0.02, f"和弦音 {midi}（{hz:.0f} Hz）在垫子里听不见"


def test_reed_is_harmonic():
    """箫要是谐波列；非谐的话就变成又一块钢板了。"""
    midi = 74
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = voices.reed(midi, 1.6)
    t = np.arange(len(y)) / FS
    y = y * np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / FS)
    def energy_at(fc, w=3.0):
        b = (freqs > fc - w) & (freqs < fc + w)
        return spec[b].max() if b.any() else 0.0
    harm = np.mean([energy_at(f0 * k) for k in range(1, 6)])
    between = np.mean([energy_at(f0 * (k + 0.5)) for k in range(1, 5)])
    assert harm > between * 2.0


def test_room_ir_is_decorrelated_stereo():
    l, r = voices.room_ir()
    assert np.corrcoef(l, r)[0, 1] < 0.5


def test_everything_shares_one_source():
    """来源账：所有声部的音源标签必须是同一个。"""
    assert voices.SRC_TAG == "steel_impulse@seed20261201"
    # IMP 必须就是 steel_impulse() 的默认产出
    from ns.source import steel_impulse
    assert np.array_equal(voices.IMP, steel_impulse())


# ------------------------------------------------------------------ 乐谱

def _pc(m):
    return m % 12


def test_motif_shape():
    """动机是 5 个音、8 拍：上四度、上三度、下二度、下三度，落在 9 音上。"""
    pitches = [m for m, _ in score.MOTIF]
    durs = [d for _, d in score.MOTIF]
    assert len(pitches) == 5
    assert abs(sum(durs) - 8.0) < 1e-9
    assert [b - a for a, b in zip(pitches, pitches[1:])] == [3, 4, -2, -3]
    assert _pc(pitches[-1]) == 4          # E = 9 音，故意不解决


def test_no_flat_seven_in_the_melody():
    """全曲没有 B♭（pc 10）出现在旋律里——那是第四段借来的颜色，只能出现在和声里。"""
    for _, _, notes, _, _ in score.reed_plan():
        for midi, _ in notes:
            assert _pc(midi) != 10, f"旋律里出现了 B♭：MIDI {midi}"
    for _, _, midi, _ in score.BELLS:
        assert _pc(midi) != 10
    for _, _, midi, _ in score.GONGS:
        assert _pc(midi) != 10


def test_chords_are_in_range_and_ascending():
    for name, notes in score.CHORDS.items():
        assert notes == sorted(notes), name
        assert min(notes) >= 26, name       # 不低过 D1
        assert max(notes) <= 90, name       # 不高过 F#6


def test_no_grinding_seconds_in_the_low_register():
    """低音区（MIDI < 55）不许有小二度/大二度同时响——那是浑浊的来源。"""
    for name, notes in score.CHORDS.items():
        low = [n for n in notes if n < 55]
        for i in range(len(low)):
            for j in range(i + 1, len(low)):
                gap = abs(low[i] - low[j])
                assert gap > 2, f"{name}: 低音区 {low[i]} 与 {low[j]} 只差 {gap} 个半音"


def test_pad_plan_covers_the_whole_piece_without_gaps():
    plan = sorted(score.pad_plan())
    cursor = 1
    for bar, nb, name, _ in plan:
        assert bar == cursor, f"和声在 bar {bar} 断档（应该从 {cursor} 开始）"
        cursor = bar + nb
    assert cursor == score.BARS + 1, f"和声铺到 bar {cursor - 1}，曲子有 {score.BARS} 小节"


def test_pad_plan_never_overlaps():
    plan = sorted(score.pad_plan())
    for (b1, n1, _, _), (b2, _, _, _) in zip(plan, plan[1:]):
        assert b1 + n1 <= b2


def test_reed_events_do_not_overlap():
    spans = []
    for bar, beat, notes, _, _ in score.reed_plan():
        t = score.bt(bar) + beat * score.BEAT
        for midi, d in notes:
            spans.append((t, t + d * score.BEAT, midi))
            t += d * score.BEAT
    spans.sort()
    for (a1, b1, m1), (a2, b2, m2) in zip(spans, spans[1:]):
        assert a2 >= b1 - 1e-9, f"旋律在 {a1:.2f}–{b1:.2f} 与 {a2:.2f} 重叠"


def test_everything_fits_inside_the_piece():
    end = score.TOTAL_S
    for bar, beat, notes, _, _ in score.reed_plan():
        t = score.bt(bar) + beat * score.BEAT
        total = sum(d for _, d in notes) * score.BEAT
        # 允许最后一口余响越过后 4 分钟，但不许超过总长
        assert t < end - 1.0
        assert t + total < end
    for bar, beat, _, _ in score.BELLS:
        assert score.bt(bar) + beat * score.BEAT < end - 1.0
    for bar, beat, _, _ in score.GONGS:
        assert score.bt(bar) + beat * score.BEAT < end - 1.0


def test_dynamic_arc_peaks_in_section_three():
    """第 III 段（日出）必须是全曲最响的地方，IV 段要退回去。"""
    gains = score.SECTION_GAIN
    peak = max(g[2] for g in gains)
    assert peak == 0.85
    assert any(g[0] <= 120 <= g[1] and g[2] == 0.85 for g in gains)
    assert any(g[0] >= 228 and g[2] < 0.4 for g in gains)


def test_cadence_is_not_resolved():
    """最后两件发声的东西不许落在主音 D 上——这是这首曲子的意思。"""
    last_reed_bar = max(b for b, _, _, _, _ in score.reed_plan())
    last = [n for b, _, notes, _, _ in score.reed_plan() if b == last_reed_bar for n, _ in notes]
    assert _pc(last[-1]) in (4, 9), "最后落在主音上，等于把话说完了"
