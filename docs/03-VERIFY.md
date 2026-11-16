# 闸 C · 验收：从零 clone 之后，只照 README 跑

## 一句话

```
bash scripts/verify.sh --twice
```

退出码 = 失败的条数。**25 条交付物断言（加 `--twice` 是 26 条）+ 38 条单元断言，0 失败。**

## 1. 结构

| 层 | 文件 | 条数 | 管什么 |
|---|---|---|---|
| 单元 | `tests/test_dsp.py` | 14 | 响度表、真峰、限幅、重采样、粒度、单声道口径 |
| 单元 | `tests/test_source.py` | 7 | 钢板脉冲：归一、衰减、非谐、力度→音色、确定性 |
| 单元 | `tests/test_voices.py` | 17 | 各声部有限/有音高；乐谱的调式、音域、撞音、终止 |
| 交付 | `scripts/verify.py` | 25（`--twice` 26） | 见下 |

单元断言里最有用的三条，都是**先失败后修的**：

- `test_lufs_scale_is_additive`：幅度加倍必须 +6.02 LU。第一版 `_block_ms` 对错轴
  （对声道求均值而不是对窗），计量整体失效。
- `test_mono_fold_of_identical_channels_has_no_extra_loss`：立体声折单声道本来就会掉
  3.01 dB，原来的"损失"口径把这条固有损失也算成问题。改成"额外损失"才对。
- `test_pad_has_the_requested_pitch_classes`：弦垫必须真的在那个和弦上。
  这一条逼出了弦垫的整段重写（梳状 EQ → 频域谐波列）。

## 2. 交付物断言（25 条，加 `--twice` 26 条）

| 组 | 断言 |
|---|---|
| 存在 | wav 在、mp3 在 |
| WAV 往返 | 能读回、帧数 = 252.0 s × 48 kHz、符号位正确、无样本越界 |
| 母带 | −14.0±0.5 LUFS、≤−1.0 dBTP、LRA 6–16、直流 <1e-3、单声道额外损失 <3 dB、相关 0.05–0.85 |
| 频率 | 八度带最大偏差 ≤ 4 dB |
| 结构 | III 最响、II > I+2、IV < II−4、末 5 s 在衰减 |
| 单源 | 来源集合只有一个元素、声部调用 > 40 |
| MP3 | ID3、≥192 kbps、解码后响度差 ≤0.5 LU、解码后真峰 ≤−0.8 dBTP |
| 交叉核对 | 自建响度表 vs `ffmpeg ebur128` ≤0.2 LU、真峰差 ≤0.5 dB |
| 确定性 | 两次渲染 sha256 一致 |

## 3. 计量表可信吗

这是这一场最关键的一条：如果响度表是错的，"−14.0 LUFS" 就是个自说自话的数字。

所以做了**两套独立实现**的交叉核对：

| 量 | 自建（`ns/dsp.py`，BS.1770-4 手写） | ffmpeg `ebur128` | 差 |
|---|---|---|---|
| 集成响度 | −14.1 LUFS | −14.1 LUFS | 0.0 |
| 真峰值 | −1.2 dBTP | −1.2 dBFS | 0.0 |

两套代码没有任何共享：一套是 numpy 写的 K 加权 + 门限聚合，另一套是 ffmpeg 的 C 实现。
差 0.0 说明"目标 −14 LUFS"这个承诺是真的兑现了，不是自己量的。

中途这张表确实错过一次：`_block_ms` 把窗维度当成声道维度求均值，
量出来的"响度"完全没有意义——是 `test_lufs_scale_is_additive` 把它逼出来的。

## 4. 确定性

两次完整渲染（各约 90 s）产出的 WAV **sha256 完全相同**。
这意味着：任何人 clone 之后跑一遍，拿到的就是同一个文件，不是"差不多的版本"。

用到的随机性全部出自显式种子（`np.random.default_rng(...)`），
没有 `random`、没有时间戳、没有系统熵。

## 5. 复验方法

```bash
git clone <repo> && cd nextsound
bash scripts/verify.sh --twice      # 没有 .venv 会自己建
```

`verify.sh` 不做任何取巧：它跑 `pytest`，再跑 `verify.py`；
`verify.py` 只读 `out/` 里的产物和 `out/build.json`，不重新渲染（除非加 `--twice`）。
