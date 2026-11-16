# 渡口 · Crossing at First Light

一首 4 分 12 秒的器乐作品，投 **NEXT SOUND Awards 2026 · Best Cinematic Single**。

**整首曲子里没有一个采样、没有一个预置音色、没有一段外部音频。** 所有的声音——江水、柴油机、
船体的闷响、钟、弦垫、箫、以及这些声音所在的房间——都由**同一个 400 毫秒的合成脉冲**派生：
一块被敲响的钢板。

```
python scripts/build.py     # 从零重建 out/muxi.wav + out/muxi.mp3（约 90 秒）
bash  scripts/verify.sh     # 25 条可判定断言（闸 C）
```

## 目录

| 路径 | 是什么 |
|---|---|
| `src/ns/source.py` | 那块钢板：36 阶非谐模态 + 3 ms 激励 |
| `src/ns/voices.py` | 每一样乐器怎么从那块板长出来 |
| `src/ns/score.py` | 乐谱：52 小节、D 多利亚、四段、一个五音动机 |
| `src/ns/render.py` | 九条总线的混音与母带 |
| `src/ns/dsp.py` | 滤波器、卷积、粒度、限幅、BS.1770-4 响度表 |
| `src/ns/analysis.py` | 八度带平衡与动态分析 |
| `src/ns/plot.py` | 聆听表（盲排用的那三张图） |
| `src/ns/wavio.py` | 24-bit WAV 读写（不用标准库 wave：它读 24-bit 会少读） |
| `docs/` | SPEC、尺子、制作记录、验收、交件 |
| `tests/` | 38 条单元断言 |
| `scripts/verify.py` | 25 条交付物断言 |

## 依赖

Python 3.11+，`numpy / scipy / lameenc / matplotlib / pytest`。
`scripts/verify.sh` 会在没有 `.venv` 时自己建一个。

## 声明的 AI 使用（比赛要求申报）

用到的 AI 工具是**一个大语言模型 agent**（CodeBuddy / WorkBuddy 上的 CB 主驾）。
它写了 DSP 引擎、乐谱、混音自动化与母带链，并运行构建。**没有使用任何生成式音频模型**
（没有 Suno、没有 Udio、没有 MusicGen），也没有任何采样库。所有声音是程序合成的。

## 已知的边界

写这份东西的 agent **没有耳朵**。它没有听过这首曲子，也不能听。
所以这个仓库里能用数字说的都说尽了，不能说的一律标注为"未知"，见 `docs/04-PROOF.md` 末尾。
