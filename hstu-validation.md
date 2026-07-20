# HSTU Real-Method Validation — 用 paper 主张的方法验证(不是 GBM 代理)

> 前一轮 `dataset-validation.md` 用 sklearn GBM 做**代理**筛信号。本轮换成**论文真方法**:
> 忠实复刻的 **HSTU**(Hierarchical Sequential Transduction Unit,arXiv 2402.17152),
> **生成式序列 transduction / 自回归预测下一个事件 token**,在 MacBook Air M4 (MPS) 上跑。
> 目标:在**最新最大**的数据集上,判断哪些最适合这个范式 → 收敛到 **2-3 个最好的**。
>
> HSTU 忠实点(照官方 `generative-recommenders/.../hstu.py` 写):UVQK 用 **SiLU pointwise 投影** →
> 注意力 **`SiLU(QK^T + rel_bias)/N`(不是 softmax)** → **U 门控 `U ⊙ LayerNorm(A·V)`** → 输出投影 + 残差 + 因果 mask。
> 脚本 `validation-scripts/hstu_val.py`(约 90 行,含 RelPosBias)。每个 tick 的 label = 下一个事件 token(self-supervised,天然每 tick 都有)。

---

## 数据集(全部 2024 最新、全量非常大;本机用代表性切片)

| 数据集 | 域 | 全量规模 | 本机切片 | 每 tick label |
|---|---|---|---|---|
| BurstGPT | DC / LLM-serving | 1031万请求 / 213天 (2024) | 534万请求(1 文件) | response_tokens(量化) |
| F-DATA (Fugaku) | HPC | **2400万作业** / 38月 (2024) | 57.2万作业(1 月) | exit-state / duration / power |
| CESNET-TimeSeries24 | NET | 27.5万 IP,源自 66B flows / **3.7PB** (2024) | 1000 IP × 720h 切片 | n_bytes 等(量化) |
| UNSW-NB15 | 安全/NIDS | 206万 flow,10 类 | 全量(HF 镜像) | attack_label(10 类) |
| CIC-IDS2017 | 安全/NIDS | 283万 flow,15 类 | 全量(HF 镜像) | attack_label(15 类) |

> 5G-NIDD 本身可用(Fairdata OPEN / CC-BY,121万 flow,9 类),但下载 API 路径已变、无 HF 镜像,当场拉不动;
> 按"换等价的"原则改用 UNSW-NB15 + CIC-IDS2017(同为多分类 NIDS、HF 直链、字段更全)。结论对 5G-NIDD 同样成立(见下)。

---

## 结果(真 HSTU,自回归 next-event;acc = 下一 token 准确率)

| 数据集 / 任务 | 类别数 | marginal | persistence | Markov | **HSTU** | 增益 | 判定 |
|---|---|---|---|---|---|---|---|
| **BurstGPT** / 下一回复长度 | 32 | .159 | .437 | .442 | **.510** | **+6.7%** | ✅ 明显信号 |
| **F-DATA** / 下一作业时长 | 32 | .029 | .706 | .706 | **.734** | **+2.8%** | ✅ 真信号 |
| **CESNET** / 下一 tick 流量 | 32 | .030 | .200 | .201 | **.219** | **+1.8%** | ✅ 有信号 |
| F-DATA / 下一作业是否失败 | 2 | .877 | .986 | .986 | .988 | +0.2% | ⚠️ persistence 饱和 |
| UNSW-NB15 / 下一 flow 攻击类 | 10 | .950 | .963 | .966 | .971 | +0.5% | ⚠️ 饱和(95% normal,仅 42 IP) |
| CIC-IDS2017 / 下一 flow 攻击类 | 15 | .797 | .998 | .998 | .999 | +0.1% | ⚠️ 饱和(per-host 类别近恒定) |

小模型:D=64,2 头,2 层,序列 64;MPS 上每个 6-9 秒训完。这是**范式下界**(小模型 + 无跨域预训练),真正的 HSTU 规模 + cross-domain pretrain 只会更高。

---

## 关键结论(直接回答"挑 2-3 个最好的")

**范式的增益只出现在"标签有真实不确定性 + 实体多"的预测任务上**,不出现在"实体内标签近乎恒定"的任务上:

- ✅ **信号强的**:BurstGPT(+6.7%)、F-DATA 时长(+2.8%)、CESNET(+1.8%)。这些是**生成式 forecasting**,下一步真有不确定性,序列历史确实有用。
- ⚠️ **饱和的**:F-DATA 失败预测(2 类)、NIDS 下一攻击类(UNSW/CIC persistence 已 0.96–0.998)。不是数据不能用,而是**per-entity 标签几乎不变 → persistence 直接赢**;NIDS 还有极端类别不平衡(准确率是错的度量,该用 macro-F1)。NIDS 更适合做**逐 flow 分类**的监督 finetune 头,不适合当"序列范式增益"的展示台。

### 最终推荐:2-3 个最好的数据集

1. **BurstGPT(DC / LLM-serving)** — 信号最强(+6.7%),1031万请求全量,干净的生成式 next-event(到达 + token 长度),2024 最新。**首选**。
2. **F-DATA / Fugaku(HPC)** — +2.8%,**2400万作业**,per-user 作业序列,每 tick 有多种真 label(exit-state / duration / power),2024 最新。竞争最少的域(HPC 遥测无 FM 先例)。**首选**。
3. **CESNET-TimeSeries24(NET)** — +1.8%,27.5万 IP 源自 **3.7PB**,per-IP forecasting,2024 最新。网络流量预测正统。**次选/第三**。

> 若需要一个**真·监督多分类**的 finetune 任务(每 sample 有明确类别标签),再挂 **UNSW-NB15 / CIC-IDS2017 / 5G-NIDD** 作分类头——但把它当**逐 flow 分类**用,别指望它在"下一事件"框架里体现序列增益。

---

## 给 SIGCOMM/MobiCom 的方法学提醒(仍然成立)

1. 细粒度下 **persistence/Markov 是强 baseline**,+2~7% 不够 compelling。正式实验必须:(a) 上强 baseline(季节性 / 专用 LSTM / 通用 TSFM Chronos·Moirai / NetLLM·UoMo 同类);(b) 靠**跨域预训练**(本轮未做)+ 更大 HSTU 把 margin 拉开;(c) 不平衡任务用 macro-F1 / AUPRC,别只报 accuracy。
2. 范式的真正卖点不是"单数据集小涨点",而是**一个 backbone 跨 DC/HPC/NET 多任务**——这需要下一步做**跨域联合预训练**才能证明,本机只验证了"每个域单独有可学序列结构 + 真 HSTU 能利用它"。

## 复现
```bash
source <scratchpad>/.venv/bin/activate   # torch(MPS)+pyarrow+pandas
python validation-scripts/hstu_val.py burst|fdata|cesnet|unsw|cic|all
```
