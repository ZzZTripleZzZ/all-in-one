# NFM 实验设计（从用户逐条规格拼装,2026-07-10）

方法基于 **meta-recsys/generative-recommenders** 的 HSTU,论文 arXiv 2402.17152。所有步骤对齐该 repo + 论文。

## 命题（要证的东西）
经典 ML 时序预测 = **固定输入长度 → 固定输出长度**,每个 (输入,输出,任务) 要单独训一个模型。
我们 = **一个生成式模型,任意输入长度、任意输出长度**(自回归 rollout),一个模型覆盖所有 horizon。
卖点不是「架构更准」,是 **1 个模型 vs N 个专用模型**,且追平/打过 SOTA。

## 两个 bucketize（都必须有,用户强调）
1. **值 → token(我们自己定义的 vocab)**：把每个连续遥测目标值按**分位数**桶化成 K 个 bin,每个 bin = 一个 token。K 是我们的设计选择(用 **256 或 1024**,不是 32——32 太粗,反量化误差大,对 SOTA 不公平)。这就是「自己定义 vocab / 自己定义 token」。反量化时 bin → bin 中位值。
   - 可选:categorical 事件(如 CIC attack 类)直接是 token;数值特征也可各自桶化进一个统一 vocab(cross-feature),但先做单变量目标 tokenize。
2. **时间差 → 时间偏置(rab^{p,t} 的 t 部分)**：HSTU 的 `RelativeBucketedTimeAndPositionBasedBias`。桶化函数照抄 repo:`bucket = (log(|Δt|.clamp(min=1)) / 0.301).long()`,clamp 到 [0,128],128 桶。加到位置偏置上。**论文 Table 5 消融显示这一项是 HSTU 打赢 Transformer 的关键;之前我漏了它,所以 HSTU 和 Transformer 打平——补上很可能拉开。**

## HSTU 层（对齐 repo + 论文 Eq1-3,已核对）
- Eq1 `U,V,Q,K = SiLU(f1(layernorm(X)))` ✓
- Eq2 `A·V = (SiLU(QKᵀ + rab^{p,t}) / N) · V`（SiLU 归一除以序列长度 N,**非 softmax**）✓
- Eq3 `Y = f2(U ⊙ Norm(A·V))`（U-gating + 残差）✓
- 损失:自回归 next-token,小词表用 full cross-entropy(= sampled-softmax 精确版)✓

## 数据（4 域,单变量目标序列 + 时间戳 + 实体分组）
| 域 | 数据集 | 实体 | 目标(tokenize) | 时间戳 |
|---|---|---|---|---|
| DC | BurstGPT | 单流 | response tokens | Timestamp |
| HPC | F-DATA/Fugaku | per-user | job duration | adt(到达) |
| NET | CESNET-TS24 | per-IP | n_bytes | id_time(小时) |
| SEC | CIC-IDS2017 | per-source_ip | attack class(原生离散) | Timestamp |

## 评估
- **实体留出**(训练实体 / 测试全新实体),检验 FM 跨实体泛化,并避免随机划分泄漏。
- **多 horizon**:h ∈ {1,2,4,8,16,...}。我们的模型自回归 rollout 到所有 h;固定-horizon baseline 每个 h 训一个。
- **指标 = MASE + MAE**(连续,反量化后)+ 分类任务用 macro-F1。所有模型同一指标同台比。
- **headline**:MASE-vs-horizon 曲线 + 「覆盖所有 horizon 需要的模型数:我们=1,专用=N」。

## Baseline 三档（含 SOTA,用户强调"别忘了合适的 SOTA"）
- **SOTA one-model-any-horizon(直接对手)**：Chronos、TimesFM、Moirai 零样本(它们也 tokenize+生成/任意 horizon,是我们最该比的)。库已装:chronos-forecasting / uni2ts。
- **SOTA 专用预测器(固定 horizon 那派)**：PatchTST、DLinear、N-HiTS(neuralforecast 训)。库已装:neuralforecast。
- **经典**：persistence、seasonal-naive、每 horizon 一个的 GBM。

## 实现顺序
1. tokenizer 模块(值→K-bin vocab + 反量化)+ 时间戳穿过 adapters。
2. 时间感知 HSTU(两个 bucketize 都进去)+ 时间 Transformer 对照。
3. 多 horizon rollout + MASE。
4. 接 Chronos/Moirai/TimesFM 零样本 + PatchTST/DLinear/NHITS 训练。
5. 一张总表:每 horizon 的 MASE,我们 1 模型 vs 各 SOTA。

## 机器/环境
lab (RTX 4090/24GB, WSL Ubuntu20.04), venv `~/nfm/.venv` (torch2.6+cu124, chronos/neuralforecast/uni2ts 已装)。数据 `~/nfm/data/`(BurstGPT×3、F-DATA 多月、CESNET sample、CIC 全量)。
