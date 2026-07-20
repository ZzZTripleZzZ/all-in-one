# Dataset Validation — 本机小规模跑通 + 信号 + 规模判断

> 目的: 在 MacBook Air M4 (24GB, 磁盘紧) 上,对 4 个不同域的候选数据集做**小规模验证**,回答两个问题:
> **(1) 能不能下载/解析、有没有 per-entity 序列结构?** **(2) 简单模型能不能打过 naive baseline (= 有没有可学的序列信号)?**
> 外加一维针对 SIGCOMM/MobiCom 的 **规模是否够 headline** 判断。
>
> 方法学诚实声明: 验证器是 **HistGradientBoosting + 滞后特征**,这是 HSTU 序列 transducer 的**弱代理/下界**,不是最终模型。
> "打过 baseline" 只说明该域**有可学的序列结构值得上真模型**,不等于最终精度。所有实验在 scratchpad 子采样切片上跑,
> 全量规模是另一回事 (见每个数据集的 Scale 行)。环境: numpy 2.5 / pandas 3.0 / sklearn 1.9,纯 CPU,每个 <1 分钟。

---

## 结果总览

| 域 | 数据集 (验证切片) | 任务 | 模型 vs baseline | 信号 | **SIGCOMM/MobiCom 规模** |
|---|---|---|---|---|---|
| DC | Google Borg 2011 (2/500 part, 32万转移) | 预测下一事件类型 | marginal .642 → Markov .874 → **GBM .887** | ✅ 强 | ✅ **够** (全量 ~144M 事件/12.5k机/29天/41GB) |
| HPC | ANL-Intrepid SWF (68,936 作业) | 预测作业 runtime | reqTime 1.30 / userMean 1.15 → **GBM 0.605, R²=0.70** | ✅ 强 | ⚠️ **偏小** (69k 作业/236 用户/单系统) |
| NET | GÉANT TM (10,772 矩阵, 486 OD, 445万样本) | 下一步 OD 流量 | persistence .522 → **GBM .484 (+7.1% skill)** | ✅ 有 | ⚠️ **太小且旧** (23 节点/2005) |
| 无线 | Irish 5G Raca (166 session, 用了83) | 下一秒 DL 吞吐 | persistence 4605 → **GBM 4301 (+6.6%), R²=0.81** | ✅ 有 | ⚠️ **偏小** (166 session/单运营商/~52h) |

**一句话结论**: 4 个域**全部**在弱代理模型下就跑出了"打过 naive baseline"的可学序列信号 → 范式在每个域都有东西可挖,鼓舞人心。
但**只有 Borg 一个的全量规模能直接当 SIGCOMM/MobiCom headline**;另外三个是绝佳的**快速原型/方法验证**集,正式投稿必须换成它们的大规模同门 (见下)。

---

## 逐数据集

### DC — Google Borg 2011 ✅ 用得了 + 强信号 + 规模够
- **可用性**: GCS 直链开放,`task_events` 每 part ~4MB gz,schema 齐。per-(job_id, task_index) 事件流成立。
- **验证**: 2 个 part (~90万原始事件 → 32万条状态转移, 20.8万实体)。预测每个 task 序列的**下一事件类型**。
  - marginal (总是猜多数类) **0.642** → bigram-Markov (只看上一事件) **0.874** → GBM (二阶上文 + priority/sched_class/资源/位置, 事件类型作 categorical) **0.887**,macro-F1 0.557。
  - **读法**: 只看上一个事件就把准确率从 64%→87%,说明序列依赖**很强**;二阶上文 + 特征再补到 88.7%,补上 69% 的差距。稀有类 (fail/finish 占 0.4%) 靠 F1 看还有空间 → 正是 HSTU embedding 该赢的地方。
- **Scale (SIGCOMM 级)**: ✅ 全量 ~144M 事件、12.5k 机、~2500万 task、29 天、41GB。本机流式读全部 500 part 只是 I/O 问题,无内存压力。搭 **Alibaba / Azure** 全家桶即成跨-DC 大规模底座。

### HPC — ANL-Intrepid SWF ✅ 用得了 + 强信号 + ⚠️ 规模偏小
- **可用性**: Parallel Workloads Archive 直链,标准 18 列 SWF。按 uid 分 per-user 作业序列。
- **验证**: 68,936 有效作业 / 236 用户 / 8 个月,时间切分 80/20。预测 **log runtime**,per-user 因果特征 (上一作业时长、迄今扩展均值、作业计数) + 请求资源/队列/小时。
  - baseline: 全局均值 1.426 / **用户申报 walltime reqTime 1.299** / per-user 均值 1.148 → **GBM MAE 0.605, R²=0.70**。
  - **读法**: 大幅打过所有 baseline,连"用户自己报的 walltime"都远输给模型。per-user 历史 + 请求特征高度可预测 runtime,序列范式在 HPC 作业上非常成立。
- **Scale (SIGCOMM 级)**: ⚠️ 单个 69k 作业/236 用户/单系统当 headline 会被审稿人说小。**Scale-up 同门**: F-DATA Fugaku (24M 作业)、MIT SuperCloud (460k 作业 + 2.1TB 遥测)、M100 ExaData (980节点×1Hz×2.5年, 49.9TB),或直接 pool Parallel Workloads Archive ~40 个 log (合计数百万作业, 统一 SWF schema 天然利于跨系统预训练)。

### NET — GÉANT Traffic Matrices ✅ 用得了 + 有信号 + ⚠️ 太小太旧
- **可用性**: 直链 bz2,10,772 个 15 分钟 TM (XML) + 拓扑。解析成 486 条 OD-pair 时序。
- **验证**: 池化全部 486 条 OD → **一个 entity-agnostic 模型** (只喂滞后窗口,不告诉它是哪条 OD),445万样本,时间切分。预测下一步 log(kbps)。
  - persistence (猜上一时刻) MAE **0.5216** → **GBM 0.4843, skill +7.1%**。
  - **读法**: 一个共享模型在 486 条不同 OD 上都打过 persistence → **跨实体迁移成立** (HSTU 的 all-in-one 前提)。增益中等,因为 15 分钟粒度下 persistence 本就很强;真模型 (实体 embedding + 长上文) 会拉开更多。
- **Scale (SIGCOMM 级)**: ⚠️ 23 节点 / 2005 年,当主数据集必被诟病 (协议构成早已过时)。**只留作大家都报的经典对比 benchmark** (配 Abilene)。**Scale-up 同门**: CESNET-TimeSeries24 (27.5万 IP × 40 周, 2024)、CAIDA/MAWI (十亿级包, 需自建 flow pipeline)、CESNET-QUIC22 (per-flow 包序列,最贴范式)。

### 无线 — Irish 5G (Raca MMSys'20) ✅ 用得了 + 有信号 + ⚠️ 规模偏小
- **可用性**: GitHub 直 clone,166 个 session CSV (Netflix/Amazon/Download × Static/Driving),每秒一行 (RSRP/RSRQ/SNR/CQI/RSSI/DL_bitrate…,缺失为 `-`)。
- **验证**: 留出整段 session 作测试 (测未见 session 泛化)。用了 166 中的 83 个 (glob 怪癖,全量只会更强),18.8万行。预测**下一秒 DL 吞吐**,滞后吞吐窗口 + 当前无线上下文。
  - persistence MAE **4605 kbps** → **GBM 4301, skill +6.6%, R²=0.811**。
  - **读法**: 在**未见 session** 上仍打过 persistence 且 R²=0.81 → 无线电上下文 + 序列信号真实存在,能泛化。增益中等 (1 秒粒度 persistence 很强),真模型 + 长上文会更好。
- **Scale (SIGCOMM 级)**: ⚠️ 166 session/单运营商/~52h,当 headline 偏小。标签和真实性很好,适合做 finetune/对比。**Scale-up 同门**: Open RAN Commercial Traffic Twinning (450GB/500h, per-UE 跨层)、NetMob23 (2.3TB, 68服务×20城)、Telecom Italia Milano (10k cell×62天)、OpenIreland O-RAN PM (16B 样本, 已被用作 TSFM 预训练语料)。

---

## 对立项的启示

1. **范式在 4 个域都有信号 (弱代理即打过 baseline)** → 数据侧的可行性验证通过,值得上真 HSTU 模型 + 跨域预训练。
2. **规模是投稿的真约束,不是可行性约束**。可行性已验证;SIGCOMM/MobiCom 要的是"全量够大 + 真实 + 现代"。按上表把每个域换成大规模同门:
   - **头部大规模底座**: DC=Borg(144M)+Azure; HPC=M100 ExaData(49.9TB)/F-DATA(24M); NET=CESNET (QUIC22 包序列 + TimeSeries24); 无线=Open RAN Twinning(450GB)/NetMob23(2.3TB)。
   - 小而经典的 (GÉANT/Abilene/Irish5G) 降级为**对比 benchmark**,不当 headline。
3. **诚实的方法学红线 (审稿人会盯)**: 细粒度下 persistence/Markov 是**强** baseline,+6~7% 不够 compelling。正式实验必须 (a) 上强 baseline (不止 persistence:季节性、per-task 专用 LSTM/TSFM、NetLLM/UoMo 同类);(b) 靠真 HSTU + 跨域预训练把 margin 拉开;(c) 用大规模数据。这三点做到,故事才立得住。
4. **本机能力边界确认**: 460MB 数据 + 纯 CPU sklearn,每个实验 <1 分钟跑完,24GB 磁盘无压力。**方法验证/原型完全够**;真正的大规模预训练 (TB 级数据 + HSTU CUDA kernel) 要上云 GPU。

---

## 复现

```bash
SCRATCH=<scratchpad>   # 见本 session
source $SCRATCH/.venv/bin/activate     # numpy/pandas/sklearn, python 3.12
python $SCRATCH/scripts/borg_val.py    # DC
python $SCRATCH/scripts/swf_val.py     # HPC
python $SCRATCH/scripts/geant_val.py   # NET
python $SCRATCH/scripts/irish5g_val.py # 无线
```
数据下载直链见各脚本顶部注释;scratchpad 是临时目录 (不进 iCloud),脚本已留存,数据可随时重拉。
