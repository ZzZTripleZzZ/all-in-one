# Dataset Scouting — Sequential-Transduction Foundation Model for Network/System Prediction

> 立项一句话: 借 HSTU / generative-recommenders (arXiv 2402.17152) 的 **sequential transduction + pretrain/finetune** 范式,
> 把 datacenter / HPC / network / wireless 里**各种系统预测任务**统一成"给定实体的历史事件序列,生成下一个事件/测量",
> 用**一个跨域基础模型**替代"一个任务训一个模型"的旧范式。范围不限于 flow volume 预测。
>
> 本文件 = 5 个并行 agent (4 域 + 1 定位) 的搜索结果,去重交叉核对后的汇总。所有 URL 由 agent 本 session 抓取核对;
> 标 ⚠️ 的是访问受限/链接存疑,承诺前需再确认。日期: 见 git 提交。

---

## 0. 选型判据 (为什么是这些,不是随便的时序数据)

这套范式对数据集有硬要求,评分 (Fit 1–5) 按此打:

1. **per-entity 事件序列** — 实体 (flow / host / VM / job / node / link / cell / UE / rank) 随时间有一串测量或事件。这是"用户→动作序列"的映射。
2. **异构特征可 tokenize** — categorical (ID / 事件类型 / 优先级 / band) + numerical (util / bytes / latency / RSRP)。
3. **规模够 pretrain** — 长序列 × 多实体。
4. **能挂多个下游任务** — 一个数据源同时喂 pretrain 和多个异构 finetune head,all-in-one 才成立。
5. **可获取 + license 清楚**。

**贯穿所有域的规律**: 每域 fit 最高的,都是"实体本身已是带时间戳事件流"的数据 (Borg `task_events` / Loghub 日志 / CESNET-QUIC22 包序列 / OpenRAN per-UE KPM)。这些无需造 pipeline,直接是 HSTU token 序列。

---

## 1. 跨域首选清单 (Fit 5,建议优先落地)

| 域 | 数据集 | 实体 = 序列 | 规模 | 获取 | 链接 |
|---|---|---|---|---|---|
| DC | **Google Borg 2011** (`clusterdata-2011-2`) | per-task 事件流 (SUBMIT→SCHEDULE→EVICT/FAIL/FINISH) + 5min usage | 12.5k机 / 20M task / 29天 / 41GB | 开放 (GCS) | https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md |
| DC | **Azure Public Dataset 全家桶** (VM17/19, Functions, LLM-inf) | per-VM util 序列 / per-function 调用序列 / per-request token 序列 | VM≈2.6M实体×1.9B读数 | 开放 (GitHub Releases) | https://github.com/Azure/AzurePublicDataset |
| DC | **BurstGPT** (LLM serving) | per-request 到达+token 序列 | 10.31M 请求 / 213天 | 开放 | https://github.com/HPMLL/BurstGPT |
| HPC | **M100 ExaData** (Marconi100) | per-node 573指标 @1Hz | 980节点 × 2.5年 / 49.9TB / Parquet | 开放 CC-BY (Zenodo) | https://www.nature.com/articles/s41597-023-02174-3 |
| HPC | **MIT SuperCloud** | per-job/per-node 多采样率 (100ms GPU / 10s CPU / 5min node) | 460k job / 2.1TB | 开放 (AWS/Kaggle) | https://dcc.mit.edu/ |
| HPC | **Loghub** (BGL/Thunderbird/Spirit) | per-node 带时间戳日志事件流 (最贴 next-event) | 上亿行 / 带 alert 标签 | 开放 | https://github.com/logpai/loghub |
| NET | **CESNET-QUIC22 / TLS-Year22** ★ | **per-flow 已自带前N包 {size,dir,IAT} 序列** | 153M flow / 满一年 | 开放 (Zenodo) | https://zenodo.org/records/10728760 |
| NET | **CESNET-TimeSeries24** | per-IP/机构 volume 时序 (多粒度) | 275k+ IP / 40周 | 开放 (Zenodo) | https://arxiv.org/abs/2409.18874 |
| WL | **Open RAN Commercial Traffic Twinning** ★ | per-UE 跨层 KPM CSV (一 UE 一文件) | >500h / 450GB / 30配置 | 开放 CC-BY-SA (size 是唯一摩擦) | https://openrangym.com/datasets/open-ran-commercial-traffic-twinning-dataset |
| WL | **Colosseum O-RAN COMMAG** | per-UE per-experiment KPM 序列 (~30 KPM) | 4BS/40UE/3slice | 开放 | https://github.com/wineslab/colosseum-oran-commag-dataset |
| WL | **Telecom Italia Milano/Trentino** | per-cell 5通道时序 | 10k cell × 5通道 / 62天 | 开放 ODbL | https://doi.org/10.7910/DVN/EGZHFV |
| WL | **OpenIreland O-RAN PM** | per-UE RAN PM @1ms + 丰富标签 (已被用作 TSFM 预训练语料) | ~16B 样本 / 47特征 | 开放 CC-BY (Mendeley) | https://data.mendeley.com/datasets/t2rzh9y4mp/1 |
| WL(CSI) | **DICHASUS** | per-trajectory 时序 CSI + 真值3D位置 | ~14 set / 32天线 / 100k+样本/set | 开放 (用 DaRUS 镜像 ⚠️主站证书告警) | https://darus.uni-stuttgart.de/dataverse/dichasus |
| WL(CSI) | **DeepSense 6G** | per-link 时序 64波束功率 + GPS/相机/雷达/LiDAR,自带 future-beam 任务 | 40+场景 | 免费注册 | https://www.deepsense6g.net/ |

★ = 跨全部域里 fit 最突出、最"开箱即用"的两个。

---

## 2. 各域完整候选 (含 finetune/eval 用的次选)

### 2.1 Datacenter / Cloud
- **Fit5**: Borg 2011; Azure 全家桶; BurstGPT。
- **Fit4**: Borg 2019 (更全但 BigQuery-only, 2.4TiB, 直方图 usage); Alibaba v2018 (机器+容器时序 + batch-task DAG); Azure Functions 2019/2021 (per-function 调用序列); Philly GPU trace (少见的 per-minute GPU util); Helios (1.58M job 生命周期); Acme (LLM 集群多模态); Alibaba PAI-GPU-v2020 (job/task/instance 层级, ⚠️ util 是生命周期均值); Azure-LLM-Inference 23/24; Mooncake (含 KV-cache block hash 序列)。
- **打包好的先例/baseline**: **Salesforce `cloudops_tsf`** (HF) = Azure+Borg+Alibaba 已切成预测窗口的**现成 cloud-trace 预训练 benchmark** → 当"要打败的先例"和快速起步 shard。 https://huggingface.co/datasets/Salesforce/cloudops_tsf
- **缺口**: DC 网络级 per-flow 遥测公开数据很弱 (Facebook 2010 太旧且 ⚠️ 疑似失效)。

### 2.2 HPC / Interconnect
- **Fit5**: M100 ExaData; MIT SuperCloud。
- **Fit4**: F-DATA (Fugaku 24M job, CC-BY, per-user job 序列); ORNL Summit 遥测 (1Hz power/thermal + GPU XID 故障事件, ⚠️ Globus 传输); ALCF Data Catalog (2008–2025 多系统 job+RAS, ⚠️ IEEE DataPort 订阅); Loghub。
- **Fit3–4**: Darshan I/O 日志 (百万级 per-job I/O 摘要); LANL failure + USENIX CFDR (canonical 故障事件, 稀疏); ATLAS Trinity/Mustang (per-job queue-wait); Parallel Workloads Archive (SWF, 几十个系统统一 schema, 老)。
- **唯一公开的真 interconnect 序列**: **DOE Design Forward / DUMPI** per-rank MPI 通信事件序列。 https://portal.nersc.gov/project/CAL/designforward.htm
- **缺口 (与 NET 域重合)**: 生产 router/link counter 时序 (dragonfly/Aries/Slingshot) **不公开** — Cori/Perlmutter LDMS 只有论文无数据下载。这是"interconnect 流量序列"最硬的空白。

### 2.3 Network Traffic / Flow
- **两类要分清**: (A) forecasting = per-link/OD/IP volume 时序; (B) packet/flow = per-flow 包/连接序列。
- **Fit5**: CESNET-QUIC22/TLS-Year22 (B类,自带包序列); CESNET-TimeSeries24 (A类,275k per-entity)。
- **Fit4**: CAIDA Anonymized Traces (⚠️ 申请表, 骨干真流量, 需自建 flow pipeline); MAWI/MAWILab (开放, 2001至今每日15min + 异常标签); UGR'16 (17B NetFlow, per-host 连接序列 + 攻击标签); Abilene TM (2004, 144 OD-pair, **TM 预测经典 benchmark**); GÉANT TM (2005, 529 OD-pair)。
- **IDS 类 (synthetic/短, 只做 finetune/eval)**: CIC-IDS2017/DDoS2019, UNSW-NB15, UQ NetFlow-v2 (统一43字段 schema,利于跨集迁移), Kitsune。**NSL-KDD 不是序列,别用**。
- **缺口**: 没有同一网络**同时**给 raw 包序列 + OD/TM 矩阵的公开集; DC 侧无 FCT (flow-completion-time) 标签的开放 benchmark; 匿名化会打断跨天 per-host 连续性。

### 2.4 Wireless / Cellular
- **Fit5**: Open RAN Commercial Traffic Twinning; Colosseum O-RAN COMMAG; Telecom Italia Milano; OpenIreland O-RAN PM; Irish 5G (Raca, per-秒 RSRP/RSRQ/SNR/CQI/throughput); Shanghai Telecom (per-UE 接入事件流, 结构最贴 HSTU); NetMob23 (2.3TB, 68服务×20城, ⚠️ 申请门); DICHASUS; DeepSense 6G; GeoLife / T-Drive (per-user 轨迹, 若把 mobility 也纳入)。
- **Fit4**: Lumos5G (mmWave per-run); Irish 4G; Salzburg 高速4G; Vienna 4G/5G; C2TM (per-BS hourly, 仅8天); ColO-RAN (per-slice); Raymobtime/Argos/Sionna-RT (合成 CSI 序列/引擎); MDC-Nokia (mobility+cell context, ⚠️ 重度 gated)。
- **⚠️ 死链/存疑, 别引**: Orange D4D (portal 已废); "Cadvise" (**不存在此数据集**); Microsoft Spectrum Observatory (legacy); Electrosense API (2026 状态未确认); WAIR-D / NYU-METS / NYCU-5G (主下载链未核实); DeepMIMO/WAIR-D 是**快照非序列** (只能轨迹拼接)。

---

## 3. 推荐的跨域 all-in-one 组装方案

按"一个 backbone 预训练 + 多个域/任务 finetune head"来搭。原则: **每域挑 1 个开放、原生事件序列的主力做 pretrain,再挂该域经典 benchmark 做 finetune/eval。**

**Pretrain 语料池 (全开放, 尽量原生事件序列)**
- DC: Borg 2011 (`task_events`) + Azure VM/Functions
- HPC: M100 ExaData (node 遥测) + Loghub (日志事件) + F-DATA (job)
- NET: CESNET-QUIC22 (flow 包序列) + CESNET-TimeSeries24 (IP volume)
- WL: Open RAN Twinning + Colosseum O-RAN + Telecom Italia Milano

**Finetune / eval head (每域挂典型任务)**
- 故障/抢占: CFDR·LANL / Summit XID / Borg EVICT / Helios·Acme status
- 时长/排队: F-DATA · ATLAS · PWA · Borg
- 资源/功耗/热: M100 · Summit · Azure VM
- 流量/TM 预测: Abilene · GÉANT · CESNET-TimeSeries24 · Milano
- flow 分类/大小: CESNET-QUIC22 · UGR'16 · CIC/UNSW
- 无线 KPI/吞吐/波束: OpenIreland · Irish 5G · Lumos5G · DeepSense (future-beam)
- I/O / interconnect: Darshan · DUMPI (唯一公开通信序列)

**先例/baseline 必对标**: Salesforce `cloudops_tsf` (cloud-trace 预训练已有人做); 通用 TSFM (Chronos/Moirai/TimesFM/Moirai) 做 zero-shot 对照 (GIFT-Eval / Monash archive)。

---

## 4. Novelty 定位与 gap (来自定位 agent, 核对过 200 篇 HSTU 引用)

**结论: HSTU / generative-recommender 范式尚未被搬到 networking / systems。** HSTU 前 200 篇引用全是推荐/广告/搜索。

**已有 "networking foundation model" 都是单一 silo:**
- packet-traffic: ET-BERT (WWW'22, masked), netFound (masked, 4.2B flow, 安全域), YaTC/NetMamba/TrafficFormer (masked), **NetGPT** (GPT 自回归, 单域, 最接近生成式兄弟), **Lens** (T5 encoder-decoder), TrafficLLM, **NetFlowGen** (flow-as-events 自监督, 但只测 1 任务)。
- mobile-traffic: **UoMo** (KDD'25, 单一 mobile 域 multi-task), **MobiGPT** (3数据×3任务, 单一无线域内最接近 all-in-one)。
- wireless-channel: **LWM** (masked, DeepMIMO), WirelessGPT, MUSE-FM。
- 决策类: **NetLLM** (SIGCOMM'24, 冻结 NL-LLM → 3 networking 任务, 最强 "one model many tasks" claim, 但非从网络数据 from-scratch 预训练, 任务是控制非 next-event)。

**gap (我们可能 first at):**
1. **跨域 all-in-one 真的开放** — 没有模型横跨 DC+HPC+network+wireless。最强、最可辩护的 novelty 轴。
2. **HSTU sequential-transduction 范式本身未被移植** — 把网络/系统预测重构为"高基数、非平稳、streaming 的 entity-event transduction",并借 HSTU 长序列高效算子。
3. **HPC/DC 系统遥测完全没有 FM** — 现在还全是 bespoke LSTM/transformer/RL per task。竞争最少。
4. **生成式 next-event vs 全场 masked-BERT 默认** — 生成式阵营小 (NetGPT/NetFlowGen/Lens) 且单域。
5. **cross-domain 共享 benchmark 不存在** (两篇无线综述都点名) → 自建 benchmark 本身是贡献,也是护城河。

**claim 必须收窄 (先证伪再锁)**: 不能说 "first multi-task networking FM" (NetLLM 已占)。能站住的精确表述:
> **first to (i) 从网络/系统数据 from-scratch 预训练 (不同于 NetLLM 冻结 NL-LLM) + (ii) 用 generative entity-event sequential transduction + (iii) 跨异构域一个模型。**
> 锁 priority claim 前对 NetLLM + MobiGPT + UoMo + netFound 做一遍 falsification。

**可复用的已释放 benchmark**: ISCX-VPN / USTC-TFC / CSTNET-TLS1.3 (ET-BERT); netFound 语料+code (SNL-UCSB/netFound); Telecom-Italia Milano; UoMo 多城 (tsinghua-fib-lab/UoMo); DeepMIMO (LWM base); NetLLM 三任务套件 (含 Alibaba/Borg 调度); M100 (HPC 异常真值)。

---

## 5. 待办 / 需再确认

- [ ] ⚠️ 访问确认: CAIDA (申请表, US-academic 优先, 2–3工作日); NetMob23 / MDC-Nokia (institutional 审批); Facebook DC (疑似失效)。
- [ ] ⚠️ 别引已证伪源: Orange D4D、Cadvise (不存在)、Microsoft Spectrum Observatory、WAIR-D/NYU-METS/NYCU-5G (链未核实)。
- [ ] 2026 年份 arXiv (2601.* / 2602.* / 2604.* / 2606.*) 是极新预印本,入 ref.bib 前用 /ref-check 兜底。
- [ ] 决定 pretrain 是否纳入 mobility (GeoLife/T-Drive) — 扩边界但增异构。
- [ ] **核心设计负担 = 统一 event schema / tokenizer**: 实体跨越 dense numeric (Azure VM)、categorical 事件流 (Borg)、request/token (LLM)、10min cell-traffic vs 1ms O-RAN KPM vs per-frame CSI 的不同 cadence。数据可得性不是瓶颈,**共享词表设计才是真正的研究负担**。
- [ ] 立项要正面处理的最硬空白: **生产 interconnect/router/link counter 时序不公开** (HPC 和 NET 两域都撞到)。要么 stitch/合成,要么向 NERSC/ORNL 申请 LDMS。

---

## 6. 本地已存资料

- `idea/foundation/2402.17152.pdf` — HSTU 原论文 (Actions Speak Louder than Words)
- `idea/foundation/generative-recommenders/` — Meta 官方代码 (浅克隆; 若要跑需 `git submodule update --init --recursive`)
- `idea/foundation/dataset-scouting.md` — 本文件
