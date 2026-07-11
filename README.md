# All-in-One: an HSTU Foundation Model for Network / Systems Telemetry

One generative sequential-transduction backbone (HSTU, arXiv:2402.17152) pretrained autoregressively
on system telemetry, evaluated as: (i) a **any-input / any-output-length forecaster** that beats
per-horizon specialists and a general time-series foundation model (Chronos), and (ii) a **frozen
backbone + swappable head** transferred to heterogeneous downstream tasks at near-zero finetune cost.

## Key finding

- **Irregularly-timed event streams** (job/task logs, variable inter-arrival gaps): HSTU is the best
  forecaster, leading a matched Transformer, LSTM/GRU, a per-horizon GBM, and zero-shot **Chronos**,
  with the margin largest at long horizons. The edge comes from HSTU's **time-aware relative bias**.
- **Uniformly-sampled telemetry** (fixed grid): HSTU still beats Chronos + GBM but only ties the
  deep sequence baselines, because a constant sampling interval makes the time bias uninformative.
- **Transfer**: a frozen pretrained backbone + a ~0-parameter head beats a full from-scratch model
  on a distinct classification task (failure prediction, AUC/AUPRC), and honestly fails to help on a
  target uncorrelated with pretraining (node-count regression).

## Datasets

Five public datasets, three "event stream" + two "uniform telemetry". Download into `~/nfm/data/<name>/`.

| name | domain | type | entity → target | timestamp | download |
|---|---|---|---|---|---|
| **fdata** | HPC | event | per-user job → **duration** | job arrival | Zenodo `11467483`, monthly parquet `YY_MM.parquet` |
| **borg** | DC | event | per-user task → **duration** (pair SCHEDULE/FINISH) | µs event time | Google `clusterdata-2011-2/task_events/part-*.csv.gz` |
| **alibaba** | DC | event | per-job batch-task → **duration** | task start_time | Alibaba OSS `v2018Traces/batch_task.tar.gz` |
| **azure** | DC | uniform (5-min) | per-VM → **avg CPU** | 5-min grid | GitHub `AzurePublicDataset` release `dataset-v1`, `...vm_cpu_readings-file-N-of-125.csv.gz` |
| **m100** | HPC | uniform (15-min) | per-node → **PSU power** (`ps0_input_power_avg`) | 15-min grid | Zenodo `7541722`, per-rack `N.tar` → node parquets |

### Exact download commands
```bash
D=~/nfm/data

# fdata (Fugaku HPC jobs) -- get several months for scale (>=2GB)
mkdir -p $D/fdata && cd $D/fdata
for m in 21_04 21_05 21_06 21_07 21_08 21_09 21_10 21_11 21_12 22_01 22_02 22_03 22_04 22_05; do
  wget -O $m.parquet "https://zenodo.org/api/records/11467483/files/$m.parquet/content"; done

# borg (Google 2011 task_events) -- 200 parts ~8GB uncompressed
mkdir -p $D/borg && cd $D/borg
for i in $(seq 0 199); do p=$(printf "part-%05d-of-00500.csv.gz" $i);
  wget -q "https://storage.googleapis.com/clusterdata-2011-2/task_events/$p"; done

# alibaba (2018 batch_task) -- 14.3M tasks
mkdir -p $D/alibaba && cd $D/alibaba
wget "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz" && tar xzf batch_task.tar.gz

# azure (VM 2017 CPU) -- several files for scale
mkdir -p $D/azure && cd $D/azure
for n in 1 2 3 4 5 6 7 8; do
  wget -O vm_cpu_$n.csv.gz "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1/trace_data_vm_cpu_readings_vm_cpu_readings-file-$n-of-125.csv.gz"; done

# m100 (Marconi100 node power) -- several racks
mkdir -p $D/m100 && cd $D/m100
for r in 0 1 2 3 4 5 6; do wget -O $r.tar "https://zenodo.org/api/records/7541722/files/$r.tar/content" && tar -xf $r.tar && rm $r.tar; done
```
All are open (CC-BY / public), no login. Adapters live in `nfm_data.py` (one function per dataset, returning per-entity `(value_seq, timestamp_seq)`).

## Environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # cu124 matches driver <=12.4
uv pip install numpy pandas pyarrow scikit-learn chronos-forecasting
```
Note: installing extra TS libraries may upgrade torch and break CUDA; keep `torch==2.6.0+cu124`.
GPU used: one RTX 4090 (24 GB), bf16.

## How to run

**1. Forecasting comparison table** (Table 1 in the paper) — one dataset at a time:
```bash
# env vars: K vocab, CTX context, HMAX max horizon, D/H/L model, EP epochs, CHRONOS 0/1, SEASON period
K=1024 CTX=32 HMAX=16 D=256 H=4 L=4 EP=10 CHRONOS=1 python run_horizon.py fdata   # or borg/alibaba/azure/m100
```
Prints per-horizon MASE for persistence, ctx-mean, per-horizon GBM, LSTM, GRU, Transformer, HSTU, Chronos,
plus HSTU-faithful next-token ranking (HR@K / NDCG@K / MRR).

**2. Full FM pipeline demo** (pretrain → freeze → head-swap finetune → 3 downstream tasks):
```bash
D=256 L=4 PRE_EP=10 FT_EP=6 CHRONOS=1 python demo_pipeline.py    # F-DATA: forecast + failure-cls + nodecount-reg
```

## Files
| file | what |
|---|---|
| `nfm_core.py` | HSTU backbone (SiLU attn, U-gate, time+position `rab`), `backbone()/freeze_backbone()/swap_head()`; Quantizer tokenizer; Transformer/LSTM/GRU baselines; `train_gen`, median `rollout`, `mase`, `rank_metrics` |
| `nfm_data.py` | dataset adapters: `fdata/borg/alibaba/azure/m100/beam` → per-entity (values, timestamps) |
| `run_horizon.py` | multi-horizon forecasting runner + all baselines + Chronos, MASE + ranking |
| `baselines_sota.py` | GBM-regressor (per-horizon) + Chronos-Bolt zero-shot |
| `demo_pipeline.py` | pretrain backbone → freeze → finetune head for forecasting/classification/regression |
| `pretrain.py` | cross-domain pretraining scaffold (shared-vocab scale tokenizer) — optional |
| `methods_design.md` | design notes / correspondence to the HSTU paper |

## Method correspondence (HSTU paper, arXiv:2402.17152)
Attention `SiLU(QKᵀ + rab^{p,t})/N` (not softmax), U-gating, SiLU pointwise projection, and the
`RelativeBucketedTimeAndPositionBasedBias` (position + log-bucketed time gaps) are reproduced in
`nfm_core.py`. Autoregressive next-token training with full softmax over the 1024-bin vocab is the
unsampled equivalent of the paper's sampled-softmax retrieval loss for a small vocabulary.
