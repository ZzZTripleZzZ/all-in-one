# Data

Adapters in `nfm_data.py` turn each raw trace into per-entity `(value_seq, timestamp_seq)` tuples, one
function per dataset. Raw downloads go in `raw/`, processed artifacts in `processed/`, and both are
git-ignored: only this README, the adapters, and the folder skeleton are tracked. Entry scripts read
traces from `~/nfm/data/<name>/`, so that is the layout the commands below produce.

Nothing here is redistributed. Every trace is public and downloaded from its original source under its
own license.

---

## Traces used in the paper

| name | role | domain | sampling | entity → target | source |
|---|---|---|---|---|---|
| **fdata** | Decision 1, backfilling replay | HPC | irregular, per job | per-user job → duration | Zenodo `11467483`, monthly `YY_MM.parquet` |
| **cesnet** | Decision 2, provisioning | network | hourly grid | per-institution → `n_bytes` | Zenodo `13382427`, `institutions/agg_1_hour/` |
| **azure** | cross-domain pretraining source | cloud | 5-min grid | per-VM → avg CPU | `AzurePublicDataset` release `dataset-v1` |

## Traces used in the supporting runs

| name | role | domain | sampling | entity → target | source |
|---|---|---|---|---|---|
| **m100** | pretraining scale, bake-off | HPC | 15-min grid | per-node → PSU power | Zenodo `7541722`, per-rack `N.tar` |
| **borg** | pretraining scale, bake-off | datacenter | irregular, µs event time | per-user task → duration | Google `clusterdata-2011-2/task_events/` |
| **alibaba** | bake-off | datacenter | irregular, task start time | per-job batch task → duration | Alibaba OSS `v2018Traces/batch_task.tar.gz` |

---

## Download

```bash
D=~/nfm/data

# fdata (Fugaku job logs) -- one month is enough for the quick start; more months for the scale sweep
mkdir -p $D/fdata && cd $D/fdata
for m in 21_04 21_05 21_06 21_07 21_08 21_09 21_10; do
  wget -O $m.parquet "https://zenodo.org/api/records/11467483/files/$m.parquet/content"; done

# cesnet (CESNET-TimeSeries24, institution-level hourly) -- needs institutions/ and times/
mkdir -p $D/cesnet && cd $D/cesnet
wget -O institutions.tar.gz "https://zenodo.org/api/records/13382427/files/institutions.tar.gz/content" && tar xzf institutions.tar.gz
wget -O times.tar.gz        "https://zenodo.org/api/records/13382427/files/times.tar.gz/content"        && tar xzf times.tar.gz

# azure (VM 2017 CPU readings) -- cross-domain pretraining source
mkdir -p $D/azure && cd $D/azure
for n in 1 2 3 4 5 6 7 8; do
  wget -O vm_cpu_$n.csv.gz "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1/trace_data_vm_cpu_readings_vm_cpu_readings-file-$n-of-125.csv.gz"; done

# m100 (Marconi100 node power)
mkdir -p $D/m100 && cd $D/m100
for r in 0 1 2 3 4 5 6; do wget -O $r.tar "https://zenodo.org/api/records/7541722/files/$r.tar/content" && tar -xf $r.tar && rm $r.tar; done

# borg (Google 2011 task_events) -- 200 parts, ~8 GB uncompressed
mkdir -p $D/borg && cd $D/borg
for i in $(seq 0 199); do p=$(printf "part-%05d-of-00500.csv.gz" $i);
  wget -q "https://storage.googleapis.com/clusterdata-2011-2/task_events/$p"; done

# alibaba (2018 batch_task) -- 14.3M tasks
mkdir -p $D/alibaba && cd $D/alibaba
wget "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz" && tar xzf batch_task.tar.gz
```

All open access, no login required.

---

## Preprocessing

Only CESNET needs a pass before use. It is reindexed onto the full hourly grid, missing bins are marked
invalid rather than interpolated, and institutions with under 50% coverage are dropped.

```bash
python data/prep_cesnet.py     # -> ~/nfm/data/cesnet/cesnet_hourly.npz  {r, valid, ts, sid}
```

The other traces are read directly by their adapters. F-DATA is **not** in Standard Workload Format; the
backfilling replay reads the parquet columns it needs (`usr`, `adt` submit time, `nnumr` nodes,
`elpl` user-declared limit, `duration` true runtime) and enforces `elpl >= duration` for trace
consistency, so no job outlives its own limit.

---

## Adding your own trace

Write one function in `nfm_data.py` that returns a list of `(values, timestamps)` pairs, one per entity,
in strictly increasing time order, and register it in `LOADERS`. Values are the quantity a decision
consumes, timestamps are seconds on the entity's own clock. Irregular gaps are expected, not a problem:
the gap is a model input.

---

## Exploratory datasets

`prep_loghub.py`, `prep_lanl*.sh`, and the `sft/run_security_*.py` runners target log-anomaly and
security-event corpora (Loghub BGL/Thunderbird, LANL Unified Host and Network 2017). These were part of
the earlier exploration and are **not** used in the paper. They are kept because the adapters are
reusable and the campaign log is more useful complete than curated.
