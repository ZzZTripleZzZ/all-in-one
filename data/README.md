# Data

Adapters in `nfm_data.py` turn each raw trace into per-entity `(value_seq, timestamp_seq)`
tuples (one function per dataset). Raw downloads go in `raw/`, processed artifacts in
`processed/` — both are git-ignored; only this README and the folder skeleton are tracked.
On the lab box the real data lives under `~/nfm/data/<name>/`.

---

## Pretraining datasets (HPC / DC telemetry)

| name | domain | type | entity → target | timestamp | source |
|---|---|---|---|---|---|
| **fdata** | HPC | event | per-user job → duration | job arrival (irregular) | Zenodo `11467483`, monthly `YY_MM.parquet` |
| **borg** | DC | event | per-user task → duration (SCHEDULE/FINISH) | µs event time (irregular) | Google `clusterdata-2011-2/task_events/` |
| **alibaba** | DC | event | per-job batch-task → duration | task start_time | Alibaba OSS `v2018Traces/batch_task.tar.gz` |
| **azure** | DC | uniform (5-min) | per-VM → avg CPU | 5-min grid | GitHub `AzurePublicDataset` release `dataset-v1` |
| **m100** | HPC | uniform (15-min) | per-node → PSU power | 15-min grid | Zenodo `7541722`, per-rack `N.tar` |

### Download commands
```bash
D=~/nfm/data

# fdata (Fugaku HPC jobs) -- several months for scale (>= 2 GB)
mkdir -p $D/fdata && cd $D/fdata
for m in 21_04 21_05 21_06 21_07 21_08 21_09 21_10 21_11 21_12 22_01 22_02 22_03 22_04 22_05; do
  wget -O $m.parquet "https://zenodo.org/api/records/11467483/files/$m.parquet/content"; done

# borg (Google 2011 task_events) -- 200 parts ~8 GB uncompressed
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
All open (CC-BY / public), no login.

---

## Security transfer datasets (Frame 2 — real attack labels)

These are the cross-domain transfer targets: pretrain on telemetry above, then finetune a
head to detect malicious vs benign behavior. They are enterprise / national-lab scale
(the closest public setting to HPC with real attack labels — no HPC-compute intrusion trace
is public).

| name | year | entity → event | labels | license | link |
|---|---|---|---|---|---|
| **DARPA OpTC** | 2020 | host / process / flow → eCAR record (irregular) | 3 red-team days, malicious/benign | public domain (DARPA) | [github.com/FiveDirections/OpTC-data](https://github.com/FiveDirections/OpTC-data) |
| **LANL Unified Host+Net** | 2017 | computer / user → auth + process events (irregular) | lateral-movement / compromised-credential | CC0 | [csr.lanl.gov/data/2017](https://csr.lanl.gov/data/2017/) |
| **AIT-LDS v2.0** | 2022 | host → auth/audit/dns/vpn/syslog (irregular) | line-level MITRE ATT&CK steps | CC-BY-NC-SA | [zenodo.org/records/5789064](https://zenodo.org/records/5789064) |

Notes:
- **OpTC** is the primary target: native `(host, object, action, timestamp)` tuples at
  foundation-model scale (~17B events, 1000 hosts, ~1 TB). Parsing the eCAR JSON is the main
  engineering cost. Downstream = sequence-level malicious/benign classification on the three
  red-team eval days (23–25 Sep 2019).
- **LANL 2017** is the clean CC0 anchor; pair with the LANL 2015 "Comprehensive" set for
  stronger red-team auth labels.
- **AIT-LDS v2.0** is the most recent (2022) and has per-event ATT&CK ground truth, but its
  license is non-commercial (fine for academic use, blocks commercial).
- A security adapter (`optc()` / `lanl()`) is TODO in `nfm_data.py`; it must emit the same
  per-entity `(event_token_seq, timestamp_seq)` shape as the telemetry adapters so the same
  backbone consumes it.

> Pure flow-table NIDS (CIC-IDS, UNSW-NB15, 5G-NIDD) are **not** used as sequence-pretraining
> showcases — per-entity sequential signal there is weak (persistence saturates them). They
> can only serve as a supervised per-flow classification head.
