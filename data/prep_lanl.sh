#!/bin/bash
# Prefilter LANL 2015 auth.txt.gz (~1B lines) -> a bounded per-user event table with red-team labels.
# Output: ~/nfm/data/lanl/$OUT  with columns: time,user,srccomp,dstcomp,label
# label=1 iff (time,user,srccomp,dstcomp) is a known red-team compromise event (from redteam.txt).
# Keeps ALL events of red-team users (so positives survive) + a ~1/SAMPLE user sample of benign users.
set -e
D=$HOME/nfm/data/lanl
cd "$D"
SAMPLE=${SAMPLE:-20}        # keep ~1/SAMPLE of benign users (by numeric user id)
OUT=${OUT:-lanl_events_s${SAMPLE}.csv}   # SAMPLE-tagged output so different base-rate runs coexist
[ -f redteam.txt ] || zcat redteam.txt.gz > redteam.txt

echo "streaming auth.txt.gz (this reads ~1B lines, a few minutes)..."
# Two-file awk idiom: first file (redteam.txt) builds the label arrays, second file (auth stream via -)
# is filtered+labeled. Avoids getline/array-init quirks. redteam fields: time,user,srccomp,dstcomp.
awk -F, -v OFS=, -v S="$SAMPLE" '
  FNR==NR { RTK[$1","$2","$3","$4]=1; RTU[$2]=1; nrt++; next }   # first file = redteam.txt
  FNR==1 { printf("loaded %d redteam events\n", nrt) > "/dev/stderr" }
  # auth fields: 1=time 2=srcuser 3=dstuser 4=srccomp 5=dstcomp 6=authtype 7=logontype 8=orient 9=success
  {
    u=$2
    # SYMMETRIC filter applied to BOTH classes (fixes the label-correlated-preprocessing confound):
    # human "U#" accounts + successful LogOn only, for red-team AND benign alike. Lateral-movement
    # compromises are themselves successful logons, so malicious signal survives; but red-team and
    # benign sequences now come from the SAME event distribution (no machine-account/LogOff/failure
    # fingerprint that a detector could exploit instead of the actual anomalous-destination signal).
    if (u !~ /^U[0-9]+@/ || $8!="LogOn" || $9!="Success") next
    key=$1","$2","$4","$5; ismal=((key in RTK)?1:0); kp+=ismal
    # keep ALL events of red-team users (label per event) + a 1/S sample of benign users
    if ((u in RTU) || (substr(u,2)+0)%S==0) print $1,$2,$4,$5,ismal
  }
  END { printf("red-team-labeled rows kept: %d\n", kp) > "/dev/stderr" }
' redteam.txt <(zcat auth.txt.gz) > $OUT

echo "rows: $(wc -l < $OUT)"
echo "red-team positive rows: $(awk -F, '$5==1' $OUT | wc -l)"
echo "distinct users: $(cut -d, -f2 $OUT | sort -u | wc -l)"
echo LANL_PREP_DONE
