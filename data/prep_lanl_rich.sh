#!/bin/bash
# RICH-event LANL prep: auth.txt.gz -> per-user event table with the FULL authentication tuple.
# Output: ~/nfm/data/lanl/$OUT  columns: time,user,dst,src,auth,logon,success,label
#   dst=destination computer, src=source computer, auth=auth-type (normalized), logon=logon-type,
#   success=1/0, label=1 iff red-team event. src is kept so the new-edge (unseen src->dst) heuristic
#   baseline and an authentication-graph baseline are computable (both expected by LANL reviewers).
# Motivation: the dst-only representation ties the frequency baseline (gate finding); the discriminative
# lateral-movement signal lives in the JOINT (destination, auth-type, logon-type, outcome) tuple, so we
# keep all four fields and let the generative model score their joint plausibility in context.
# SYMMETRIC filter (both classes): human "U#" accounts + LogOn orientation. Success/Fail is now a FIELD,
# not a filter, so the failed-auth probing signal is preserved without introducing a per-class fingerprint.
set -e
D=$HOME/nfm/data/lanl
cd "$D"
SAMPLE=${SAMPLE:-20}
OUT=${OUT:-lanl_rich_s${SAMPLE}.csv}
[ -f redteam.txt ] || zcat redteam.txt.gz > redteam.txt

echo "streaming auth.txt.gz (rich event extraction, reads ~1B lines)..."
awk -F, -v OFS=, -v S="$SAMPLE" '
  FNR==NR { RTK[$1","$2","$3","$4]=1; RTU[$2]=1; nrt++; next }   # first file = redteam.txt
  FNR==1 { printf("loaded %d redteam events\n", nrt) > "/dev/stderr" }
  # auth fields: 1=time 2=srcuser 3=dstuser 4=srccomp 5=dstcomp 6=authtype 7=logontype 8=orient 9=success
  {
    u=$2
    if (u !~ /^U[0-9]+@/ || $8!="LogOn") next                   # human U# accounts, LogOn orientation
    at=$6; if (at ~ /^MICROSOFT/) at="MSAUTH"; else if (at=="?"||at=="") at="UNK"
    lt=$7; if (lt=="?"||lt=="") lt="UNK"
    su=($9=="Success")?1:0
    key=$1","$2","$4","$5; ismal=((key in RTK)?1:0); kp+=ismal
    if ((u in RTU) || (substr(u,2)+0)%S==0) print $1,$2,$5,$4,at,lt,su,ismal
  }
  END { printf("red-team-labeled rows kept: %d\n", kp) > "/dev/stderr" }
' redteam.txt <(zcat auth.txt.gz) > $OUT

echo "rows: $(wc -l < $OUT)"
echo "red-team positive rows: $(awk -F, '$8==1' $OUT | wc -l)"
echo "distinct users: $(cut -d, -f2 $OUT | sort -u | wc -l)"
echo "distinct compromised users: $(awk -F, '$8==1{print $2}' $OUT | sort -u | wc -l)"
echo "auth-type dist:"; cut -d, -f5 $OUT | sort | uniq -c | sort -rn | head
echo "logon-type dist:"; cut -d, -f6 $OUT | sort | uniq -c | sort -rn | head
echo LANL_RICH_PREP_DONE
