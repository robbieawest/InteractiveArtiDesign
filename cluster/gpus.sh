#!/bin/bash
# Free GPUs by type on a partition — the view Slurm does not give you.
#
#   cluster/gpus.sh [partition]        default: Teaching
#
# `sinfo %G` reports what is *installed*; GresUsed reports what is *allocated*.
# Neither answers "how many can I have right now", so this subtracts them per
# type and aggregates across nodes.
#
# Nodes in a state that cannot accept work (down, drain, maint, fail) are
# counted separately rather than as free: their GPUs are configured, and
# entirely unobtainable, which is exactly the trap that makes a job sit in PD
# against a partition that looks half empty.

set -uo pipefail
PART="${1:-Teaching}"

# SINFO_FIXTURE lets the parser be exercised against captured output — useful
# for testing, and for asking someone else to look at a confusing partition:
#   sinfo -h -N -p Teaching -O "NodeList:40,StateLong:24,Gres:100,GresUsed:100" > out.txt
#   SINFO_FIXTURE=out.txt cluster/gpus.sh
if [[ -n "${SINFO_FIXTURE:-}" ]]; then
    cat "$SINFO_FIXTURE"
else
    sinfo -h -N -p "$PART" -O "NodeList:40,StateLong:24,Gres:100,GresUsed:100,Reason:60"
fi \
| python3 -c '
import re, sys
from collections import defaultdict

# Substrings of StateLong that mean a new job cannot land here. "reserv"
# rather than "resv": Slurm prints RESERVED, and a node inside a reservation
# is configured, idle, and completely unobtainable without --reservation —
# which is exactly how a partition looks free while every request comes back
# ReqNodeNotAvail.
UNUSABLE = ("down", "drain", "drng", "fail", "maint", "unk",
            "reserv", "futur", "boot", "pow")

total, used, stuck = defaultdict(int), defaultdict(int), defaultdict(int)
nodes_by_type = defaultdict(set)
blocked = []  # (node, state, reason) for anything a job cannot use

def parse(field):
    """gpu:h200_1g.18gb:35(S:0-1),gpu:h200:1(S:0) -> {type: count}"""
    out = defaultdict(int)
    if not field or field in ("(null)", "N/A"):
        return out
    for token in field.split(","):
        token = re.sub(r"\(.*?\)", "", token).strip()   # drop (S:0-1) / (IDX:...)
        if not token.startswith("gpu:"):
            continue                                    # mps, nic, etc
        bits = token.split(":")
        if len(bits) == 3:                              # gpu:<type>:<n>
            kind, count = bits[1], bits[2]
        elif len(bits) == 2:                            # gpu:<n>, untyped
            kind, count = "(untyped)", bits[1]
        else:
            continue
        try:
            out[kind] += int(count)
        except ValueError:
            pass
    return out

rows = 0
for line in sys.stdin:
    parts = line.split(None, 3)
    if len(parts) < 3:
        continue
    node, state, rest = parts[0], parts[1].lower(), " ".join(parts[2:])
    # Gres then GresUsed then Reason; Reason may contain spaces, so it is
    # whatever is left after the first two fields
    fields = rest.split(None, 2)
    cfg = fields[0] if len(fields) > 0 else ""
    usd = fields[1] if len(fields) > 1 else ""
    reason = fields[2].strip() if len(fields) > 2 else ""
    rows += 1
    bad = any(s in state for s in UNUSABLE)
    if bad:
        blocked.append((node, parts[1], reason))
    for kind, n in parse(cfg).items():
        total[kind] += n
        nodes_by_type[kind].add(node)
        if bad:
            stuck[kind] += n
    if not bad:
        for kind, n in parse(usd).items():
            used[kind] += n

if not rows:
    sys.exit("no nodes reported — check the partition name")

print("%-30s %6s %6s %6s %9s  %s" % ("gpu type", "free", "used", "total", "unusable", "nodes"))
print("-" * 78)
for kind in sorted(total, key=lambda k: -(total[k] - used[k] - stuck[k])):
    free = total[kind] - used[kind] - stuck[kind]
    print(f"{kind:<30} {free:>6} {used[kind]:>6} {total[kind]:>6} "
          f"{stuck[kind]:>9}  {len(nodes_by_type[kind])}")
print()
print("free = total - used - unusable (down/drain/maint/reserved).")
print("Request one with:  --gres=gpu:<gpu type>:1")

if blocked:
    print()
    print("nodes a job cannot land on:")
    for node, state, reason in blocked:
        print("  %-20s %-16s %s" % (node, state, reason or "-"))

print()
print("If a type shows free but every request returns ReqNodeNotAvail, the")
print("node is probably inside a reservation this job is not entitled to:")
print("  scontrol show reservation")
print("  squeue -j <jobid> -o \"%.10i %.8T %.80R\"   # names UnavailableNodes")
print("  scontrol show node <node> | grep -E \"State|Reason|Partitions\"")
'
