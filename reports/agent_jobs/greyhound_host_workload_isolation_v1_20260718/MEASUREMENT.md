# Controlled I/O measurement

result: WORKING

## Workload

One exact report root was searched for a deliberately absent fixed string:

```text
root=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/reports/agent_jobs
operation=rg over *.md and *.json, generated reports explicitly included
workers=1 timeout=60s rbps=1MiB/s riops=16
result=exit 1 (expected no match), approximately 35s, container removed
```

No unbounded comparison was run because recreating the incident was outside the
safety boundary.

## Before, during, and after

At 17:13:30 AEST, before the transient child, global I/O PSI was
`some avg10=2.32` and `full avg10=2.07`. The device had read sector counter
`23856243067`. This broad before-to-after interval also contained unrelated
host and live-collection activity and is not attributed to the wrapper.

The cleanest same-process measurement interval was 17:13:49.234 to
17:14:15.710 AEST (26.476 seconds):

| Counter | Start | End | Delta |
|---|---:|---:|---:|
| `/proc/<pid>/io` `rchar` | 84,067 | 395,862 | 311,795 B |
| `/proc/<pid>/io` `read_bytes` | 1,720,320 | 3,039,232 | 1,318,912 B |
| `nvme0n1` sectors read | 23,856,653,731 | 23,856,656,283 | 2,552 sectors / 1,306,624 B |
| `nvme0n1` sectors written | 189,740,920 | 189,741,752 | 832 sectors / 425,984 B |
| I/O PSI `some total` | 14,621,864,486 | 14,632,849,370 | +10,984,884 us |
| I/O PSI `full total` | 13,696,705,134 | 13,705,083,607 | +8,378,473 us |

The process-attributed physical-read delta and whole-device read delta differed
by only 12,288 bytes in that interval. Docker reported 3.1 MB read and one PID
near the end of the run.

`rchar` is only bytes returned by read-like syscalls; it is not total file
content traversed when `rg` uses memory mapping. `read_bytes` is storage I/O
attributed to the process, while the disk sector delta is physical activity for
the entire device and can include other processes. The close agreement above is
useful attribution evidence, but it does not make the counters interchangeable.

Global PSI rose during intentional throttling (`some avg10` 14.35 to 39.89;
`full avg10` 13.86 to 32.47) because the single bounded task spent time waiting
for I/O. This is not reported as zero pressure. Concurrent inspection commands
still completed in 0.3 to 1.3 seconds, the child stayed at one PID, and natural
odds cycles succeeded. At 17:14:46, after child cleanup, PSI avg10 had receded to
`some=17.91`, `full=16.35`; the transient container was absent.

## Conclusion

The intended hard ceiling was applied and verified before the search, device
activity remained attributable and bounded rather than a multi-thread surge,
the host remained interactively responsive, cleanup completed, and live odds
collection was not disrupted. PSI is retained as a cautionary signal: even a
properly throttled reader can appear as pressure while it waits, so future
checks must pair PSI with per-process I/O, cgroup limits, and diskstats.
