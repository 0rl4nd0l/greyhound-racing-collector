# Deployment Manifest

result: WORKING

- Runtime worktree:
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-p0-20260710`
- Runtime state: clean detached worktree at proof head
  `7b59f4d34e027f409166447f7fded2def5f7b447`.
- Full service generated/installed SHA-256:
  `8067d1a0dfbd94d76ee51821a0dfc73b9b9a6a32056f0e066c343c565239e6ce`.
- Odds-only service generated/installed SHA-256:
  `a4fb752b4d04b3107bc93dfb4b7e86e004e22993c92dcabd2e3c805ba2ff5b1e`.
- Generated and installed unit files are byte-identical.
- `shadow-autopilot.timer`: disabled/inactive.
- `shadow-autopilot-odds-capture.timer`: disabled/inactive.
- `shadow-autopilot.service`: inactive/dead, previous result success.
- `shadow-autopilot-odds-capture.service`: inactive/failed with retained earlier
  exit status `2`; it was not started during this proof.
- Production main DB SHA-256:
  `470e97b83b02bc8070277945c062052572ce209a58d1d5bacb0f24076cedd61b`.
- Production writable DB SHA-256:
  `61b9ee76a52068435ef3c96528bbdbd9d4498180f6b055ab0e828a7f3559436e`.
- Production stage DB SHA-256:
  `7af475c57e63f2ad69cac2c2281c8a59d06bc073e1ef5e722729dc9f1cfbe6f1`.
- Rollback: keep both timers disabled; the production DBs and installed units
  already match their pre-run state, so no DB restore or unit rewrite is
  required.
