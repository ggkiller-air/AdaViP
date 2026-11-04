# K64 label window count audit (2026-10-28)

- Audited the existing immutable archive at
  `/public/home/wangzihao/projects/AdaViP/outputs/stage1_utility_labels_ep500_k64`.
  Its manifest covers all nine tasks and all 400 demonstrations: 71,265 real
  timestep anchors, with no held-out episodes in the label source.
- Frozen label protocol is observation history 2, action horizon 16, executed
  actions 8. At episode-relative anchor `t`, observations use `[t-1,t]` and
  the action target uses `[t-1,t+14]`. For episode length `L`, complete anchors
  are `1 <= t <= L-15`, implemented by
  `complete_window_index()` as `range(1, length - 14)`.
  This excludes the first anchor and last 14 anchors of each episode.
  Every episode has at least 57 frames, so the exact total is
  `71,265 - 400 * (1 + 14) = 65,265` (91.58% retained).
- Rebuilt the index from the actual manifest and compared every structured row
  with the saved `window_index.npy`; exact match. Independently checked every
  episode's ordered anchor range and presence. No additional missing windows.
  Recomputed sizes/SHA256 for all seven published final files against
  `COMPLETE.json`; all passed. The utility, variance and weighted-loss arrays
  have 65,265 finite rows. All 255 shard records form contiguous coverage of
  `[0,65265)`; raw shard payloads were not rehashed during this count audit.

| Task | Episodes | Raw anchors | K64 windows | Excluded |
| --- | ---: | ---: | ---: | ---: |
| peg_insertion | 50 | 7,371 | 6,621 | 750 |
| usb_insertion | 50 | 5,976 | 5,226 | 750 |
| power_plug_insertion | 50 | 5,425 | 4,675 | 750 |
| gear_assembly | 50 | 5,516 | 4,766 | 750 |
| nut_bolt_assembly | 30 | 14,021 | 13,571 | 450 |
| bulb_installation | 20 | 12,553 | 12,253 | 300 |
| peg_reorientation | 50 | 7,065 | 6,315 | 750 |
| object_search | 50 | 8,112 | 7,362 | 750 |
| ball_sorting | 50 | 5,226 | 4,476 | 750 |
| Total | 400 | 71,265 | 65,265 | 6,000 |

- The base full-sensor dataset can expose all 71,265 anchors using clipped
  boundary padding and validity/loss masks. K64 labels intentionally require
  complete observations and action targets, and Stage2 joins only these
  labeled rows. K64 specifies draws per window, not selection of windows.
  No code, training jobs, labels or data were changed by this audit.
