# Training progress at23:26CST on2026-10-31

- Read-only scheduler/metrics/checkpoint and independent GPU checks confirm five active training jobs. Four Future stderr files and the current dropout stderr are empty; logged Future losses/gradient norms are finite. All five GPUs are NVIDIA H200143771MiB.

| Run | Current/completed epochs | Step | Last epoch Action | Last epoch Future | Recent samples/s | Rough remaining hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | 79/78 | 175200 | 0.01414404 | 0.0001541673 | 183.7 | 2.4 |
| C | 40/39 | 89060 | 0.02146322 | 0.0188683 | 177.0 | 7.1 |
| D | 36/35 | 79960 | 0.02246485 | 0.001086903 | 161.2 | 8.4 |
| E | 25/24 | 53620 | 0.02469738 | 0.01608422 | 159.4 | 9.9 |
| L1 dropout10 | 6/5 | 11220 | 0.03550022 | none | 193.1 | 10.4 |

- B407886 has fixed E20/40/60; C407887,D407888,E407889 each have E20. Every run has nonempty latest.ckpt; dropout409885 has not reached E20. Losses are complete-epoch means; current epochs are still in progress. ETA extrapolates recent epoch wall time and is not guaranteed.
- Direct GPU reads B/C/D/E:40977/41053/40989/41307MiB, approximately28.5--28.7%. Dropout initially17767MiB (12.36%); added its output/job409885 to existing407906 supervisor manifest under the prior >=15% instruction. Verified holder1269069,24017601822bytes; dropout now41289/143771MiB (28.72%). No model/optimizer/checkpoint or training hyperparameter changed.
- Dropout device-fix preflight actually passed on CUDA, normalizer and normalized inputs bothcuda:0. Formal updates and checkpoint now exist, replacing the prior pending/unverified status.
- Recent weighted Future/action fusion-gradient ratios B/C/D/E approximately0.00367/0.0600/0.0312/0.0489. Lambda remains fixed as requested; the initial10% calibration does not enforce a constant ratio throughout training.
- Runtime benchmark407932 completed36cases with0errors; no benchmark profile was applied to production. No rollout result is inferred from training loss.
- Evidence: /data/wangzihao/outputs/manifeel/stage1_future_k4_batch32_ep100_seed42_20261031_133616/analysis/training_status_20261031_2325/status.json
