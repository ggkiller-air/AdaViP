# Code publication and S2/S3/S4 checkpoint release

- User requested all current code on GitHub for pulling on the evaluation
  server. Current branch descends from origin/main; publish by normal
  fast-forward, without changing running trainers or third_party revisions.
- Includes runtime/IPC/cache improvements, explicit LC precision profiles,
  isolated runtime/memory benchmarks, combined-partition submission helper,
  documentation and regression tests. Existing checkpoint loader supports
  the S2/S3/S4 encoder strategies.
- Validation: changed Python sources parse; changed shell/Slurm scripts pass
  bash -n; git diff --check passes. Full suite: 270 passed, 1 skipped,
  5 existing Transformer warnings in195.53s. Command:
  `TMPDIR=/public/home/wangzihao/.cache/lc/i/push_tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python -m pytest -q --capture=sys --basetemp=/dev/shm/adavip_push_pytest_sys_20261030`.
  Log: `/public/home/wangzihao/.cache/lc/i/push_tests/pytest_sys.log`.
  Initial default-capture invocation failed before tests due to NAS temporary
  file behavior; the explicit capture/memory-rendezvous invocation passes.
- User separately authorized direct HTTPS ModelScope upload with all proxy
  variables unset and no Git transport. Twelve fixed snapshots (S2/S3/S4,
  epochs20/50/100/200), 63979228872bytes, are hard-linked under
  `/data/wangzihao/outputs/manifeel/modelscope_uploads/stage2_s234_20261030/files/`.
  Local SHA256 manifest and upload log are in its parent directory. Remote
  prefix: `ggkiller/Manifeel-ckpts1018/stage2_encoder_ablation_t2_batch32_seed42_20261028/`.
  Upload is in progress at this entry; completion requires all twelve remote
  byte sizes and SHA256 values to match the local manifest.
