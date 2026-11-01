# OpenViTac isolated preparation

- Independent worktree/branch: AdaViP-openvitac / benchmark/openvitac_20261101,
  based on d66c407. Existing ManiFeel/UniVTAC source and jobs stay untouched.
- Download complete: HF revision5fb2757b88826917242adefd9196424ef8cf2d98,
  6,599files /9,455,749,868bytes, all Git/LFS hashes checked; zero failures.
  Raw root: /data/wangzihao/datasets/openvitac/raw/5fb2757b.
- Converted all11tasks/1,300episodes/306,218frames. Output:
  /data/wangzihao/datasets/openvitac/converted/uvip_v1_5fb2757b.
  All5,200video headers agree with numeric episode lengths/shapes/60FPS.
  Masked201non-nominal source transitions;304,717valid next-state actions;
  282,967complete H16/history2 windows. Train/validation255,040/27,927windows
  from1,172/128disjoint, variant-stratified episodes. Converted metadata47,122,176bytes;
  immutable MP4 references avoid full image-cache expansion.
- Public simulation has no action. Derived A8 is next measured arm q7(rad)
  plus next measured LEFT finger(m), separately from measured P8's open fraction.
  This is explicitly versioned proxy supervision; controller/contact replay is
  still required on the independently deployed3090 simulator.
- New modules: six-input encoder, adapted 8D DP, independent Stage0 workspace,
  nine conditions/eight Utility dimensions; Stage0.5 paired labels, verified
  label joins, Stage1 raw future heads, four-image-HyperNet Stage2 MLP adapter.
  Frozen actual CLIP embeddings preserve all three insert_block_v2 prompts.
- Isolated venv:/public/home/wangzihao/.venvs/openvitac. Base Python3.8/
  PyTorch2.4.1 are read-only; pyarrow17 is installed only in the venv.
- New tests21passed, including bounded step1 checkpoint -> resumed step2,
  both at epoch0, and nine-condition validation. Related legacy tests59passed.
  A repeated combined regression hit the documented NFS FileLock wait;
  interrupted only that test process after53passes, then remaining26passed.
- Real-data full DP CPU smoke passed in34.66s,301,447,174parameters;
  batch2/history2/H16/A8, four full ResNet18+GN and original U-Net512/1024/2048.
  DataLoader14windows covers11tasks and12unique instructions. ActionLoss1.315205;
  V1/V2/T1/T2/P/L/denoiser gradients finite/nonzero. Stage0.5 uses only2windows
  x2draws x9conditions; Stage1 and Stage2 actual-data backward pass.
  No real optimizer updates, no full epoch and no large Utility generation.
  Evidence:/data/wangzihao/outputs/openvitac/cpu_smoke_full_20261101/report.json.
- Existing running jobs412402/412403 and queued412404/412405/412407 were not
  cancelled, reprioritized or modified. A separate bounded H200 smoke uses a
  frozen source snapshot,20min limit and nice10000; GPU success is reported
  only if its report is actually produced.

- Active bounded GPU job412580 is PENDING/AssocGrpGRES. Frozen source plus
  Diffusion Policy5ba07ac is source_20261101_v2,422hashed files; its21new tests
  passed in16.37s. Only our earlier pending smoke412577 was cancelled before
  execution to include the dependency in the final immutable snapshot.

Detailed data/action/mask/storage/remaining checks: ../openvitac_integration.md.
