# ManiFeel Environment Deployment Progress

Updated: 2026-08-07 04:20 CST
Workspace: `/public/home/wangzihao/projects/AdaViP`

## Current Status

- The ManiFeel Python environment, official USB data, headless simulator smoke,
  minimal DP training, and standalone `eval.py` smoke are complete.
- All nine official datasets are downloaded, ZIP-tested, and extracted. The
  main route was `hf-mirror.com`; after repeated SSL EOFs on `pih`, the job
  switched once to the same mirror with local proxy variables unset and then
  resumed from completed chunks.
- Persistent artifacts are under `/public/home/wangzihao/` or
  `/data/wangzihao/`, not `/tmp/`.

## Verified Environment

- Conda environment: `/public/home/wangzihao/.local/miniforge3/envs/manifeel`
- Python: 3.8.20
- PyTorch: `2.4.1+cu121`; CUDA runtime: 12.1
- GPU verification: `torch.cuda.is_available() == True` on an NVIDIA H200,
  compute capability 9.0.
- Isaac Gym/TacSL, `gymtorch`, `isaacgymenvs`, Diffusion Policy, and ManiFeel
  are installed. The headless smoke reset the USB environment and completed
  five GPU simulation steps successfully. Evidence:
  `/data/wangzihao/outputs/manifeel/logs/smoke_env_20260807.log`.
- The missing Autodesk FBX module is a non-blocking warning for this URDF-based
  USB path; it did not prevent environment, training, or evaluation smokes.
- `pip check` still reports two upstream metadata conflicts: `pymanopt 2.2.1`
  excludes SciPy 1.10, and `scanpy 1.9.8` requires `networkx>=2.3` while
  `urdfpy 0.0.22` installs NetworkX 2.2. The verified vision-DP path does not
  exercise those packages; specialized FMDP/representation paths need a
  separate dependency test before use.

## USB Dataset

- Archive: `/data/wangzihao/datasets/manifeel/archives/usb_quan_Aug05.zip`
- Size: 3,991,771,935 bytes
- SHA-256: `25e7912dec28282a2294adc34f819be59a01cebc73ec21501d938dc78f64cb00`
- `unzip -tq` passed. The extracted Zarr dataset is at
  `/data/wangzihao/datasets/manifeel/usb_quan_Aug05` and contains 12,204 files.

## All Official Datasets

Every archive below matched the expected byte size, passed `unzip -tq`, and
has a root `.zgroup` in the corresponding extracted directory:

| Dataset | Archive bytes | Extracted files |
| --- | ---: | ---: |
| `usb_quan_Aug05` | 3,991,771,935 | 12,204 |
| `sorting_quan_Aug8` | 3,657,177,415 | 10,765 |
| `gear_quan_Sep15` | 3,845,302,652 | 11,289 |
| `pih_quan_June06` | 4,704,135,973 | 14,926 |
| `plug_quan_Aug02` | 3,794,136,821 | 11,050 |
| `nutbolt_quan_July1` | 8,774,121,109 | 28,168 |
| `bulb_quan_Sep19` | 8,835,001,402 | 25,233 |
| `blindinsert_quan_Aug15` | 5,787,270,412 | 14,469 |
| `explore_quan_June17` | 5,640,103,031 | 16,412 |

Archives and extracted data are under `/data/wangzihao/datasets/manifeel/`.
Completed ranged-download part caches remain under `archives/*.zip.parts`
(about 46 GB) for resumability; they are not needed for evaluation once the
verified archives are retained.

## Training and Evaluation Smoke

- A one-epoch, one-train-step vision-wrist DP smoke completed on the USB data.
- Checkpoint:
  `/data/wangzihao/outputs/manifeel/dp_usb_vision_wrist_smoke/checkpoints/latest_epoch0.ckpt`
- Standalone `eval.py` then loaded the checkpoint and its saved Hydra config,
  created the GPU simulator, ran five evaluation steps, and wrote:
  `/data/wangzihao/outputs/manifeel/dp_usb_vision_wrist_smoke/eval/eval_log.json`
- Result: `test/mean_score = 0.0`; rollout video:
  `/data/wangzihao/outputs/manifeel/dp_usb_vision_wrist_smoke/eval/media/2eixoo2v.mp4`
- Logs:
  `/data/wangzihao/outputs/manifeel/logs/smoke_train_dp_20260807.log` and
  `/data/wangzihao/outputs/manifeel/logs/smoke_eval_dp_20260807.log`.

This smoke checkpoint proves that data loading, model construction,
checkpoint serialization, Isaac Gym rollout, and `eval.py` work end to end. It
is not a trained benchmark policy, and its score is not evidence of task
performance.

The batch script is resumable and accepts an optional list of archive names;
the completed run is recorded in
`/data/wangzihao/outputs/manifeel/logs/download_all_datasets_20260807.log`.

## Policies and Public Resources

- The master branch uses the upstream image-conditioned
  `DiffusionUnetImagePolicy` as its standard baseline and includes
  `DiffusionUnitPolicy`, `DiffusionT3Policy`, `DiffusionAnyTouchPolicy`,
  `DiffusionTacffPolicy`, `DiffusionEquiUNetCNNEncRelPolicy`, `FMDP`, and
  `FMDP3`. Task configs cover vision, TacRGB, and tactile force-field inputs.
- RDP is present only on `origin/rdp`, not the checked-out master branch.
- Official task datasets and UniT/T3/AnyTouch representation checkpoints are
  public, but no trained ManiFeel task/action-policy checkpoint was found in
  the official README, GitHub master tree, or Hugging Face space.
- A DP that changes observation modalities normally needs a matching
  `shape_meta`, dataset keys, simulator observations, policy architecture, and
  newly trained or explicitly fine-tuned checkpoint. An unrelated standard DP
  checkpoint generally cannot be substituted.

## Next Actions

1. Define the modified-modality DP contract and train a real checkpoint on the
   relevant ManiFeel demonstrations before comparing evaluation performance.
2. Remove the redundant ranged-download caches only after confirming the
   verified archives are backed up as needed.
