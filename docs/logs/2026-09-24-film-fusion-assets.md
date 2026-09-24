# 2026-09-24 FiLM Fusion Assets

- Uploaded the frozen backbone checkpoints to ModelScope `ggkiller/multi-fm`
  over a direct connection:
  `pretrained/film_fusion/clip/RN50.pt` (SHA256
  `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`) and
  `pretrained/film_fusion/sparsh/dino_vitbase.ckpt` (SHA256
  `f1f97da5c26bf2b1ba1f31ec9f5d02ee1b4ef649c9f3df119822a06d264c4115`).
- FiLM Fusion eval requires both checkpoints and source trees: CLIP's
  `clip/model.py` and Sparsh's `tactile_ssl` package are loaded at runtime.
  Added environment-variable overrides for all four paths and a direct
  ModelScope downloader for the checkpoints.
- Focused FiLM Fusion tests passed (`8 passed`); full repository tests should
  be rerun before committing follow-up changes.
