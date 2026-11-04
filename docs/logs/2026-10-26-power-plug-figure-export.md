# Power Plug paper figure export (2026-10-26)

- User selected Power Plug and requested five same-frame PNGs plus a task
  prompt under the project's `figures/` directory. Read source Zarr
  `/data/wangzihao/datasets/manifeel/plug_quan_Aug02` with `mode='r'`;
  no training, simulation, GPU use, or source dataset writes.
- Exported zero-based episode 0, local/global frame 79; episode bounds
  `[0, 124)`. Fields: `front`, `side`, `wrist`,
  `left_tactile_camera_taxim`, `right_tactile_camera_taxim`. Original HWC
  resolution is 256x256x3 for cameras and 320x240x3 for Taxim.
- Conversion: `np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)`.
  Preserved source channel order and resolution without crop or enhancement.
  All five RGB PNGs passed exact decoded-pixel equality against the converted
  source frame using the existing ManiFeel Python environment and Pillow.
- `figures/power_plug_prompt.txt` contains the authored instruction
  `Insert the power plug into the socket.` The configured task text is
  `power plug insertion`; the full instruction is not claimed as source-Zarr
  language annotation. `figures/power_plug_source.json` records provenance,
  conversion, shapes, frame alignment limits, and PNG SHA-256 hashes.
  Generated artifacts are ignored through `figures/.gitignore`.
