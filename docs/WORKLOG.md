# AdaViP Work Log

This is an append-only record of repository changes, investigations, and
verification results. Durable decisions belong in `docs/PROJECT_CONTEXT.md`.

## 2026-08-03

- Reviewed `docs/AdaViP_ICRA2027.pdf` and `docs/pipeline.png`.
- Summarized the proposed adaptive perception architecture, real-robot tasks,
  comparison methods, ablations, and unresolved implementation details.
- Confirmed that the workspace initially contained only documentation and did
  not contain a valid initialized Git repository.
- Verified outbound GitHub and GitHub API connectivity after Full Access was
  enabled for the execution environment.
- Located the official repositories for Diffusion Policy, Reactive Diffusion
  Policy, and Policy Consensus.
- Cloned the three baseline repositories into `third_party/` and pinned their
  current HEAD commits in `docs/PROJECT_CONTEXT.md`.
- Ran `git fsck --no-dangling` on each clone and verified clean working trees.
- Inspected Reactive Diffusion Policy and confirmed that it contains DP policy,
  workspace, configuration, dataset, and training-script implementations.
- Determined from arXiv v2 that ReTac-ACT code had not yet been released.
- Removed ReTac-ACT from the experiment scope. Table I now contains VT-DP,
  RDP, Policy Consensus, VT-DP + AdaViP, and RDP + AdaViP.
- Removed the no-longer-needed `third_party/act` clone by moving it to the
  system trash. It remains recoverable until the trash is emptied.
- Audited Policy Consensus documentation and confirmed its RGB + Tactile
  compositional-policy configuration and real-world tactile path.
- Decided to keep the three complementary visuotactile baseline families in
  the primary table and defer pi0/openpi-style VLA comparisons to a separate
  future extension.
- Set the first validation order: LeRobot record/replay for hardware
  connectivity, RGB + proprioception Diffusion Policy smoke test, then VT-DP,
  RDP, Policy Consensus, and finally AdaViP. GelSight-dependent runs wait for
  the sensor to arrive.
- Added this work log and `docs/PROJECT_CONTEXT.md`.

No Python dependencies, datasets, model weights, or checkpoints have been
installed or downloaded yet.

## 2026-08-04

- Downloaded one `aloha_multiview` HDF5 episode outside the repository for
  schema inspection and future RGB + proprioception smoke tests.
- Kept dataset format decisions open until the collection pipeline and tactile
  synchronization contract are concrete.
- Established the repository boundary: data contracts remain separate from
  format-specific readers, synchronization, conversion, and sampling.
- Added the data-format-independent AdaViP policy core under `adavip/`.
- Implemented task-conditioned HyperNet parameter generation, low-rank
  modality and fusion encoder residuals, shared decoders, cross-modal
  attention, frozen base-encoder injection, and an unchanged backbone-policy
  interface.
- Added shape, conditioning, and encoder-freezing unit tests. Static Python
  compilation passed; runtime tests remain pending because PyTorch and pytest
  are not installed in the current environment.
- Verified that all three pinned repositories under `third_party/` remain
  unmodified.
