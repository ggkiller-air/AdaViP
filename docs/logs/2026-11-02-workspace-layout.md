# Three-repository workspace organization

- User authorized retaining AdaViP-main, UniVTAC and OpenViTac as independent
  repositories, and committing changes directly to main without new branches.
- Preserved old workspaces and artifacts under
  `/public/home/wangzihao/project-archive/2026-11-02/`. Pre-existing dirty T2
  documentation and untracked artifacts remain intact; the archive includes
  an additional binary patch and before/after content inventories.
- AdaViP-main now has independent Git metadata. Its three existing submodule
  clones previously borrowed objects from the Stage2 worktree; materialized
  those objects, removed the dependency and passed connectivity checks.
  Repaired all archived worktree/submodule Git paths. Before code integration,
  every original tracked-file hash, HEAD and Git status matched the inventory.
- Fast-forwarded the existing local main to OpenViTac adapter commit 8684165,
  which already descends from the active single-task experiment d66c407.
  Repaired 34 active configuration/launcher files to use current repository,
  legacy label and Piper dependency paths. Model and training settings remain
  unchanged. Historical logs and archived source files retain their content.
- UniVTAC remains unchanged. Piper has no local Git repository or recorded
  GitHub association, so it is archived without creating/pushing a repository.
- Active jobs 412402/412403/412404/412405/412407 use immutable source under
  `/data/wangzihao/outputs/manifeel/`; none were moved, cancelled or restarted.
- Verification preserved 5,462 tracked-file hashes across seven repositories,
  along with their original HEADs and dirty/untracked status. Independent Git
  and submodule connectivity checks passed; all archived initialized submodules
  still resolve their original revisions.
- Focused OpenViTac and Slurm submission regressions: 30 passed in 31.36s.
  All 29 changed shell launchers pass bash syntax checks; the changed Python
  entrypoint parses successfully. All five referenced legacy label/Piper
  dependencies exist. Git whitespace checks pass.
- Publication target is the existing origin/main; no new branch or force push.
  Final remote commit verification is retained in the local archive inventory.
