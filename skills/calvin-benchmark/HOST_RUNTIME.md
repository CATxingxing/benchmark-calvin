# CALVIN host data

The Nodes bring their own Python environments (uv, locked). The host supplies only data: the
CALVIN source checkout and the evaluation dataset. The MCIL weight is downloaded by the start hook.
Validated on Ubuntu 20.04.6 / x86_64.

## CALVIN checkout (`calvin.root`)

- https://github.com/mees/calvin, commit `fa03f01f19c65920e18cf37398a9ce859274af76`
- `calvin_env` submodule `1431a46bd36bde5903fb6345e68b5ccc30def666`
- `calvin_env/tacto` submodule `dd53360d9a8c186f0d6439372ec0be0fa5e21731`

```bash
git clone --recursive https://github.com/mees/calvin && cd calvin
git checkout fa03f01f19c65920e18cf37398a9ce859274af76 && git submodule update --init --recursive
```

Nothing is installed from it; the Nodes put `calvin_env`, `calvin_models` and `tacto` on `sys.path`.

## MCIL weight (`mcil-d_d-1207`)

Official archive http://calvin.cs.uni-freiburg.de/model_weights/D_D_static_rgb_baseline.zip
(673403024 bytes, sha256 `e2e7a4d1...5736455c`). The server answers plain HTTP only, so the hook
accepts it only after the archive sha256 matches the lock. Extracted `mcil_baseline.ckpt` sha256
`cf2bc864900ce38b85daa68300addb8fdef79cc7102bd652a99f5f1924b69346`.

## Evaluation dataset (`calvin.dataset`)

One dataset serves both suites. Start from the CALVIN debug dataset
(http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip) and take from the official
task_D_D release (http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip):

1. `validation/lang_annotations/embeddings.npy` and `auto_lang_ann.npy`: the L6 MiniLM
   (`paraphrase-MiniLM-L6-v2`) files the MCIL checkpoint was trained with. The debug archive ships
   L3 files, which silently collapse the score.
2. `validation/ep_start_end_ids.npy`: the official episode bounds.
3. The 334 validation reset frames (`episode_*.npz`) referenced by the MTLC task map of the
   checkpoint. The debug archive does not contain all of them.

Range requests into task_D_D.zip fetch only these members; the full archive is not needed. Keep the
debug `.hydra/merged_config.yaml` and `statistics.yaml`; they are identical to task_D_D. The hook
checks all five files above against `locks/calvin_dataset.lock.json`.

The dataset uses the official sequences and reset states. A run is an official score only when it
covers the whole suite (`is_full_suite: true` in the result).
