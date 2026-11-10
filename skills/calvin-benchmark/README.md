# calvin-benchmark 0.2.1 candidate

Not accepted for publication until the real-model and PAOS lifecycle tests pass.

Gateway: http://127.0.0.1:19131. Tools: `calvin.benchmark.describe`, `calvin.benchmark.run`,
`calvin.policy.describe`.

| Profile | Paradigm | Policy | Weight | Suites | Nodes |
| --- | --- | --- | --- | --- | --- |
| `vla-mcil-1207` | VLA | MCIL baseline (official CALVIN) | `mcil-d_d-1207` | `lh_mtlc`, `mtlc` | gateway 1.1.0, calvin_benchmark 1.1.1, calvin_policy 1.1.0 |

Profile names are `<paradigm>-<policy>-<weight date MMDD>`. `weights.yaml` lists each profile's
weight; `locks/` pins the exact weight files and the dataset files that decide the score (size and
sha256). 0.1.0 had one profile per suite (`mcil_lh`, `mcil_mtlc`); in 0.2.0 one runtime serves both
and `calvin.benchmark.run` takes `suite`.

## User configuration

Everything goes in one file, `~/.PhyAgentOS/benchmarks.yaml`. No environment variables; select
the GPU with `CUDA_VISIBLE_DEVICES` (default 0, EGL rendering uses the same index).

```yaml
calvin:
  root: /path/to/calvin                      # required once: mees/calvin checkout with submodules
  dataset: /path/to/calvin-mtlc-evaluation   # required once: evaluation dataset, see HOST_RUNTIME.md
  # weights_dir: /big/disk/weights           # optional, default ~/.PhyAgentOS/weights/calvin-benchmark
  # weights:                                 # optional: reuse a local copy instead of downloading
  #   mcil-d_d-1207: /models/mcil/D_D_static_rgb_baseline
```

Then `paos skill start calvin-benchmark --profile vla-mcil-1207`.

The start hook (`start.sh` -> `scripts/prepare.py`) links the root and the dataset to
`~/.PhyAgentOS/data/calvin` and `~/.PhyAgentOS/data/calvin-dataset` and checks the dataset's
language embeddings, annotations, episode bounds and configs against `locks/calvin_dataset.lock.json`.
It verifies the MCIL weight against its lock; when files are missing it downloads the official
archive (0.67 GB, resumable), checks the archive sha256 and extracts only the locked files. It then
runs `calvin_benchmark --prepare` and `calvin_policy --prepare`: each Node builds its own uv
environment (CUDA torch) from its lock. PAOS starts the `startup_timeout_s` (900 s) clock only after
the hook, so a cold download on a clean machine does not count against it. Later starts reuse
`~/.PhyAgentOS/cache/nodes/<node>/<lock digest>` and skip it.

See HOST_RUNTIME.md for the CALVIN sources and how to build the evaluation dataset.

0.2.1: calvin_benchmark 1.1.1 checks every policy action against the environment action space (7D, or 8D for absolute_quaternion) before `env.step`. A mismatch fails the whole run with `action_dim_mismatch` instead of scoring each episode as 0. `describe` now reports `action_dim`.
