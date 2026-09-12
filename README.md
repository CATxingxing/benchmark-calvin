# Forge benchmark: CALVIN

Self-contained Forge benchmark node for CALVIN MTLC and LH-MTLC. The generic `_core/`, `bench_tool/`, and `result.py` are copied unchanged from the `benchmark-libero` reference implementation. CALVIN-specific evaluation semantics live under `adapters/` and in `config.py`.

## Architecture

Exactly two Dora nodes are used: this benchmark node owns the CALVIN environment in-process and communicates only through Forge messages with an unmodified Policy node. No Simulator node is introduced.

## Required runtime inputs

Set `CALVIN_ROOT` to the pinned CALVIN checkout, point `dataset_path` at the CALVIN dataset root, and explicitly configure `suite`, `protocol`, `action_mode`, and `policy_id`. MTLC additionally requires `checkpoint_path` so `task_to_id_dict` comes from the official checkpoint. Unknown Policy I/O semantics are not guessed.

## Official semantics preserved

- LH-MTLC uses the official `get_sequences(1000)` set, one environment reset per 5-subtask sequence, official validation language, fresh TaskOracle start-info per subtask, and a 360-action budget per subtask.
- MTLC uses checkpoint `task_to_id_dict`, official validation dataset indexing/reset state, official language, TaskOracle, and a 240-action budget.
- Environment reward/done are not substituted for CALVIN TaskOracle success.
- The benchmark layer does not resize, flip, normalize, or otherwise perform Policy preprocessing.

`config.example.yaml` selects one LH-MTLC sequence for a smoke run while still loading/validating the official 1000-sequence set; remove `task_ids` only when intentionally running the full benchmark.

## Reference-core boundary

`_core/`, `bench_tool/`, and `result.py` are intended to remain byte-identical to the corresponding files in the reference `benchmark-libero` revision used for this integration. Benchmark-specific changes should stay in `adapters/`, `config.py`, examples, and minimal assembly code.

## Observation contract and startup self-check

The node renders CALVIN's native resolution and publishes it unchanged (scaling is the
policy's job, manual §3.2):

| dora channel | CALVIN camera | native resolution |
| :--- | :--- | :--- |
| `image/main` | `rgb_static` | 200 x 200 x 3 |
| `image/second` | `rgb_gripper` | 84 x 84 x 3 |
| `proprio_state` | `robot_obs` (native) | 15 floats (euler_obs=true) |

At construction the adapter logs a one-time startup self-check
(`[calvin_adapter] observation contract: ...`, manual §3.2 item 5 / §8) and repeats the
same information under `reproducibility` in the result JSON. No flip/resize/normalize is
applied at the benchmark layer.

## Language embeddings (required for a meaningful score)

CALVIN decides success with the environment TaskOracle; the policy conditions on
precomputed MiniLM sentence embeddings stored in the dataset (`lang_annotations/embeddings.npy`).
The released baselines use **`paraphrase-MiniLM-L6-v2`** (CALVIN's `conf/model/sbert.yaml`;
before 2023 the config value `mini` resolved to the same model). Some dataset packages ship
the **L3** variant in `lang_annotations`, which does not match the checkpoint and silently
collapses the score. When preparing an evaluation dataset, ensure
`lang_annotations/embeddings.npy` is the L6 embedding for the checkpoint; the acceptance
harness used a repo-external copy with L6 embeddings.

## Known deviations from benchmark-libero

- `main.py` differs by a small amount (LIBERO-only `sys.path` handling removed, log tag and
  `sys.exit` style); it is not modified further by this node.
- Sub-task budget exhaustion is reported as `termination="timed_out"` (manual §3.4 rule 3 / §8).
  CALVIN has no `failed` signal, so the only non-success terminal state is a timeout. The
  adapter publishes the current sub-task deadline through the shared runner's `cfg.max_steps`
  (cumulative actions + remaining sub-task budget) on every step, so the frozen `_core`
  runner's `timed_out` branch fires on the exact official 360/240-action boundary. This is an
  explicit adapter→core coupling: it relies on `_core/runner.py` re-reading `cfg.max_steps`
  each step and on `_core/session.py` sharing the live cfg object with the adapter. It is
  locked by `tests/test_termination_contract.py`; the graded metrics (`avg_seq_len`,
  `SR1..SR5`, `num_steps`, `success`) are unchanged.
- CALVIN joint names / state semantics are derived in the adapter from the dataset
  `merged_config.yaml` (`_resolve_state_names`) rather than declared as `config.py` constants.

## Preflight probes (read-only)

`preflight/probe_env.py` (reset determinism, frame dump), `preflight/probe_official_trace.py`
(official env vs adapter fixed-action trace), `preflight/probe_oracle.py` (TaskOracle call
path). They observe only and never edit CALVIN, the shared core or the adapter.
