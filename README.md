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
