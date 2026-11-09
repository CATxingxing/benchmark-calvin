---
name: calvin-benchmark
description: Run CALVIN evaluation (LH-MTLC and MTLC). Profile vla-mcil-1207 runs the official MCIL baseline (weight mcil-d_d-1207) inside the Skill.
---

# CALVIN evaluation

Profile `vla-mcil-1207` starts everything it needs: the Gateway, the CALVIN benchmark Node and the
MCIL policy Node. MCIL loads once at Skill startup and stays resident; stopping a benchmark
Session returns it to idle and later batches reuse the process. There is no external model server.

1. Start the profile and require runtime status `ready`. The first start builds each Node's
   environment and verifies or downloads the weights (several minutes); later starts are fast.
2. Query `calvin.policy.describe`; require `ready=true` and action dimension 7.
3. Query `calvin.benchmark.describe`. It lists both suites (`lh_mtlc`, `mtlc`) with their task
   tables and default `max_steps`. Use them; do not infer task IDs from names or invent coverage.
4. Start `calvin.benchmark.run` as a Session with `suite` set (`lh_mtlc` is the default). Default
   smoke: task_ids=[0], init_state_ids=[0], num_runs=1, seed=0. Ask for the evaluation scope when
   the user did not give one.
5. Poll the run Session, consume terminal events, then fetch its result. Report completed_episodes,
   total_episodes, success_rate, failure_summary, result_file and `primary_metric` (LH-MTLC:
   avg_seq_len with SR1-SR5 in `metrics`; MTLC: success_rate). `is_full_suite` is true only for a
   complete suite.
6. To stop a batch, stop the run Session and wait for its terminal result; the current episode
   finishes first.
7. Another batch, also of the other suite, may start after the previous one is terminal. When
   finished, stop the Skill and check that the Gateway and Node processes are gone.

If start fails in the start hook, report its message: it names the missing setting in
`~/.PhyAgentOS/benchmarks.yaml` (CALVIN root or dataset), the dataset file or the weight file that
failed verification. Treat missing weights, assets, runtime errors and partial/stopped batches as
such. Unit tests or a stub policy do not establish model accuracy. A subset score
(`is_full_suite=false`) must never be described as a full official benchmark score.
