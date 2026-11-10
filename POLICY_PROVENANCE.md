# MCIL evaluation policy source

The original public benchmark revision `f322c9301f3cec290d35f65520c1cb16fcc68641`
did not include the two policy modules shipped in `calvin_policy` 1.0.0
(archive SHA-256 `1153ede383679ae0a26af24050e84c8281634022784b8eb8340181849565cc70`).
That binary explicitly described itself as validation-only. It is retained as
an audit baseline, not relabelled as a new release.

The maintained `mcil_runtime.py` reconstructs its loading/preprocessing/inference
path, cross-checked against official CALVIN `fa03f01f19c65920e18cf37398a9ce859274af76`
(MIT). It loads the official checkpoint, official dataset transforms, official
language embeddings, and calls `model.step` without additional action chunking.

Intentional lifecycle changes in `real_policy.py`:

- `stop` clears runtime state and returns to idle instead of terminating the
  process, permitting repeated benchmark Sessions in one Skill runtime.
- Actions must contain seven finite values before being sent to the simulator.
- The required reset barrier starts closed. Observations arriving before the
  first instruction/reset cannot trigger inference.
- Runtime dependency paths must exist and use the same Python minor version.

The profile is for MCIL benchmark evaluation. Unit tests, real Ubuntu 20.04 GPU
evaluation and PAOS lifecycle tests must pass before any new artifact is published.
The prior student's numerical acceptance results are not new test evidence.
