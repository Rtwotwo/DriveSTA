# DriveSTA release validation

Validated on 18 November 2026 using Python 3.9, PyTorch 2.5.1 + CUDA 12.4 and the server's existing NAVSIM data. This records the current release refactor, independently of prior experiment results.

| Check | Result |
| --- | --- |
| Existing model/data/mining/distributed-resume regression suite | 411 passed |
| New download-integrity and immutable-index tests | 4 passed |
| Final targeted release-boundary suite (including those four) | 37 passed |
| Four training and four evaluation CLI configuration resolutions | Passed |
| Four real-data/mixed-data training preflights | Passed |
| Real NAVTEST forward comparison, all four checkpoints | Strict load; trajectory, all proposals and scores exactly equal; maximum absolute error 0 |
| Single-scene inference CLI | Saved eight finite trajectory poses as JSON |
| Single-scene official PDMS, proposal and refined model | Complete: one expected/succeeded scene, zero failed/missing scenes |
| All 55 checkpoint/backbone/archive files | Same underlying files as the original release; tensor/archive bytes unchanged |
| Four training recipes | Numeric parameters unchanged after Hydra reorganization |
| Package build | DriveSTA wheel built successfully |

The forward and PDMS smoke checks used NAVTEST token `10909749099354e6`. They validate the execution and serialization changes, not full benchmark performance. The training preflights verified the supplied 103,263-scene real and 232,672-scene mixed recipes against existing external data. Full training and full benchmark evaluation were not rerun, and a fresh-machine environment installation was not performed.

Runtime dependency deprecation warnings were present; they did not cause test failures. Historical serialized keys and immutable archive provenance remain supported in the compatibility layer, and original third-party notices are retained.

Machine-readable evidence: [validation.json](release/validation.json), [model_parity.json](release/model_parity.json), and [recipe_equivalence.json](release/recipe_equivalence.json).
