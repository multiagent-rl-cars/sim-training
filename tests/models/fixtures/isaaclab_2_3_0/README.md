# Pinned Isaac Lab boundary fixtures

These are unmodified upstream source files, not a replacement simulator.
CPU tests extract selected methods with AST and execute them against fake
environment state. This preserves boundary coverage on a fresh checkout without
installing Isaac Lab or silently skipping tests. GPU integration remains separate.

Source: NVIDIA Isaac Lab tag `v2.3.0`, commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`.

- `observation_manager.py`: upstream `source/isaaclab/isaaclab/managers/observation_manager.py`;
  SHA-256 `468ed890539f2de29dc0c27024b0ad74667209b884e0431456e1455c860cafae`.
- `sb3.py`: upstream `source/isaaclab_rl/isaaclab_rl/sb3.py`;
  SHA-256 `fee16bf9acf58b802e9a2d483ac90769a661dc745f7d504ab222eeb2fec1d9e4`.

Both files are byte-identical to the corresponding files in the official
`isaaclab-2.3.0-cp311-none-manylinux_2_35_x86_64.whl` published on PyPI
(wheel SHA-256 `6fa943792f3771e6358023af1f8a88d486afbc126a774db09c4bd4123756db74`).
The original BSD-3-Clause notices are preserved, with the upstream license in
`LICENSE`. Do not format, patch, or import these fixtures. A deliberate upstream
upgrade must update the source revision, byte hashes, lock, and boundary tests
together; unchanged hashes are enforced by the tests.
