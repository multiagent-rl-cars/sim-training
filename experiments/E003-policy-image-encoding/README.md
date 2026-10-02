# E003 — Policy image encoding

Status: local implementation and independent review passed; GPU validation blocked
by the new payload's upload-approval requirement. No candidate source was uploaded
or executed remotely. Date: 2026-09-27.

The [preregistered protocol](protocol.md) isolates image amplitude from known
ego/history/reward defects. The full local gate passed 453 tests plus Ruff,
formatting and ty. Independent targeted review verified that the two task configs
change only four policy image-scale settings; raw helper/history/reward/action
and extractor bytes remain unchanged from the reference.

Reference: [E001](../E001-foundation-integration/README.md) measured actual network
image values of 0..1/255 in both tasks. The candidate should produce 0..1 without
changing raw helper maps or accepting incompatible old checkpoints.

## Ready-to-run identity and compatibility

The [prepared capsule](../../outputs/gpu-sessions/astra-images-v3-20260927/)
contains 168 source files and 14 byte-identical fixture files, with verified
checksums, extraction and a full local gate. Source provenance SHA256:
`5edc856964da36bc6e89856a08aedecf6e49e98cb38487837d863c86f0497256`.
The new behavior is `astra-foundation-v3-image-amplitude`; encoding is
`binary-255-sb3-normalized-v1`. Subsequent report edits are not part of that frozen
runtime snapshot. See [record.json](record.json).

The supported evaluator rejects old/missing manifest encoding before Kit and
checks the checkpoint marker/normalization after load. Inherited car checkpoint
loading in `scripts/sb3/` now fails explicitly, including for newly created
legacy-script checkpoints: use the versioned evaluator with its matching manifest,
or the original frozen source for old policies. Generic non-car behavior remains
unchanged. External custom calls to `PPO.load` are outside these entrypoint guards.

## Blocker and restart

Automatic tool review rejected transfer because prior explicit upload approval
covered only the original 153-file payload. Nothing from this candidate reached
the host. The reference session was backed up for deletion instead of retaining
idle compute indefinitely. Obtain explicit approval for this 168-file snapshot
and its destination before a future bounded deployment. Then run the four gates
and cross-reference invariants in the protocol; do not call this GPU-validated
or start a sweep based only on its local tests.
