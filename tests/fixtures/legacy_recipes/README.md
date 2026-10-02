# Immutable recipe fixtures

These four byte-preserved JSON files are regression inputs from the pre-YAML
workflow, not supported training configurations. Tests use them to detect
unintended numerical changes. Their dated limitation notes remain historical.

Edit only the root `config.yaml` for current work and run `uv run rc-car`.
Runtime code must never load recipes from this directory. Frozen source bundles
and experiment results retain their original paths and bytes.
