# Team Run Workflow

Use `run_team.sh` to avoid output collisions and keep run metadata.

## Basic run

```bash
cd /work3/s240459/pypsa-eur-thesis
./run_team.sh --config config/Myruns/config.L3-short.yaml --scenario L3-short --jobs 8
```

This writes to:

- `results/<scenario>/<user>/<run_id>/...`
- `resources/<scenario>/<user>/<run_id>/...`
- `logs/<scenario>/<user>/<run_id>/...`

## Dry run

```bash
./run_team.sh --config config/Myruns/config.L3-short.yaml --scenario L3-short --dry-run -- --cores 1
```

## Required team rules

1. Use `run_team.sh` for all shared runs.
2. Do not run plain `run_L*.sh` into shared scenario paths.
3. Keep scenario config tracked and committed before running.
4. Keep `umask 0002` for both users.

## Metadata

Each run stores `run-metadata.json` in:

- `results/<scenario>/<user>/<run_id>/run-metadata.json`
- `logs/<scenario>/<user>/<run_id>/run-metadata.json`

Metadata includes run id, user, UTC timestamp, git commit, config path, and full snakemake command.

## Notes

- By default, the wrapper blocks untracked or modified config files. This enforces committed configs.
- You can bypass this with `--allow-dirty-config` when testing.
