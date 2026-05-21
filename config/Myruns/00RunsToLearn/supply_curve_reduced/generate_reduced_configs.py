"""
Generate reduced-resolution supply-curve configs from the main supply-curve set.

This keeps the economic assumptions from config/Myruns/supply_curve but changes
the computational levers that caused the 96x72 runs to stall:

- spatial clusters: 96 -> 41
- temporal aggregation: 72seg -> 24seg
- Gurobi crossover: auto/-1 -> 0
- per-solve time limit: 48h -> 12h

The generated run names end with "-r41-24seg" so they do not overwrite the
existing 96x72 results.
"""

from __future__ import annotations

import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "supply_curve"

CLUSTERS = 41
SECTOR_OPTS = "24seg"
TIME_LIMIT_SECONDS = 43_200
RUN_SUFFIX = f"-r{CLUSTERS}-{SECTOR_OPTS}"


def transform_config(text: str) -> str:
    text = re.sub(
        r'(run:\n\s+name:\s+")([^"]+)(")',
        rf"\1\2{RUN_SUFFIX}\3",
        text,
        count=1,
    )
    text = re.sub(r"clusters:\s+\[[0-9]+\]", f"clusters: [{CLUSTERS}]", text)
    text = re.sub(
        r"sector_opts:\s+\['[^']+'\]",
        f"sector_opts: ['{SECTOR_OPTS}']",
        text,
    )
    text = re.sub(
        r'resolution_sector:\s+"[^"]+"',
        f'resolution_sector: "{SECTOR_OPTS}"',
        text,
    )
    text = re.sub(r"crossover:\s+-?1", "crossover: 0", text)
    text = re.sub(r"TimeLimit:\s+[0-9]+", f"TimeLimit: {TIME_LIMIT_SECONDS}", text)

    banner = (
        "# Reduced supply-curve run generated from config/Myruns/supply_curve.\n"
        f"# Resolution: {CLUSTERS} clusters, {SECTOR_OPTS}; Gurobi crossover disabled.\n"
        "# Use for relative S0-vs-credit-price comparisons, not absolute capacity planning.\n"
    )
    return banner + text


def main() -> None:
    sources = sorted(SOURCE_DIR.glob("config.S*.yaml"))
    if not sources:
        raise SystemExit(f"No source configs found in {SOURCE_DIR}")

    for source in sources:
        target = HERE / source.name
        target.write_text(transform_config(source.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"wrote {target.relative_to(HERE.parent.parent.parent)}")

    print(
        f"\nGenerated {len(sources)} configs in {HERE} "
        f"with {CLUSTERS} clusters, {SECTOR_OPTS}, TimeLimit={TIME_LIMIT_SECONDS}s."
    )


if __name__ == "__main__":
    main()
