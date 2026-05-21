#!/usr/bin/env python3
"""Audit supply-curve run completeness and CDR accounting consistency."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_YEARS = (2030, 2040, 2050)
DEFAULT_PATTERN = "S??t-cdr-*-168seg"
TOLERANCE_MT = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan supply-curve result folders and flag solver/accounting issues."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results directory to scan. Default: results",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern below --results. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Planning horizons to audit. Default: 2030 2040 2050",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path. Prints CSV to stdout when omitted.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return data or {}


def get_nested(data: dict, *keys, default=None):
    out = data
    for key in keys:
        if not isinstance(out, dict) or key not in out:
            return default
        out = out[key]
    return out


def read_config_for_year(
    result_dir: Path, year: int, fallback: dict | None = None
) -> dict:
    config_path = result_dir / "configs" / f"config.base_s_96__168seg_{year}.yaml"
    return read_yaml(config_path) or (fallback or {})


def read_fallback_config(result_dir: Path) -> dict:
    for config_path in sorted(
        (result_dir / "configs").glob("config.base_s_96__*.yaml")
    ):
        config = read_yaml(config_path)
        if config:
            return config
    return {}


def read_metric(path: Path) -> dict:
    if not path.exists():
        return {}
    series = pd.read_csv(path, index_col=0).iloc[:, 0]
    return {
        "total_costs": series.get("total costs"),
        "electricity_price_mean": series.get("electricity_price_mean"),
        "electricity_price_zero_hours": series.get("electricity_price_zero_hours"),
        "line_volume": series.get("line_volume"),
        "co2_storage_shadow": series.get("co2_storage_shadow"),
    }


def read_cdr_accounting(path: Path) -> dict:
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0].to_dict()
    return {f"cdr_{key}": value for key, value in row.items()}


def parse_solver_log(path: Path) -> dict:
    out = {
        "solver_log_exists": path.exists(),
        "solver_optimal": False,
        "solver_interrupted": False,
        "solver_aborted": False,
        "solver_infeasible": False,
        "solver_status_line": "",
        "solver_termination": "",
        "solver_objective": np.nan,
    }
    if not path.exists():
        return out

    text = path.read_text(errors="ignore")
    out["solver_optimal"] = "Optimal objective" in text
    out["solver_interrupted"] = "Solve interrupted" in text
    out["solver_aborted"] = "Status: aborted" in text
    out["solver_infeasible"] = "infeasible" in text.lower()

    status_matches = re.findall(r"Status:\s*([^\n]+)", text)
    if status_matches:
        out["solver_status_line"] = status_matches[-1].strip()

    term_matches = re.findall(r"termination condition '([^']+)'", text)
    if term_matches:
        out["solver_termination"] = term_matches[-1].strip()

    obj_matches = re.findall(
        r"Optimal objective\s+([+-]?[0-9.]+e[+-]?[0-9]+)", text
    )
    if obj_matches:
        out["solver_objective"] = float(obj_matches[-1])
    return out


def to_float(value, default=np.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_flags(row: dict) -> str:
    flags = []
    has_network = bool(row.get("has_network"))
    solver_log_exists = bool(row.get("solver_log_exists"))
    solver_optimal = bool(row.get("solver_optimal"))

    if not has_network:
        flags.append("missing_network")
    if solver_log_exists and not solver_optimal:
        flags.append("solver_not_optimal")
    if row.get("solver_interrupted"):
        flags.append("solver_interrupted")
    if row.get("solver_aborted"):
        flags.append("solver_aborted")
    if row.get("solver_infeasible"):
        flags.append("solver_mentions_infeasible")

    timing = row.get("cdr_credit_timing_config") or row.get("cdr_cdr_credit_timing")
    method = row.get("cdr_method")
    if timing == "sequestration" and method == "capture_proxy":
        flags.append("cdr_capture_proxy_for_sequestration")
    if has_network and timing == "sequestration" and not method:
        flags.append("missing_cdr_accounting")

    credited = to_float(row.get("cdr_credited_total_mtco2_per_yr"))
    limit = to_float(row.get("cdr_credit_limit_mtco2_per_yr"))
    physical = to_float(row.get("cdr_physical_sequestration_mtco2_per_yr"))
    attributed = to_float(row.get("cdr_attributed_total_mtco2_per_yr"))
    capture_proxy = to_float(row.get("cdr_capture_proxy_total_mtco2_per_yr"))

    if not np.isnan(limit) and credited > limit + TOLERANCE_MT:
        flags.append("cdr_above_credit_limit")
    if not np.isnan(capture_proxy) and credited > capture_proxy + TOLERANCE_MT:
        flags.append("cdr_above_capture_proxy")
    if not np.isnan(attributed) and credited > attributed + TOLERANCE_MT:
        flags.append("cdr_above_attributed")
    if not np.isnan(attributed) and not np.isnan(physical):
        if attributed > physical + TOLERANCE_MT:
            flags.append("attributed_above_physical_sequestration")
    if np.isnan(attributed) and timing == "sequestration" and method:
        flags.append("missing_attributed_cdr")

    return ";".join(flags)


def audit(results_dir: Path, pattern: str, years: list[int]) -> pd.DataFrame:
    rows = []
    for result_dir in sorted(results_dir.glob(pattern)):
        if not result_dir.is_dir():
            continue
        price_match = re.search(r"cdr-(\d+)eur", result_dir.name)
        price = int(price_match.group(1)) if price_match else np.nan
        fallback_config = read_fallback_config(result_dir)

        for year in years:
            config = read_config_for_year(result_dir, year, fallback=fallback_config)
            sector = config.get("sector", {})
            row = {
                "scenario": result_dir.name,
                "price_eur_per_tco2_name": price,
                "planning_horizon": year,
                "has_network": (
                    result_dir / "networks" / f"base_s_96__168seg_{year}.nc"
                ).exists(),
                "config_exists": bool(config),
                "cdr_credit_timing_config": get_nested(
                    config, "sector", "cdr_credit_timing"
                ),
                "cdr_credit_standalone_config": get_nested(
                    config, "sector", "cdr_credit_standalone"
                ),
                "cdr_credit_limit_config_mtco2_per_yr": get_nested(
                    sector.get("cdr_credit_limit_by_year", {}), year
                ),
            }
            row.update(
                parse_solver_log(
                    result_dir / "logs" / f"base_s_96__168seg_{year}_solver.log"
                )
            )
            row.update(
                read_metric(
                    result_dir
                    / "csvs"
                    / "individual"
                    / f"metrics_s_96__168seg_{year}.csv"
                )
            )
            row.update(
                read_cdr_accounting(
                    result_dir
                    / "csvs"
                    / "individual"
                    / f"cdr_credit_accounting_s_96__168seg_{year}.csv"
                )
            )
            row["flags"] = build_flags(row)
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = audit(args.results, args.pattern, args.years)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} audit rows to {args.output}")
    else:
        print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
