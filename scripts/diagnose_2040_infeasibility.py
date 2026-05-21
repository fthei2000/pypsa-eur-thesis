#!/usr/bin/env python3
"""
Diagnose why the 2040 myopic solve is numerically failing.

Strategy:
  1. Load the brownfield 2040 network (the actual LP input).
  2. Reconstruct the full model using solve_network.create_optimization_model(),
     which adds all custom constraints (CDR credits, sequestration limits, etc.).
  3. Attempt to solve with *dual simplex* (method=1) instead of barrier.
     Simplex detects infeasibility / unboundedness definitively; barrier does not.
  4. If infeasible, compute the IIS via linopy and print the offending constraints.
  5. Regardless of outcome, print key constraint bounds and shadow prices.

Usage (from repo root, in the pixi env):
  pixi run python scripts/diagnose_2040_infeasibility.py [SCENARIO] [YEAR]

  SCENARIO  default: S0-no-cdr-revenue  (simplest, no CDR revenue)
  YEAR      default: 2040
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pypsa
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))  # needed for solve_network.py's "from scripts._X import ..."


# ---------------------------------------------------------------------------
# Helpers to load config and params the way Snakemake would
# ---------------------------------------------------------------------------

def load_config(scenario: str, config_family: str = "supply_curve") -> dict:
    """Load the merged config for a supply-curve scenario."""
    default_cfg_path = REPO / "config" / "config.default.yaml"
    scenario_cfg_path = REPO / "config" / "Myruns" / config_family / f"config.{scenario}.yaml"

    if not scenario_cfg_path.exists():
        # Try without the leading S-prefix mapping
        candidates = list((REPO / "config" / "Myruns" / config_family).glob(f"config.*{scenario}*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No config found for scenario '{scenario}'")
        scenario_cfg_path = candidates[0]

    with open(default_cfg_path) as f:
        config = yaml.safe_load(f)

    # merge plotting config
    plot_cfg_path = REPO / "config" / "plotting.default.yaml"
    if plot_cfg_path.exists():
        with open(plot_cfg_path) as f:
            plot_cfg = yaml.safe_load(f) or {}
        config.update(plot_cfg)

    with open(scenario_cfg_path) as f:
        overrides = yaml.safe_load(f) or {}

    def deep_update(base, override):
        """Recursively update base dict with override."""
        import copy
        result = copy.deepcopy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_update(result[k], v)
            else:
                result[k] = v
        return result

    return deep_update(config, overrides)


def get_nested(d, *keys, default=None):
    """Get a nested dict value with a default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
        if cur is None:
            return default
    return cur


def build_params(config: dict) -> dict:
    """Extract the params that Snakemake would pass to solve_network."""
    sector = config.get("sector", {})
    return {
        "solving": config.get("solving", {}),
        "foresight": config.get("foresight", "myopic"),
        "co2_sequestration_potential": sector.get("co2_sequestration_potential", 200),
        "cdr_credit_limit": sector.get("cdr_credit_limit_by_year", None),
        "cdr_credit_limit_by_year": sector.get("cdr_credit_limit_by_year", None),
        "cdr_credit_scope": sector.get("cdr_credit_scope", []),
        "cdr_credit_timing": sector.get("cdr_credit_timing", "capture"),
        "cdr_credit_price": sector.get("cdr_credit_price", 0.0),
        "cdr_credit_prices_by_scope": sector.get("cdr_credit_prices_by_scope", {}),
        "cdr_credit_standalone": sector.get("cdr_credit_standalone", False),
        "emission_prices_co2": get_nested(config, "costs", "emission_prices", "co2", default={}),
        "custom_extra_functionality": None,
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def resolve_run_name(scenario: str, config_family: str = "supply_curve") -> str:
    """Get the run name (directory name) from the config key."""
    config = load_config(scenario, config_family=config_family)
    return config.get("run", {}).get("name", scenario)


def run_diagnostic(
    scenario: str,
    year: int,
    config_family: str = "supply_curve",
    sector_opts: str = "24h",
) -> None:
    run_name = resolve_run_name(scenario, config_family=config_family)
    brownfield_path = (
        REPO / "resources" / run_name / "networks"
        / f"base_s_96__{sector_opts}_{year}_brownfield.nc"
    )
    if not brownfield_path.exists():
        raise FileNotFoundError(f"Brownfield network not found: {brownfield_path}")

    logger.info("Loading brownfield network: %s", brownfield_path)
    n = pypsa.Network(str(brownfield_path))
    logger.info(
        "Network loaded — buses: %d, links: %d, stores: %d, global_constraints: %d",
        len(n.buses), len(n.links), len(n.stores), len(n.global_constraints),
    )

    # --- Print existing global constraints for reference ---
    print("\n=== Global constraints in brownfield network ===")
    print(n.global_constraints[["type", "carrier_attribute", "sense", "constant"]].to_string())

    # --- Load config + params ---
    logger.info("Loading config for scenario '%s'", scenario)
    config = load_config(scenario, config_family=config_family)
    params = build_params(config)
    # Snakemake's Params object supports both attribute access and .get()
    class _Params(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)
        def __setattr__(self, key, value):
            self[key] = value

    n.config = config
    n.params = _Params(params)

    # --- Determine model kwargs (mirrors solve_network logic) ---
    solving = config.get("solving", {})
    model_kwargs = {
        "multi_investment_periods": False,
    }

    # Solver kwargs: use DUAL SIMPLEX for infeasibility detection
    # Simplex detects infeasibility/unboundedness definitively; barrier does not.
    solve_kwargs = {
        "solver_name": "gurobi",
        "solver_options": {
            "method": 1,          # dual simplex (reliable infeasibility detection)
            "threads": 8,
            "InfUnbdInfo": 1,     # compute infeasibility certificate if infeasible
            "NumericFocus": 3,
            "ScaleFlag": 2,
            "ObjScale": -0.5,
            "TimeLimit": 3600,    # 1-hour cap for the diagnostic
        },
        "log_fn": str(REPO / "logs" / f"diag_{scenario}_{year}_solver.log"),
    }

    # --- Import solve_network functions ---
    logger.info("Importing solve_network helpers…")
    from solve_network import create_optimization_model  # noqa: E402

    # --- Build the full LP (base + custom constraints) ---
    logger.info("Creating optimization model for year %s…", year)
    create_optimization_model(
        n=n,
        config=config,
        params=n.params,
        model_kwargs=model_kwargs,
        solve_kwargs=solve_kwargs,
        planning_horizons=str(year),
    )
    logger.info("Model built — variables: %d, constraints: %d",
                n.model.nvars, n.model.ncons)

    # --- Attempt solve with simplex ---
    logger.info("Solving with dual simplex (TimeLimit=3600s)…")
    status, condition = n.optimize.solve_model(**solve_kwargs)
    logger.info("Solver returned: status='%s', condition='%s'", status, condition)

    # --- Diagnose result ---
    if "infeasible" in (status + " " + condition).lower():
        print("\n" + "=" * 60)
        print("MODEL IS INFEASIBLE — computing IIS")
        print("=" * 60)
        try:
            labels = n.model.compute_infeasibilities()
            print("\nIrreducible Infeasible Subsystem (IIS):")
            n.model.print_infeasibilities()
        except Exception as exc:
            logger.warning("linopy IIS failed (%s); trying Gurobi directly", exc)
            _gurobi_iis_fallback(n, scenario, year)

    elif "optimal" in (status + " " + condition).lower():
        print("\n" + "=" * 60)
        print("MODEL IS FEASIBLE (simplex found optimum)")
        print("Objective: %.6e" % n.objective)
        print("=" * 60)
        print("\nConclusion: the infeasibility is NOT in the LP itself.")
        print("The barrier's numerical divergence must stem from ill-conditioning.")
        print("Try BarIterLimit increase, tighter tolerances, or model rescaling.")
        _print_shadow_prices(n)

    elif "unbounded" in (status + " " + condition).lower():
        print("\n" + "=" * 60)
        print("MODEL IS UNBOUNDED")
        print("=" * 60)
        print("The objective can decrease without bound.")
        print("Check for missing lower bounds on decision variables.")

    else:
        print(f"\nUnexpected solver outcome: status='{status}', condition='{condition}'")
        print("See solver log:", solve_kwargs["log_fn"])


def _gurobi_iis_fallback(n, scenario: str, year: int) -> None:
    """Write Gurobi IIS directly using the underlying gurobipy model."""
    try:
        grb = n.model.solver_model
        if grb is None:
            print("Gurobi model not available — cannot compute IIS directly.")
            return
        grb.computeIIS()
        iis_path = str(REPO / "logs" / f"diag_{scenario}_{year}.ilp")
        grb.write(iis_path)
        print(f"\nIIS written to: {iis_path}")
        print("\nConflicting constraints:")
        for c in grb.getConstrs():
            if c.IISConstr:
                print(f"  CONSTRAINT  {c.ConstrName}")
        for v in grb.getVars():
            if v.IISLB:
                print(f"  LOWER BOUND {v.VarName}")
            if v.IISUB:
                print(f"  UPPER BOUND {v.VarName}")
    except Exception as exc:
        print(f"Gurobi IIS fallback failed: {exc}")


def _print_shadow_prices(n) -> None:
    """Print global constraint shadow prices and binding status."""
    print("\n=== Global constraint shadow prices (solved network) ===")
    print(n.global_constraints[["type", "carrier_attribute", "sense", "constant", "mu"]].to_string())


def _check_constraint_feasibility(
    scenario: str,
    year: int,
    config_family: str = "supply_curve",
    sector_opts: str = "24h",
) -> None:
    """
    Quick analytical check: can the brownfield + constraint bounds be satisfied
    without solving the LP?  Checks obvious conflicts.
    """
    config = load_config(scenario, config_family=config_family)
    run_name = config.get("run", {}).get("name", scenario)
    brownfield_path = (
        REPO / "resources" / run_name / "networks"
        / f"base_s_96__{sector_opts}_{year}_brownfield.nc"
    )
    n = pypsa.Network(str(brownfield_path))
    sector = config.get("sector", {})

    print("\n=== Analytical constraint check ===")

    # CO2 sequestration potential — interpolate from dict or scalar
    def cfg_get(d, year):
        """Get value for year from a dict (interpolating) or return directly."""
        if not isinstance(d, dict):
            return float(d)
        years_sorted = sorted(int(k) for k in d)
        if year <= years_sorted[0]:
            return float(d[years_sorted[0]])
        if year >= years_sorted[-1]:
            return float(d[years_sorted[-1]])
        lo = max(y for y in years_sorted if y <= year)
        hi = min(y for y in years_sorted if y >= year)
        if lo == hi:
            return float(d[lo])
        frac = (year - lo) / (hi - lo)
        return float(d[lo]) + frac * (float(d[hi]) - float(d[lo]))

    seq_potential = sector.get("co2_sequestration_potential", 200)
    seq_limit = cfg_get(seq_potential, year)
    print(f"CO2 sequestration potential ({year}): {seq_limit} Mt/yr")

    # CDR credit demand cap
    cdr_limit_by_year = sector.get("cdr_credit_limit_by_year", {})
    cdr_limit = cdr_limit_by_year.get(year, None) if isinstance(cdr_limit_by_year, dict) else None
    print(f"CDR credit demand cap ({year}): {cdr_limit} Mt/yr")

    # Biomass potential
    biomass_factor = sector.get("solid_biomass_potential_factor", 1.0)
    print(f"Solid biomass potential factor: {biomass_factor}")

    # Brownfield p_nom lower bounds — sum by carrier type
    import pandas as pd
    import numpy as np

    print("\nBrownfield minimum capacities by carrier (p_nom_min > 0, MW):")
    link_mins = n.links[n.links["p_nom_min"] > 0].groupby("carrier")["p_nom_min"].sum()
    print(link_mins[link_mins > 100].sort_values(ascending=False).to_string())

    gen_mins = n.generators[n.generators["p_nom_min"] > 0].groupby("carrier")["p_nom_min"].sum()
    if not gen_mins.empty:
        print("\nGenerator minimum capacities (MW):")
        print(gen_mins[gen_mins > 100].sort_values(ascending=False).to_string())

    # -----------------------------------------------------------------------
    # LP COEFFICIENT SCALING CHECK
    # Biogas/biomass generators use p_nom = annual energy (MWh), not power (MW).
    # This creates huge LP coefficients that cause numerical instability.
    # -----------------------------------------------------------------------
    print("\n=== LP scaling check: generator p_nom distribution ===")
    gen_pnom = n.generators["p_nom"]
    print(f"p_nom range (MW): min={gen_pnom.min():.1f}, max={gen_pnom.max():.3e}, "
          f"median={gen_pnom.median():.1f}")
    print(f"Generators with p_nom > 1e6 MW (=1000 GW):")
    huge = n.generators[n.generators["p_nom"] > 1e6][["carrier", "p_nom"]].sort_values("p_nom", ascending=False)
    print(huge.groupby("carrier")["p_nom"].agg(["count", "max", "sum"]).sort_values("sum", ascending=False).head(15).to_string())

    print("\n=== LP scaling check: link p_nom_max distribution ===")
    link_pnom = n.links["p_nom_max"].replace(np.inf, np.nan).dropna()
    print(f"p_nom_max (finite) range (MW): min={link_pnom.min():.1f}, max={link_pnom.max():.3e}")

    # -----------------------------------------------------------------------
    # COMPARE 2030 SOLVED vs THIS YEAR'S BROWNFIELD biogas p_nom
    # -----------------------------------------------------------------------
    prev_year = year - 10  # 2030 for 2040
    prev_net_path = REPO / "results" / run_name / "networks" / f"base_s_96__{sector_opts}_{prev_year}.nc"
    if prev_net_path.exists():
        n_prev = pypsa.Network(str(prev_net_path))
        bio_carriers = ["biogas", "solid biomass", "unsustainable biogas",
                        "unsustainable solid biomass", "unsustainable bioliquids"]
        prev_bio = n_prev.generators[n_prev.generators.carrier.isin(bio_carriers)]
        curr_bio = n.generators[n.generators.carrier.isin(bio_carriers)]
        print(f"\n=== Biomass/biogas p_nom (MW): {prev_year} solved → {year} brownfield ===")
        prev_sum = prev_bio.groupby("carrier")["p_nom"].sum()
        curr_sum = curr_bio.groupby("carrier")["p_nom"].sum()
        comparison = pd.DataFrame({"prev": prev_sum, "curr": curr_sum}).fillna(0)
        comparison["ratio"] = comparison["curr"] / comparison["prev"].replace(0, np.nan)
        print(comparison.to_string())
        print(f"\nMax single-generator p_nom in {year} brownfield: "
              f"{curr_bio['p_nom'].max():.3e} MW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", default="S0-no-cdr-revenue",
                        help="Scenario name (default: S0-no-cdr-revenue)")
    parser.add_argument("year", nargs="?", type=int, default=2040,
                        help="Planning horizon year (default: 2040)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run analytical checks, do not solve LP")
    parser.add_argument("--config-family", default="supply_curve",
                        help="Subdirectory under config/Myruns containing config files")
    parser.add_argument("--sector-opts", default="24h",
                        help="Sector opts wildcard used in network file names")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Diagnosing 2040 infeasibility: scenario={args.scenario}, year={args.year}")
    print(f"{'='*60}\n")

    # Always run the quick analytical check first
    try:
        _check_constraint_feasibility(
            args.scenario,
            args.year,
            config_family=args.config_family,
            sector_opts=args.sector_opts,
        )
    except Exception as exc:
        logger.warning("Analytical check failed: %s", exc)

    if not args.check_only:
        run_diagnostic(
            args.scenario,
            args.year,
            config_family=args.config_family,
            sector_opts=args.sector_opts,
        )
