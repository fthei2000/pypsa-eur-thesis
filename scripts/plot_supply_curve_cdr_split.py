from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_PDF = ROOT / "notebooks" / "eval_cdr_split_by_price_year.pdf"
OUT_PNG = ROOT / "notebooks" / "eval_cdr_split_by_price_year.png"
OUT_CSV = ROOT / "notebooks" / "eval_cdr_split_by_price_year.csv"
RUN_PATTERN = re.compile(
    r"^S(?P<scenario>\d+)-cdr-(?P<price>\d+)eur(?:-(?P<family>r\d+-\d+seg))?$"
)
ZERO_RUN_PATTERN = re.compile(r"^S0-no-cdr-revenue(?:-(?P<family>r\d+-\d+seg))?$")
BIOGENIC_TOKENS = ("biomass", "biogas", "biosng", "btl", "fuelwood", "msw", "bio")


def classify_origin(carrier_name: str) -> str:
    carrier = str(carrier_name).lower()
    if "dac" in carrier:
        return "DAC"
    if any(token in carrier for token in BIOGENIC_TOKENS):
        return "BECCS"
    return "fossil"


def annual_capture_by_origin(network: pypsa.Network) -> dict[str, float]:
    buses = network.buses
    stored_buses = set(buses.index[buses.carrier.astype(str).str.startswith("co2 stored")])
    weights = network.snapshot_weightings.generators
    out = {"DAC": 0.0, "BECCS": 0.0}

    for link_name in network.links.index:
        origin = classify_origin(network.links.at[link_name, "carrier"])
        if origin not in out:
            continue

        coeff = 0.0
        for idx in [1, 2, 3, 4]:
            bus_col = f"bus{idx}"
            eff_col = f"efficiency{idx}" if idx > 1 else "efficiency"
            if bus_col not in network.links.columns or eff_col not in network.links.columns:
                continue
            bus = network.links.at[link_name, bus_col]
            eff = network.links.at[link_name, eff_col]
            if pd.notna(bus) and bus in stored_buses and pd.notna(eff) and eff > 0:
                coeff += float(eff)

        if coeff <= 0:
            continue

        dispatch = network.links_t.p0[link_name]
        out[origin] += float((dispatch * coeff * weights).sum() / 1e6)

    return out


def annual_sequestration_by_origin(network: pypsa.Network) -> dict[str, float]:
    weights = network.snapshot_weightings.generators
    out = {"DAC": 0.0, "BECCS": 0.0}
    found = False

    for link_name in network.links.index:
        carrier = str(network.links.at[link_name, "carrier"]).lower()
        if not carrier.startswith("co2 sequestered "):
            continue
        found = True
        origin = carrier.replace("co2 sequestered ", "", 1).strip()
        key = "DAC" if origin == "dac" else "BECCS" if origin == "biogenic" else None
        if key is None:
            continue
        dispatch = network.links_t.p0[link_name]
        out[key] += float((dispatch * weights).sum() / 1e6)

    if not found:
        return {}
    return out


def parse_run_metadata(run_name: str) -> dict[str, str | int] | None:
    zero_match = ZERO_RUN_PATTERN.match(run_name)
    if zero_match:
        family = zero_match.group("family") or "latest"
        return {
            "scenario": "S0-no-cdr-revenue",
            "price": 0,
            "family": family,
        }

    match = RUN_PATTERN.match(run_name)
    if not match:
        return None

    family = match.group("family") or "latest"
    return {
        "scenario": f"S{int(match.group('scenario')):02d}-cdr-{int(match.group('price')):03d}eur",
        "price": int(match.group("price")),
        "family": family,
    }


def latest_supply_curve_runs() -> list[tuple[Path, dict[str, str | int]]]:
    candidates: dict[int, tuple[float, Path, dict[str, str | int]]] = {}
    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        metadata = parse_run_metadata(run_dir.name)
        if metadata is None:
            continue
        mtime = run_dir.stat().st_mtime
        price = int(metadata["price"])
        current = candidates.get(price)
        if current is None or mtime > current[0]:
            candidates[price] = (mtime, run_dir, metadata)
    return [(run_dir, metadata) for _, run_dir, metadata in sorted(candidates.values(), key=lambda item: int(item[2]["price"]))]


def accounting_csv_for_network(network_path: Path) -> Path:
    filename = f"cdr_credit_accounting_{network_path.stem.replace('base_', '')}.csv"
    return network_path.parents[1] / "csvs" / "individual" / filename


def accounting_values(network_path: Path) -> tuple[dict[str, float], str] | None:
    csv_path = accounting_csv_for_network(network_path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    row = df.iloc[0]
    dac = float(row.get("credited_dac_mtco2_per_yr", 0.0))
    beccs = float(row.get("credited_biogenic_mtco2_per_yr", 0.0))
    return {"DAC": dac, "BECCS": beccs}, str(row.get("method", "solver_export"))


def collect() -> pd.DataFrame:
    rows = []
    for run_dir, metadata in latest_supply_curve_runs():
        scenario = str(metadata["scenario"])
        price = int(metadata["price"])
        family = str(metadata["family"])
        for network_path in sorted((run_dir / "networks").glob("base_s_*_*.nc")):
            year_match = re.search(r"_(?P<year>20\d\d)\.nc$", network_path.name)
            if not year_match:
                continue
            year = int(year_match.group("year"))
            accounting = accounting_values(network_path)
            if accounting is not None:
                values, method = accounting
            else:
                network = pypsa.Network(network_path)
                if getattr(network, "objective", None) is None:
                    logger.info("Skipping unsolved network %s", network_path)
                    continue
                values = annual_sequestration_by_origin(network)
                method = "sequestration_proxy" if values else "capture_proxy"
                if not values:
                    values = annual_capture_by_origin(network)
            rows.append(
                {
                    "scenario": scenario,
                    "run": run_dir.name,
                    "family": family,
                    "price_eur_per_t": price,
                    "year": year,
                    "DAC_MtCO2_per_yr": values["DAC"],
                    "BECCS_MtCO2_per_yr": values["BECCS"],
                    "Total_MtCO2_per_yr": values["DAC"] + values["BECCS"],
                    "method": method,
                }
            )
    if not rows:
        raise SystemExit("No supply-curve result networks found under results/.")
    df = pd.DataFrame(rows).sort_values(["year", "price_eur_per_t"])
    return df


def plot(df: pd.DataFrame) -> None:
    years = sorted(df["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(5.2 * len(years), 5.2), sharey=True)
    if len(years) == 1:
        axes = [axes]

    colors = {"DAC": "#1f77b4", "BECCS": "#2ca02c"}

    for ax, year in zip(axes, years):
        sub = df[df["year"] == year].sort_values("price_eur_per_t")
        x = sub["price_eur_per_t"].astype(str)
        dac = sub["DAC_MtCO2_per_yr"]
        beccs = sub["BECCS_MtCO2_per_yr"]
        total = dac + beccs

        ax.bar(x, beccs, color=colors["BECCS"], label="BECCS")
        ax.bar(x, dac, bottom=beccs, color=colors["DAC"], label="DAC")
        ax.plot(x, total, color="black", marker="o", linewidth=1.5, label="Total")
        ax.set_title(f"{year}")
        ax.set_xlabel("CDR credit price [EUR/tCO2]")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Deployed CDR [MtCO2/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle("CDR Deployment Split by Credit Price and Year", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")


def main() -> None:
    df = collect()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    plot(df)
    print(df.to_csv(index=False))
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
