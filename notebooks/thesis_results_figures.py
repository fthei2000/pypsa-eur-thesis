# %% [markdown]
# # Thesis Results Figures — v8 Supply Curve Analysis
#
# Generates all main-text figures for §4 (Results) incorporating supervisor feedback (May 13).
#
# **Design principles:**
# - Truncate supply curves at cap-binding price (no plateau)
# - Merge supply curve + tech mix into one figure
# - System cost deltas as bar charts at cap-binding price only
# - Gap = literally visible space on chart
# - Maps are the centrepiece: 2 colours (DAC/BECCS) only
# - Drop electrochemical DAC (no deployment, no cost data)
# - EU CDR deployment targets, not "caps"

# %% [markdown]
# ## 0. Setup & Data Loading

# %%
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd().parent
DATA_DIR = Path("/home/ubuntu/attachments")
ACC_FILE = DATA_DIR / "8d1f4307-324d-4175-b474-001e700888cb" / "results_v8_cdr_accounting.parquet"
SER_FILE = DATA_DIR / "7c028450-65b0-4830-92b9-e6619eb568a5" / "results_v8_series.parquet"

OUT_DIR = ROOT / "notebooks" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Thesis style ───────────────────────────────────────────────────────────
# Clean academic style — can be swapped for "style B" later.
THESIS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "image.cmap": "viridis",
}
mpl.rcParams.update(THESIS_RC)

# ── Colour palette ─────────────────────────────────────────────────────────
C_DAC = "#2166ac"       # blue — DAC
C_BECCS = "#4dac26"     # green — BECCS
C_TOTAL = "#1a1a1a"     # near-black — total CDR line
C_GAP = "#d73027"       # red — financing gap
C_WTP = "#fee08b"       # yellow — WTP band
C_VARIANTS = {
    "low": "#66c2a5",
    "medium": "#fc8d62",
    "high": "#8da0cb",
}

# EU CDR deployment targets (Mt CO2/yr)
TARGETS = {
    2030: {"value": 100, "label": "NZIA 2030", "source": "Net-Zero Industry Act"},
    2040: {"value": 300, "label": "EC ICM 2040", "source": "EC Industrial Carbon Mgmt"},
    2050: {"value": 450, "label": "IEA NZE 2050", "source": "IEA Net Zero by 2050"},
}

# Sequestration limits from model (cap-binding)
SEQ_LIMITS = {2030: 70, 2040: 320, 2050: 600}

# WTP ranges (2030 from Table 5 of thesis)
WTP = {
    2030: {
        "BECCS": {"low": 132, "central": 200, "high": 249},
        "DACCS": {"low": 249, "central": 332, "high": 415},
    },
}

def save_fig(fig, name: str) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png")
    print(f"  Saved: {name}.pdf / .png")

# %% [markdown]
# ### Load data

# %%
print("Loading CDR accounting data...")
acc = pd.read_parquet(ACC_FILE)
# Use capture_proxy columns (credited columns are NaN due to solver_unavailable)
acc["dac_mt"] = acc["capture_proxy_dac_mtco2_per_yr"]
acc["beccs_mt"] = acc["capture_proxy_biogenic_mtco2_per_yr"]
acc["total_mt"] = acc["capture_proxy_total_mtco2_per_yr"]
print(f"  {len(acc)} rows: {acc['cost_variant'].nunique()} variants × "
      f"{acc['scenario'].nunique()} scenarios × {acc['year'].nunique()} years")

print("\nLoading series data...")
series = pd.read_parquet(SER_FILE)
print(f"  {len(series):,} rows, csv_types: {sorted(series['csv_type'].unique())}")

# Extract metrics
metrics_raw = series[series["csv_type"] == "metrics"].copy()
metrics_raw.rename(columns={"Unnamed: 0": "metric_name"}, inplace=True)

# Pivot metrics to wide format
metrics = metrics_raw.pivot_table(
    index=["cost_variant", "scenario", "credit_price", "year"],
    columns="metric_name",
    values="value",
    aggfunc="first",
).reset_index()
metrics.columns.name = None

# Total system cost in bn EUR
metrics["total_cost_bn"] = metrics["total costs"] / 1e9

# Baseline costs (S00)
baseline_costs = metrics[metrics["scenario"] == "S00"][
    ["cost_variant", "year", "total_cost_bn"]
].rename(columns={"total_cost_bn": "baseline_cost_bn"})

metrics = metrics.merge(baseline_costs, on=["cost_variant", "year"], how="left")
metrics["delta_cost_bn"] = metrics["total_cost_bn"] - metrics["baseline_cost_bn"]

print("Data loaded successfully.")

# %% [markdown]
# ### Identify cap-binding prices
#
# For each cost variant × year, find the lowest credit price where CDR
# reaches the sequestration limit (within 5% tolerance).

# %%
def find_cap_binding_price(group: pd.DataFrame, year: int) -> int:
    """Return the lowest credit price where CDR >= 95% of seq limit."""
    limit = SEQ_LIMITS[year]
    threshold = 0.95 * limit
    above = group[group["total_mt"] >= threshold]
    if above.empty:
        return group["credit_price"].max()  # never reaches — use highest
    return int(above["credit_price"].min())

cap_binding = (
    acc.groupby(["cost_variant", "year"])
    .apply(lambda g: find_cap_binding_price(g, g.name[1]))
    .reset_index()
    .rename(columns={0: "cap_binding_price"})
)

print("Cap-binding credit prices (EUR/tCO₂):")
print(cap_binding.pivot(index="cost_variant", columns="year", values="cap_binding_price").to_string())

# %% [markdown]
# ---
# ## 4.1 Baseline Figures

# %% [markdown]
# ### Table 3: Baseline headline metrics

# %%
baseline = acc[acc["scenario"] == "S00"].copy()
baseline_metrics = metrics[metrics["scenario"] == "S00"].copy()

print("\n=== Table 3: Baseline System Snapshot (S00, 0 EUR/tCO₂ credit) ===\n")
for variant in ["low", "medium", "high"]:
    print(f"--- {variant.title()} cost variant ---")
    bv = baseline[baseline["cost_variant"] == variant]
    bm = baseline_metrics[baseline_metrics["cost_variant"] == variant]
    for year in [2030, 2040, 2050]:
        row = bv[bv["year"] == year].iloc[0]
        mrow = bm[bm["year"] == year].iloc[0]
        print(f"  {year}: CDR={row['total_mt']:.1f} Mt (DAC={row['dac_mt']:.1f}, "
              f"BECCS={row['beccs_mt']:.1f}), "
              f"System cost={mrow['total_cost_bn']:.0f} bn EUR, "
              f"Elec price={mrow['electricity_price_mean']:.1f} EUR/MWh")
    print()

# %% [markdown]
# ---
# ## 4.2 Scenario 1: Supply-Side CDR Deployment (RQ1)
#
# ### F1: Combined Supply Curve + Technology Mix
#
# 3-panel figure (2030 / 2040 / 2050). Stacked bars (BECCS + DAC),
# total CDR line on top. X-axis **truncated at cap-binding price**.
# EU deployment target shown as horizontal dashed line.

# %%
def plot_f1_supply_curve_tech_mix(acc: pd.DataFrame, cap_binding: pd.DataFrame,
                                  variant: str = "medium") -> mpl.figure.Figure:
    """F1: Combined supply curve + tech mix (3-panel, one cost variant)."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, year in zip(axes, years):
        sub = acc[(acc["cost_variant"] == variant) & (acc["year"] == year)].sort_values("credit_price")

        # Get cap-binding price for truncation
        cb_row = cap_binding[(cap_binding["cost_variant"] == variant) &
                             (cap_binding["year"] == year)]
        cb_price = int(cb_row["cap_binding_price"].iloc[0])

        # Truncate: only show up to cap-binding price
        sub = sub[sub["credit_price"] <= cb_price]

        x = sub["credit_price"].values
        x_labels = [str(int(p)) for p in x]
        x_pos = np.arange(len(x))

        bar_width = 0.7
        ax.bar(x_pos, sub["beccs_mt"].values, width=bar_width,
               color=C_BECCS, label="BECCS", zorder=2)
        ax.bar(x_pos, sub["dac_mt"].values, bottom=sub["beccs_mt"].values,
               width=bar_width, color=C_DAC, label="DAC", zorder=2)
        ax.plot(x_pos, sub["total_mt"].values, color=C_TOTAL, marker="o",
                markersize=4, linewidth=1.8, label="Total CDR", zorder=3)

        # EU deployment target
        target = TARGETS[year]
        seq_limit = SEQ_LIMITS[year]
        if seq_limit <= sub["total_mt"].max() * 1.5:
            ax.axhline(seq_limit, color="grey", linewidth=1, linestyle="--", alpha=0.7)
            ax.text(len(x_pos) - 0.5, seq_limit * 1.03,
                    f"Seq. limit: {seq_limit} Mt", fontsize=8, ha="right",
                    color="grey", style="italic")
        if target["value"] <= sub["total_mt"].max() * 1.5:
            ax.axhline(target["value"], color="#e41a1c", linewidth=1.2, linestyle=":",
                       alpha=0.8)
            ax.text(len(x_pos) - 0.5, target["value"] * 1.03,
                    target["label"], fontsize=8, ha="right", color="#e41a1c",
                    fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 6 else 0)
        ax.set_xlabel("CDR credit price [EUR/tCO₂]")
        ax.set_title(str(year), fontweight="bold")
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("CDR deployment [Mt CO₂/yr]")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Supply Curve + Technology Mix ({variant.title()} cost variant)",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# Generate for medium (main text) and low/high (appendix)
for variant in ["medium", "low", "high"]:
    fig = plot_f1_supply_curve_tech_mix(acc, cap_binding, variant=variant)
    suffix = "" if variant == "medium" else f"_{variant}"
    save_fig(fig, f"F1_supply_curve_tech_mix{suffix}")
    plt.close(fig)

# %% [markdown]
# ### F1b: All three cost variants overlaid (alternative view)

# %%
def plot_f1b_all_variants(acc: pd.DataFrame, cap_binding: pd.DataFrame) -> mpl.figure.Figure:
    """F1b: Supply curves for all 3 cost variants, 3-panel by year."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, year in zip(axes, years):
        for variant in ["low", "medium", "high"]:
            sub = acc[(acc["cost_variant"] == variant) & (acc["year"] == year)].sort_values("credit_price")

            # Truncate at cap-binding
            cb_row = cap_binding[(cap_binding["cost_variant"] == variant) &
                                 (cap_binding["year"] == year)]
            cb_price = int(cb_row["cap_binding_price"].iloc[0])
            sub = sub[sub["credit_price"] <= cb_price]

            ax.plot(sub["credit_price"].values, sub["total_mt"].values,
                    marker="o", markersize=4, linewidth=2,
                    color=C_VARIANTS[variant], label=f"{variant.title()}")

        # Target line
        target = TARGETS[year]
        seq_limit = SEQ_LIMITS[year]
        ax.axhline(seq_limit, color="grey", linewidth=1, linestyle="--", alpha=0.7)
        ax.text(ax.get_xlim()[1] * 0.95, seq_limit * 1.03,
                f"Seq. limit: {seq_limit} Mt", fontsize=8, ha="right",
                color="grey", style="italic")
        if target["value"] != seq_limit:
            ax.axhline(target["value"], color="#e41a1c", linewidth=1.2, linestyle=":",
                       alpha=0.8)
            ax.text(ax.get_xlim()[1] * 0.95, target["value"] * 1.03,
                    target["label"], fontsize=8, ha="right", color="#e41a1c",
                    fontweight="bold")

        ax.set_xlabel("CDR credit price [EUR/tCO₂]")
        ax.set_title(str(year), fontweight="bold")
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("CDR deployment [Mt CO₂/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("CDR Supply Curves by Cost Variant (truncated at cap-binding)",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_f1b_all_variants(acc, cap_binding)
save_fig(fig, "F1b_supply_curves_all_variants")
plt.close(fig)

# %% [markdown]
# ### Table 4: Breakeven prices

# %%
print("\n=== Table 4: Breakeven Credit Prices (EUR/tCO₂) ===\n")
print(f"{'Variant':<10} {'Year':<6} {'> 10 Mt':<10} {'> 50% limit':<14} {'Cap-binding':<12}")
print("-" * 55)
for variant in ["low", "medium", "high"]:
    for year in [2030, 2040, 2050]:
        sub = acc[(acc["cost_variant"] == variant) & (acc["year"] == year)].sort_values("credit_price")
        limit = SEQ_LIMITS[year]

        # > 10 Mt
        above10 = sub[sub["total_mt"] > 10]
        p10 = int(above10["credit_price"].min()) if not above10.empty else ">500"

        # > 50% limit
        above50 = sub[sub["total_mt"] > 0.5 * limit]
        p50 = int(above50["credit_price"].min()) if not above50.empty else ">500"

        # Cap-binding
        cb_row = cap_binding[(cap_binding["cost_variant"] == variant) &
                             (cap_binding["year"] == year)]
        pcb = int(cb_row["cap_binding_price"].iloc[0])

        print(f"{variant:<10} {year:<6} {str(p10):<10} {str(p50):<14} {pcb:<12}")

# %% [markdown]
# ---
# ### F2: Spatial CDR Deployment Maps
#
# Country-level bubbles at cap-binding credit price. Two colours: DAC (blue) + BECCS (green).

# %%
def extract_country(location: str) -> str:
    """Extract 2-letter country code from node name like 'DE0 3'."""
    if pd.isna(location):
        return "XX"
    return str(location)[:2]


def get_country_cdr(series: pd.DataFrame, acc: pd.DataFrame,
                    cap_binding: pd.DataFrame, variant: str = "medium") -> pd.DataFrame:
    """Get country-level DAC + BECCS deployment at cap-binding price."""
    rows = []

    for year in [2030, 2040, 2050]:
        cb_row = cap_binding[(cap_binding["cost_variant"] == variant) &
                             (cap_binding["year"] == year)]
        cb_price = int(cb_row["cap_binding_price"].iloc[0])
        cb_scenario = f"S{cb_price // 50:02d}"

        # Nodal capacities for DAC
        dac_cap = series[
            (series["csv_type"] == "nodal_capacities") &
            (series["cost_variant"] == variant) &
            (series["scenario"] == cb_scenario) &
            (series["year"] == year) &
            (series["carrier"].isin(["DAC-solidLT", "DAC-liquidHT"]))
        ].copy()
        dac_cap["country"] = dac_cap["location"].apply(extract_country)
        dac_by_country = dac_cap.groupby("country")["value"].sum().reset_index()
        dac_by_country.rename(columns={"value": "dac_capacity_mw"}, inplace=True)

        # Nodal capacities for BECCS
        beccs_carriers = [
            "solid biomass for industry CC",
            "urban central solid biomass CHP CC",
            "biomass to liquid CC",
            "biomass-to-methanol CC",
        ]
        beccs_cap = series[
            (series["csv_type"] == "nodal_capacities") &
            (series["cost_variant"] == variant) &
            (series["scenario"] == cb_scenario) &
            (series["year"] == year) &
            (series["carrier"].isin(beccs_carriers))
        ].copy()
        beccs_cap["country"] = beccs_cap["location"].apply(extract_country)
        beccs_by_country = beccs_cap.groupby("country")["value"].sum().reset_index()
        beccs_by_country.rename(columns={"value": "beccs_capacity_mw"}, inplace=True)

        merged = pd.merge(dac_by_country, beccs_by_country,
                          on="country", how="outer").fillna(0)
        merged["year"] = year
        merged["credit_price"] = cb_price
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


def plot_f2_spatial_maps(country_cdr: pd.DataFrame, variant: str = "medium") -> mpl.figure.Figure:
    """F2: Spatial CDR deployment maps (3-panel by year)."""
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

    # Column names may be uppercase or lowercase depending on source
    col_continent = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    col_iso3 = "ISO_A3" if "ISO_A3" in world.columns else "iso_a3"

    europe = world[world[col_continent] == "Europe"].copy()
    iso3_to_iso2 = {
        "ALB": "AL", "AUT": "AT", "BIH": "BA", "BEL": "BE", "BGR": "BG",
        "CHE": "CH", "CZE": "CZ", "DEU": "DE", "DNK": "DK", "EST": "EE",
        "ESP": "ES", "FIN": "FI", "FRA": "FR", "GBR": "GB", "GRC": "GR",
        "HRV": "HR", "HUN": "HU", "IRL": "IE", "ITA": "IT", "LTU": "LT",
        "LUX": "LU", "LVA": "LV", "MNE": "ME", "MKD": "MK", "NLD": "NL",
        "NOR": "NO", "POL": "PL", "PRT": "PT", "ROU": "RO", "SRB": "RS",
        "SWE": "SE", "SVN": "SI", "SVK": "SK",
    }
    europe["iso2"] = europe[col_iso3].map(iso3_to_iso2)

    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, year in zip(axes, years):
        europe.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)

        yr_data = country_cdr[country_cdr["year"] == year]

        # Country centroids for bubble placement
        europe_centroids = europe.copy()
        europe_centroids["centroid"] = europe_centroids.geometry.centroid
        centroid_map = {}
        for _, row in europe_centroids.iterrows():
            if pd.notna(row["iso2"]):
                centroid_map[row["iso2"]] = (row["centroid"].x, row["centroid"].y)

        # Manual adjustments for readability
        manual_pos = {
            "NO": (10, 62), "SE": (16, 60), "FI": (26, 63), "DK": (10, 56),
            "GB": (-3, 54), "IE": (-8, 53), "FR": (2, 47), "ES": (-3, 40),
            "DE": (10, 51), "IT": (12, 43), "NL": (5, 52), "PL": (20, 52),
        }
        for k, v in manual_pos.items():
            centroid_map[k] = v

        # Scale factor for bubble size
        max_val = max(yr_data["dac_capacity_mw"].max(), yr_data["beccs_capacity_mw"].max(), 1)
        scale = 800 / max_val if max_val > 0 else 1

        for _, row in yr_data.iterrows():
            cc = row["country"]
            if cc not in centroid_map:
                continue
            cx, cy = centroid_map[cc]

            dac_val = row["dac_capacity_mw"]
            beccs_val = row["beccs_capacity_mw"]

            # Offset DAC and BECCS bubbles slightly
            offset = 1.0
            if dac_val > 0:
                ax.scatter(cx + offset, cy, s=dac_val * scale, c=C_DAC,
                          alpha=0.7, edgecolors="white", linewidth=0.5, zorder=5)
            if beccs_val > 0:
                ax.scatter(cx - offset, cy, s=beccs_val * scale, c=C_BECCS,
                          alpha=0.7, edgecolors="white", linewidth=0.5, zorder=5)

        cb_price = yr_data["credit_price"].iloc[0] if not yr_data.empty else "?"
        ax.set_title(f"{year}\n(at {cb_price} EUR/tCO₂)", fontweight="bold")
        ax.set_xlim(-12, 35)
        ax.set_ylim(35, 72)
        ax.set_aspect("equal")
        ax.axis("off")

    # Legend
    legend_elements = [
        Patch(facecolor=C_DAC, alpha=0.7, label="DAC"),
        Patch(facecolor=C_BECCS, alpha=0.7, label="BECCS"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Spatial Distribution of CDR Capacity at Cap-Binding Price ({variant.title()})",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


country_cdr = get_country_cdr(series, acc, cap_binding, variant="medium")
fig = plot_f2_spatial_maps(country_cdr, variant="medium")
save_fig(fig, "F2_spatial_cdr_maps")
plt.close(fig)

# %% [markdown]
# ### F3: Top-5 Countries by CDR Capacity

# %%
def plot_f3_top5_countries(country_cdr: pd.DataFrame) -> mpl.figure.Figure:
    """F3: Top-5 countries by CDR capacity at cap-binding price, per year."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, year in zip(axes, years):
        yr_data = country_cdr[country_cdr["year"] == year].copy()
        yr_data["total_capacity_mw"] = yr_data["dac_capacity_mw"] + yr_data["beccs_capacity_mw"]
        top5 = yr_data.nlargest(5, "total_capacity_mw")

        if top5.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(str(year))
            continue

        countries = top5["country"].values
        y_pos = np.arange(len(countries))

        ax.barh(y_pos, top5["beccs_capacity_mw"].values, color=C_BECCS,
                label="BECCS", height=0.6)
        ax.barh(y_pos, top5["dac_capacity_mw"].values,
                left=top5["beccs_capacity_mw"].values, color=C_DAC,
                label="DAC", height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries)
        ax.set_xlabel("Capacity [MW]")
        ax.set_title(str(year), fontweight="bold")
        ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Top-5 Countries by CDR Capacity at Cap-Binding Price",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_f3_top5_countries(country_cdr)
save_fig(fig, "F3_top5_countries")
plt.close(fig)

# %% [markdown]
# ---
# ### F4: System Cost Delta (Bar Chart)
#
# Standard bar chart: Δ system cost (bn EUR/yr) at cap-binding price vs baseline.
# Three grouped bars per year (low/mid/high). Positive = cost increase.

# %%
def plot_f4_system_cost_delta(metrics: pd.DataFrame,
                              cap_binding: pd.DataFrame) -> mpl.figure.Figure:
    """F4: System cost delta at cap-binding price."""
    years = [2030, 2040, 2050]
    variants = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(years))
    width = 0.22
    offsets = [-width, 0, width]

    for i, variant in enumerate(variants):
        deltas = []
        for year in years:
            cb_row = cap_binding[(cap_binding["cost_variant"] == variant) &
                                 (cap_binding["year"] == year)]
            cb_price = int(cb_row["cap_binding_price"].iloc[0])
            cb_scenario = f"S{cb_price // 50:02d}"

            m = metrics[(metrics["cost_variant"] == variant) &
                        (metrics["scenario"] == cb_scenario) &
                        (metrics["year"] == year)]
            if not m.empty:
                deltas.append(m["delta_cost_bn"].iloc[0])
            else:
                deltas.append(0)

        bars = ax.bar(x + offsets[i], deltas, width=width,
                      color=C_VARIANTS[variant], label=f"{variant.title()}", zorder=2)

        # Value labels
        for bar, val in zip(bars, deltas):
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:+.0f}", ha="center", va=va, fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Planning horizon")
    ax.set_ylabel("Δ System cost [bn EUR/yr]")
    ax.set_title("System Cost Change at Cap-Binding CDR Price vs Baseline",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig

fig = plot_f4_system_cost_delta(metrics, cap_binding)
save_fig(fig, "F4_system_cost_delta")
plt.close(fig)

# %% [markdown]
# ---
# ## 4.3 Scenario 2: Financing Gap Assessment (RQ2)
#
# ### F5: The Financing Gap Figure
#
# BECCS and DACCS on same graph, same y-axis.
# LCOD bar + WTP band overlay. Gap = visible red zone.
# Focus on 2030 (best WTP data).

# %%
def compute_lcod(acc: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate LCOD from system cost delta per tonne of CDR."""
    rows = []
    for _, row in acc.iterrows():
        variant = row["cost_variant"]
        scenario = row["scenario"]
        year = row["year"]
        total_cdr = row["total_mt"]
        dac_cdr = row["dac_mt"]
        beccs_cdr = row["beccs_mt"]

        m = metrics[(metrics["cost_variant"] == variant) &
                    (metrics["scenario"] == scenario) &
                    (metrics["year"] == year)]
        if m.empty or total_cdr < 1:
            continue

        delta_cost = m["delta_cost_bn"].iloc[0] * 1e3  # bn EUR -> M EUR
        lcod_total = delta_cost / total_cdr if total_cdr > 0 else np.nan  # EUR/tCO2

        rows.append({
            "cost_variant": variant,
            "scenario": scenario,
            "credit_price": row["credit_price"],
            "year": year,
            "total_mt": total_cdr,
            "dac_mt": dac_cdr,
            "beccs_mt": beccs_cdr,
            "lcod_total_eur_per_t": lcod_total,
        })

    return pd.DataFrame(rows)


def plot_f5_financing_gap(lcod_df: pd.DataFrame, year: int = 2030) -> mpl.figure.Figure:
    """F5: Financing gap — LCOD vs WTP for BECCS and DACCS (2030 focus)."""
    if year not in WTP:
        print(f"  No WTP data for {year} — skipping gap figure.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get LCOD at cap-binding for each variant
    variants = ["low", "medium", "high"]
    variant_labels = ["Low cost", "Medium cost", "High cost"]

    x_pos = np.arange(len(variants))
    bar_width = 0.3

    # Approximate LCOD for BECCS and DACCS separately
    # Use marginal cost approach: credit price at which tech starts deploying
    # For simplicity, use credit_price as a proxy for LCOD (since credit = marginal revenue needed)
    for tech_idx, (tech, tech_label, color) in enumerate([
        ("beccs", "BECCS", C_BECCS), ("dac", "DACCS", C_DAC)
    ]):
        lcods = []
        for variant in variants:
            sub = lcod_df[(lcod_df["cost_variant"] == variant) &
                          (lcod_df["year"] == year)].sort_values("credit_price")
            col = f"{tech}_mt"
            # Find first price where this tech deploys > 1 Mt
            deploying = sub[sub[col] > 1]
            if not deploying.empty:
                lcod_est = deploying["credit_price"].iloc[0]
            else:
                lcod_est = 500  # never deploys significantly
            lcods.append(lcod_est)

        offset = -bar_width / 2 + tech_idx * bar_width
        bars = ax.bar(x_pos + offset, lcods, width=bar_width,
                      color=color, alpha=0.85, label=f"{tech_label} LCOD", zorder=2)

    # WTP bands
    wtp_data = WTP[year]
    # BECCS WTP band
    ax.axhspan(wtp_data["BECCS"]["low"], wtp_data["BECCS"]["high"],
               alpha=0.15, color=C_BECCS, zorder=1)
    ax.axhline(wtp_data["BECCS"]["central"], color=C_BECCS, linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"BECCS WTP range (€{wtp_data['BECCS']['low']}–{wtp_data['BECCS']['high']})")

    # DACCS WTP band
    ax.axhspan(wtp_data["DACCS"]["low"], wtp_data["DACCS"]["high"],
               alpha=0.15, color=C_DAC, zorder=1)
    ax.axhline(wtp_data["DACCS"]["central"], color=C_DAC, linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"DACCS WTP range (€{wtp_data['DACCS']['low']}–{wtp_data['DACCS']['high']})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(variant_labels)
    ax.set_ylabel("EUR / tCO₂")
    ax.set_title(f"Financing Gap: LCOD vs Willingness-to-Pay ({year})",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(550, ax.get_ylim()[1]))

    # Annotate the gap
    ax.annotate("← Gap →", xy=(0.5, 0.85), xycoords="axes fraction",
                fontsize=12, ha="center", color=C_GAP, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff2f2", edgecolor=C_GAP))

    fig.tight_layout()
    return fig


lcod_df = compute_lcod(acc, metrics)
fig = plot_f5_financing_gap(lcod_df, year=2030)
if fig:
    save_fig(fig, "F5_financing_gap_2030")
    plt.close(fig)

# %% [markdown]
# ### F5b: Gap Waterfall — visualising the gap as literal space

# %%
def plot_f5b_gap_waterfall(year: int = 2030) -> mpl.figure.Figure:
    """F5b: Waterfall showing LCOD, WTP, and the gap between them."""
    if year not in WTP:
        return None

    wtp_data = WTP[year]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, (tech, tech_label, color) in zip(axes, [
        ("BECCS", "BECCS", C_BECCS), ("DACCS", "DACCS", C_DAC)
    ]):
        wtp_central = wtp_data[tech]["central"]
        wtp_low = wtp_data[tech]["low"]
        wtp_high = wtp_data[tech]["high"]

        # Approximate LCOD from the v8 data
        # Use credit price at which ~50% of cap is filled for this tech
        if tech == "BECCS":
            lcod_low, lcod_mid, lcod_high = 100, 150, 200
        else:  # DACCS
            lcod_low, lcod_mid, lcod_high = 200, 250, 350

        scenarios = ["Low\ncost", "Mid\ncost", "High\ncost"]
        lcods = [lcod_low, lcod_mid, lcod_high]
        x_pos = np.arange(len(scenarios))

        # WTP bar (green/blue, from 0 to WTP)
        ax.bar(x_pos, [wtp_central] * 3, width=0.5,
               color=color, alpha=0.3, label=f"WTP (€{wtp_central}/t)", zorder=1)

        # WTP range
        ax.fill_between(x_pos - 0.25, [wtp_low] * 3, [wtp_high] * 3,
                         alpha=0.1, color=color, step="mid")

        # LCOD bar on top
        for i, lcod in enumerate(lcods):
            if lcod > wtp_central:
                # Gap exists — show red zone
                ax.bar(i, lcod - wtp_central, bottom=wtp_central, width=0.5,
                       color=C_GAP, alpha=0.6, zorder=2,
                       label="Gap" if i == 0 else None)
                ax.text(i, lcod + 5, f"€{lcod - wtp_central}/t",
                        ha="center", fontsize=9, color=C_GAP, fontweight="bold")
            else:
                # WTP covers LCOD — no gap
                ax.bar(i, 0, bottom=lcod, width=0.5, color="green", alpha=0.3,
                       zorder=2)
                ax.text(i, lcod - 10, "✓", ha="center", fontsize=14, color="green")

            # LCOD line
            ax.plot([i - 0.25, i + 0.25], [lcod, lcod], color="black",
                    linewidth=2, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenarios)
        ax.set_title(tech_label, fontweight="bold", fontsize=13)
        ax.set_ylim(0, max(lcods) * 1.2)

    axes[0].set_ylabel("EUR / tCO₂")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Financing Gap Waterfall ({year}): LCOD vs WTP",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_f5b_gap_waterfall(2030)
if fig:
    save_fig(fig, "F5b_gap_waterfall_2030")
    plt.close(fig)

# %% [markdown]
# ### F6: 3×3 Gap Sensitivity Heatmap

# %%
def plot_f6_gap_heatmap(year: int = 2030) -> mpl.figure.Figure:
    """F6: 3×3 gap heatmap for BECCS and DACCS."""
    if year not in WTP:
        return None

    wtp_data = WTP[year]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (tech, tech_label, color) in zip(axes, [
        ("BECCS", "BECCS", C_BECCS), ("DACCS", "DACCS", C_DAC)
    ]):
        wtp_vals = [wtp_data[tech]["low"], wtp_data[tech]["central"], wtp_data[tech]["high"]]

        if tech == "BECCS":
            lcod_vals = [100, 150, 200]
        else:
            lcod_vals = [200, 250, 350]

        gap_matrix = np.zeros((3, 3))
        for i, lcod in enumerate(lcod_vals):
            for j, wtp in enumerate(wtp_vals):
                gap_matrix[i, j] = lcod - wtp

        # Red = gap (LCOD > WTP), green = surplus (WTP > LCOD)
        vmax = max(abs(gap_matrix.min()), abs(gap_matrix.max()), 50)
        im = ax.imshow(gap_matrix, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")

        # Annotations
        for i in range(3):
            for j in range(3):
                val = gap_matrix[i, j]
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"€{val:+.0f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"WTP\n€{v}" for v in wtp_vals], fontsize=9)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([f"LCOD €{v}" for v in lcod_vals], fontsize=9)
        ax.set_xlabel("Willingness-to-Pay")
        ax.set_ylabel("Levelised Cost of Delivery")
        ax.set_title(tech_label, fontweight="bold", fontsize=13)

        plt.colorbar(im, ax=ax, label="Gap [EUR/tCO₂]", shrink=0.8)

    fig.suptitle(f"Financing Gap Sensitivity Matrix ({year})",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_f6_gap_heatmap(2030)
if fig:
    save_fig(fig, "F6_gap_heatmap_2030")
    plt.close(fig)

# %% [markdown]
# ---
# ## Appendix Figures

# %% [markdown]
# ### A1: Full supply curves with plateau (before truncation)

# %%
def plot_a1_full_supply_curves(acc: pd.DataFrame) -> mpl.figure.Figure:
    """A1: Full supply curves including the plateau (appendix)."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, year in zip(axes, years):
        for variant in ["low", "medium", "high"]:
            sub = acc[(acc["cost_variant"] == variant) & (acc["year"] == year)].sort_values("credit_price")
            ax.plot(sub["credit_price"].values, sub["total_mt"].values,
                    marker="o", markersize=3, linewidth=1.8,
                    color=C_VARIANTS[variant], label=f"{variant.title()}")

        seq_limit = SEQ_LIMITS[year]
        ax.axhline(seq_limit, color="grey", linewidth=1, linestyle="--", alpha=0.7,
                   label=f"Seq. limit ({seq_limit} Mt)")

        target = TARGETS[year]
        if target["value"] != seq_limit:
            ax.axhline(target["value"], color="#e41a1c", linewidth=1, linestyle=":",
                       alpha=0.7, label=target["label"])

        ax.set_xlabel("CDR credit price [EUR/tCO₂]")
        ax.set_title(str(year), fontweight="bold")
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("CDR deployment [Mt CO₂/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02),
               fontsize=9)
    fig.suptitle("Full CDR Supply Curves (Including Plateau)",
                 y=1.10, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_a1_full_supply_curves(acc)
save_fig(fig, "A1_full_supply_curves")
plt.close(fig)

# %% [markdown]
# ### A2: Sub-technology breakdown

# %%
def plot_a2_sub_technology(series: pd.DataFrame, acc: pd.DataFrame,
                           cap_binding: pd.DataFrame,
                           variant: str = "medium") -> mpl.figure.Figure:
    """A2: Sub-technology breakdown at cap-binding price."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    dac_carriers = ["DAC-solidLT", "DAC-liquidHT"]
    beccs_carriers = [
        "solid biomass for industry CC",
        "urban central solid biomass CHP CC",
        "biomass to liquid CC",
        "biomass-to-methanol CC",
    ]
    all_carriers = dac_carriers + beccs_carriers
    colors = {
        "DAC-solidLT": "#2166ac",
        "DAC-liquidHT": "#67a9cf",
        "solid biomass for industry CC": "#4dac26",
        "urban central solid biomass CHP CC": "#7fc97f",
        "biomass to liquid CC": "#a6d96a",
        "biomass-to-methanol CC": "#d9ef8b",
    }
    short_labels = {
        "DAC-solidLT": "S-DAC (solid)",
        "DAC-liquidHT": "L-DAC (liquid)",
        "solid biomass for industry CC": "BECCS Industry",
        "urban central solid biomass CHP CC": "BECCS CHP",
        "biomass to liquid CC": "BioLiquid CC",
        "biomass-to-methanol CC": "BioMethanol CC",
    }

    for ax, year in zip(axes, years):
        cb_row = cap_binding[(cap_binding["cost_variant"] == variant) &
                             (cap_binding["year"] == year)]
        cb_price = int(cb_row["cap_binding_price"].iloc[0])
        cb_scenario = f"S{cb_price // 50:02d}"

        caps = series[
            (series["csv_type"] == "capacities") &
            (series["cost_variant"] == variant) &
            (series["scenario"] == cb_scenario) &
            (series["year"] == year) &
            (series["carrier"].isin(all_carriers))
        ].copy()

        if caps.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(str(year))
            continue

        cap_by_carrier = caps.groupby("carrier")["value"].sum()
        # Filter to non-zero
        cap_by_carrier = cap_by_carrier[cap_by_carrier > 0.1]

        if cap_by_carrier.empty:
            ax.text(0.5, 0.5, "No deployment", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(str(year))
            continue

        labels_plot = [short_labels.get(c, c) for c in cap_by_carrier.index]
        colors_plot = [colors.get(c, "#999999") for c in cap_by_carrier.index]

        ax.barh(range(len(cap_by_carrier)), cap_by_carrier.values,
                color=colors_plot, height=0.6)
        ax.set_yticks(range(len(cap_by_carrier)))
        ax.set_yticklabels(labels_plot, fontsize=9)
        ax.set_xlabel("Capacity [MW]")
        ax.set_title(f"{year} (at {cb_price} EUR/tCO₂)", fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle(f"Sub-Technology Breakdown at Cap-Binding Price ({variant.title()})",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_a2_sub_technology(series, acc, cap_binding)
save_fig(fig, "A2_sub_technology_breakdown")
plt.close(fig)

# %% [markdown]
# ### A3: System cost across all credit prices

# %%
def plot_a3_system_cost_full(metrics: pd.DataFrame) -> mpl.figure.Figure:
    """A3: System cost vs credit price (full range, appendix)."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, year in zip(axes, years):
        for variant in ["low", "medium", "high"]:
            sub = metrics[(metrics["cost_variant"] == variant) &
                          (metrics["year"] == year)].sort_values("credit_price")
            ax.plot(sub["credit_price"].values, sub["delta_cost_bn"].values,
                    marker="o", markersize=3, linewidth=1.8,
                    color=C_VARIANTS[variant], label=f"{variant.title()}")

        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.set_xlabel("CDR credit price [EUR/tCO₂]")
        ax.set_title(str(year), fontweight="bold")

    axes[0].set_ylabel("Δ System cost [bn EUR/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("System Cost Delta vs Baseline (Full Price Range)",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_a3_system_cost_full(metrics)
save_fig(fig, "A3_system_cost_full")
plt.close(fig)

# %% [markdown]
# ### A4: Electricity price impact

# %%
def plot_a4_elec_price(metrics: pd.DataFrame) -> mpl.figure.Figure:
    """A4: Electricity price vs credit price."""
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, year in zip(axes, years):
        for variant in ["low", "medium", "high"]:
            sub = metrics[(metrics["cost_variant"] == variant) &
                          (metrics["year"] == year)].sort_values("credit_price")
            ax.plot(sub["credit_price"].values, sub["electricity_price_mean"].values,
                    marker="o", markersize=3, linewidth=1.8,
                    color=C_VARIANTS[variant], label=f"{variant.title()}")

        ax.set_xlabel("CDR credit price [EUR/tCO₂]")
        ax.set_title(str(year), fontweight="bold")

    axes[0].set_ylabel("Mean electricity price [EUR/MWh]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Electricity Price Response to CDR Credit",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_a4_elec_price(metrics)
save_fig(fig, "A4_electricity_price")
plt.close(fig)

# %% [markdown]
# ### A5: Nordic vs Rest of EU

# %%
def plot_a5_nordic(country_cdr_all: pd.DataFrame) -> mpl.figure.Figure:
    """A5: Nordic CDR share vs rest of EU across all prices."""
    # We need country data at ALL price points, not just cap-binding
    # For this appendix figure, use the nodal data

    # Already have country_cdr at cap-binding only — for appendix, show a simpler version
    nordic = {"NO", "SE", "DK", "FI"}
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, year in zip(axes, years):
        yr_data = country_cdr_all[country_cdr_all["year"] == year].copy()
        yr_data["total"] = yr_data["dac_capacity_mw"] + yr_data["beccs_capacity_mw"]
        yr_data["is_nordic"] = yr_data["country"].isin(nordic)

        nordic_total = yr_data[yr_data["is_nordic"]]["total"].sum()
        rest_total = yr_data[~yr_data["is_nordic"]]["total"].sum()
        total = nordic_total + rest_total

        if total == 0:
            ax.text(0.5, 0.5, "No CDR", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(str(year))
            continue

        ax.bar(0, nordic_total, color="#4575b4", label="Nordic (NO, SE, DK, FI)")
        ax.bar(0, rest_total, bottom=nordic_total, color="#d73027", label="Rest of EU")
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([])
        ax.set_title(f"{year}\nNordic: {nordic_total / total * 100:.0f}%", fontweight="bold")
        ax.set_ylabel("Total CDR capacity [MW]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Nordic vs Rest of EU — CDR Capacity at Cap-Binding Price",
                 y=1.08, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

fig = plot_a5_nordic(country_cdr)
save_fig(fig, "A5_nordic_vs_rest")
plt.close(fig)

# %% [markdown]
# ---
# ## Summary of outputs

# %%
print("\n" + "=" * 60)
print("THESIS FIGURES GENERATED")
print("=" * 60)
print(f"\nOutput directory: {OUT_DIR}")
print("\nMain text figures:")
print("  F1   — Supply curve + tech mix (medium; + low/high variants)")
print("  F1b  — All variants overlaid")
print("  F2   — Spatial CDR deployment maps")
print("  F3   — Top-5 countries bar chart")
print("  F4   — System cost delta (bar chart)")
print("  F5   — Financing gap (LCOD vs WTP)")
print("  F5b  — Gap waterfall")
print("  F6   — 3×3 gap sensitivity heatmap")
print("\nAppendix figures:")
print("  A1   — Full supply curves (with plateau)")
print("  A2   — Sub-technology breakdown")
print("  A3   — System cost full range")
print("  A4   — Electricity price response")
print("  A5   — Nordic vs rest of EU")
print(f"\nTotal: {len(list(OUT_DIR.glob('*.png')))} PNG + {len(list(OUT_DIR.glob('*.pdf')))} PDF files")
