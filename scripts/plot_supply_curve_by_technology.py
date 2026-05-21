from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from plot_supply_curve_cdr_split import OUT_PDF as _UNUSED  # ensures same env deps
from plot_supply_curve_cdr_split import collect

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "notebooks" / "eval_supply_curve_by_technology.pdf"
OUT_PNG = ROOT / "notebooks" / "eval_supply_curve_by_technology.png"


def plot(df):
    years = sorted(df["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(5.4 * len(years), 5.0), sharey=True)
    if len(years) == 1:
        axes = [axes]

    colors = {"BECCS": "#2ca02c", "DAC": "#1f77b4", "Total": "#222222"}

    for ax, year in zip(axes, years):
        sub = df[df["year"] == year].sort_values("price_eur_per_t")
        x = sub["price_eur_per_t"].to_numpy()
        beccs = sub["BECCS_MtCO2_per_yr"].to_numpy()
        dac = sub["DAC_MtCO2_per_yr"].to_numpy()
        total = beccs + dac

        ax.plot(x, total, color=colors["Total"], marker="o", linewidth=2.2, label="Total")
        ax.plot(x, beccs, color=colors["BECCS"], marker="o", linewidth=2.0, label="BECCS")
        ax.plot(x, dac, color=colors["DAC"], marker="o", linewidth=2.0, label="DAC")
        ax.fill_between(x, 0, beccs, color=colors["BECCS"], alpha=0.08)
        ax.fill_between(x, 0, dac, color=colors["DAC"], alpha=0.06)
        ax.set_title(str(year))
        ax.set_xlabel("CDR credit price [EUR/tCO2]")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Deployed CDR [MtCO2/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Supply Curves by CDR Technology", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")


def main():
    df = collect()
    plot(df)
    print(df[["price_eur_per_t", "year", "DAC_MtCO2_per_yr", "BECCS_MtCO2_per_yr", "method"]].to_csv(index=False))
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
