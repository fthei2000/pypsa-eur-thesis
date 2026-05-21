from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

OUT_DIR = Path("/work3/s240459/pypsa-eur-thesis/notebooks")
PNG = OUT_DIR / "cdr_market_gap_storyline.png"
PDF = OUT_DIR / "cdr_market_gap_storyline.pdf"

BLUE = "#67A9F7"
BLUE_DARK = "#2F5C8A"
YELLOW = "#FFD84D"
ORANGE = "#FF9D3D"
PURPLE = "#B8AEFF"
TEXT = "#1E1E1E"
ARROW = "#444444"
BG = "white"


def rounded_box(ax, x, y, w, h, fc, ec="none", lw=1.5, radius=0.03):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    return patch


def hex_box(ax, x, y, w, h, fc, ec="none", lw=1.5):
    dx = 0.12 * w
    pts = [
        (x + dx, y),
        (x + w - dx, y),
        (x + w, y + h / 2),
        (x + w - dx, y + h),
        (x + dx, y + h),
        (x, y + h / 2),
    ]
    patch = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, s, size=14, weight="normal", style="normal", ha="center", va="center", color=TEXT):
    ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va=va, color=color)


def add_arrow(ax, p1, p2, connectionstyle="arc3", lw=1.8, mutation_scale=12):
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=ARROW,
        connectionstyle=connectionstyle,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)
    return arrow


def box_with_heading(ax, x, y, w, h, heading, lines):
    rounded_box(ax, x, y, w, h, fc=BLUE)
    add_text(ax, x + w / 2, y + h - 0.06, heading, size=16, weight="bold")
    add_text(ax, x + w / 2, y + h / 2 - 0.03, lines, size=12, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(17, 8), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headers
    add_text(ax, 0.18, 0.93, "Supply-Side and Demand-Side\nInputs", size=19, weight="bold")
    add_text(ax, 0.43, 0.93, "Integrated Energy System\nModel", size=19, weight="bold")
    add_text(ax, 0.61, 0.93, "Policy and Scenario\nDesign", size=19, weight="bold")
    add_text(ax, 0.79, 0.93, "Model Outputs", size=19, weight="bold")
    add_text(ax, 0.93, 0.93, "Main Interpretation", size=19, weight="bold")

    # Left input cluster
    left_x, left_y, left_w, left_h = 0.06, 0.18, 0.24, 0.58
    cluster = FancyBboxPatch((left_x, left_y), left_w, left_h, boxstyle="round,pad=0.015,rounding_size=0.01", facecolor="none", edgecolor="#9A9A9A", linewidth=1.5)
    ax.add_patch(cluster)

    box_with_heading(
        ax, 0.08, 0.53, 0.10, 0.20,
        "System\nConstraints",
        "DAC costs\nElectricity use\nHeat demand\nStorage costs\nRegional CO$_2$ storage"
    )
    box_with_heading(
        ax, 0.20, 0.53, 0.10, 0.20,
        "Biomass and\nInfrastructure",
        "Biomass availability\nCO$_2$ network\nPower-system interaction\nTransmission and heat"
    )
    box_with_heading(
        ax, 0.08, 0.25, 0.10, 0.20,
        "External\nDemand Evidence",
        "Voluntary CDR market\nProcurement studies\nCompliance analogues\nPolicy targets"
    )
    box_with_heading(
        ax, 0.20, 0.25, 0.10, 0.20,
        "Willingness\nto Pay",
        "Observed price points\nLiterature ranges\nTarget demand bands\nUncertainty in WTP"
    )

    add_text(ax, 0.025, 0.47, "Specific to\nsupply or\ndemand", size=13)
    add_arrow(ax, (0.05, 0.54), (0.08, 0.63), connectionstyle="angle,angleA=180,angleB=90,rad=10")
    add_arrow(ax, (0.05, 0.40), (0.08, 0.35), connectionstyle="angle,angleA=180,angleB=-90,rad=10")

    # Model core
    rounded_box(ax, 0.38, 0.42, 0.10, 0.12, fc=YELLOW, ec="none", radius=0.01)
    add_text(ax, 0.43, 0.48, "PyPSA-Eur\nThesis", size=22, weight="bold")

    add_arrow(ax, (0.30, 0.48), (0.38, 0.48))

    # Policy scenarios
    hex_box(ax, 0.54, 0.54, 0.12, 0.16, fc=ORANGE)
    add_text(ax, 0.60, 0.62, "Standalone\nCDR Market", size=17, weight="bold")
    add_text(ax, 0.60, 0.545, "DACCS + BECCS\ncredited at sequestration", size=11)

    hex_box(ax, 0.54, 0.28, 0.12, 0.16, fc=ORANGE)
    add_text(ax, 0.60, 0.36, "Supply Curve\nScenarios", size=17, weight="bold")
    add_text(ax, 0.60, 0.285, "S0-S10\n0-750 €/tCO$_2$\n2030 / 2040 / 2050", size=11)

    add_arrow(ax, (0.48, 0.48), (0.54, 0.62))
    add_arrow(ax, (0.48, 0.48), (0.54, 0.36))

    # Outputs
    rounded_box(ax, 0.73, 0.58, 0.10, 0.08, fc=PURPLE, ec="none", radius=0.04)
    add_text(ax, 0.78, 0.62, "Supplied\ncredited CDR", size=16)

    rounded_box(ax, 0.73, 0.45, 0.10, 0.08, fc=PURPLE, ec="none", radius=0.04)
    add_text(ax, 0.78, 0.49, "DACCS vs\nBECCS mix", size=16)

    rounded_box(ax, 0.73, 0.32, 0.10, 0.08, fc=PURPLE, ec="none", radius=0.04)
    add_text(ax, 0.78, 0.36, "System cost\nand storage signal", size=16)

    add_arrow(ax, (0.66, 0.62), (0.73, 0.62))
    add_arrow(ax, (0.66, 0.36), (0.73, 0.49), connectionstyle="angle3")
    add_arrow(ax, (0.66, 0.36), (0.73, 0.36), connectionstyle="angle3")
    ax.plot([0.69, 0.69], [0.36, 0.62], color=ARROW, lw=1.8)

    add_text(ax, 0.69, 0.26, "One scenario family\nper year and credit level", size=12)

    # Final interpretation block
    rounded_box(ax, 0.88, 0.35, 0.10, 0.24, fc=BLUE, ec="none", radius=0.01)
    add_text(ax, 0.93, 0.53, "Gap Between\nSystem Supply\nand Market WTP", size=18, weight="bold")
    add_text(ax, 0.93, 0.41,
             "Price gap at\ntarget volume\n\nVolume gap at\nobserved WTP\n\nPolicy relevance",
             size=13)

    add_arrow(ax, (0.83, 0.49), (0.88, 0.49))

    # Footer note
    add_text(ax, 0.62, 0.12,
             "Current model logic: fossil CCS keeps CO$_2$-price incentive, while DACCS and BECCS receive standalone CDR credits only.",
             size=12)

    plt.tight_layout()
    fig.savefig(PNG, bbox_inches="tight", facecolor=BG)
    fig.savefig(PDF, bbox_inches="tight", facecolor=BG)
    print(f"Saved {PNG}")
    print(f"Saved {PDF}")


if __name__ == "__main__":
    main()
