#!/usr/bin/env python3
"""Build country-level DAC weather productivity factors from ERA5 temperature/dewpoint.

Relative humidity is calculated from ERA5 2m temperature and 2m dewpoint with
the Sonntag saturation-vapour-pressure formula used by the DAC weather paper.

Uses digitised performance maps from Figure S1 (adapted from literature):
  - Row A: L-DAC  — productivity increases with temperature and relative humidity
  - Row B: S-DAC  — productivity increases with *decreasing* temperature and humidity

Three separate scaling factors are computed (all relative to a configurable
reference T/RH that should match the conditions assumed by the cost table):

  ldac_elec_factor   : L-DAC electricity demand scaling  (Row A, left panel)
  sdac_elec_factor   : S-DAC electricity demand scaling  (Row B, left panel)
  sdac_heat_factor   : S-DAC thermal demand scaling      (Row B, middle panel)

Values > 1 mean the country needs *more* energy per tCO2 than the reference.
These are used in prepare_sector_network.py to scale per-node link efficiencies.

The script also retains overall productivity columns (ldac_productivity,
sdac_productivity) from the combined right-panel of each row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Lookup tables digitised from Figure S1
# Rows index temperature (°C), columns index relative humidity (%)
# ---------------------------------------------------------------------------

# --- L-DAC overall relative productivity (Row A, right panel) ---------------
# T: 0–40 °C,  RH: 0–90 %
_LDAC_T = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=float)
_LDAC_RH = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
_LDAC_PROD = np.array([
    # RH: 0     10    20    30    40    50    60    70    80    90
    [0.720, 0.720, 0.720, 0.720, 0.725, 0.730, 0.740, 0.750, 0.760, 0.770],  # T=0
    [0.730, 0.735, 0.745, 0.758, 0.770, 0.783, 0.800, 0.815, 0.830, 0.845],  # T=5
    [0.750, 0.762, 0.778, 0.796, 0.815, 0.835, 0.855, 0.872, 0.892, 0.910],  # T=10
    [0.778, 0.795, 0.815, 0.838, 0.860, 0.882, 0.905, 0.928, 0.948, 0.968],  # T=15
    [0.808, 0.830, 0.855, 0.880, 0.906, 0.932, 0.958, 0.980, 1.005, 1.025],  # T=20
    [0.840, 0.868, 0.895, 0.924, 0.952, 0.982, 1.010, 1.038, 1.065, 1.088],  # T=25
    [0.878, 0.908, 0.938, 0.970, 1.000, 1.030, 1.062, 1.092, 1.122, 1.150],  # T=30
    [0.918, 0.950, 0.982, 1.015, 1.048, 1.080, 1.112, 1.145, 1.178, 1.210],  # T=35
    [0.958, 0.993, 1.028, 1.062, 1.096, 1.130, 1.163, 1.196, 1.200, 1.200],  # T=40
], dtype=float)

# --- S-DAC overall relative productivity (Row B, right panel) ---------------
# T: 5–50 °C,  RH: 0–100 %
_SDAC_T = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
_SDAC_RH = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_SDAC_PROD = np.array([
    # RH: 0     10    20    30    40    50    60    70    80    90   100
    [1.100, 1.080, 1.040, 1.000, 0.960, 0.920, 0.880, 0.840, 0.800, 0.760, 0.720],  # T=5
    [1.040, 1.018, 0.978, 0.940, 0.900, 0.862, 0.820, 0.780, 0.740, 0.700, 0.660],  # T=10
    [0.970, 0.948, 0.910, 0.872, 0.832, 0.795, 0.758, 0.720, 0.682, 0.648, 0.618],  # T=15
    [0.908, 0.886, 0.848, 0.810, 0.772, 0.736, 0.700, 0.668, 0.638, 0.610, 0.585],  # T=20
    [0.850, 0.828, 0.790, 0.752, 0.716, 0.682, 0.650, 0.622, 0.598, 0.575, 0.558],  # T=25
    [0.796, 0.774, 0.738, 0.702, 0.668, 0.636, 0.610, 0.585, 0.565, 0.548, 0.535],  # T=30
    [0.746, 0.725, 0.692, 0.658, 0.628, 0.602, 0.578, 0.558, 0.542, 0.528, 0.518],  # T=35
    [0.705, 0.685, 0.655, 0.625, 0.600, 0.576, 0.556, 0.540, 0.526, 0.514, 0.506],  # T=40
    [0.672, 0.652, 0.624, 0.598, 0.576, 0.556, 0.540, 0.526, 0.514, 0.504, 0.496],  # T=45
    [0.645, 0.626, 0.600, 0.577, 0.558, 0.542, 0.528, 0.516, 0.506, 0.498, 0.490],  # T=50
], dtype=float)

# --- L-DAC specific electrical requirement (Row A, left panel) --------------
# MWh_el / tCO2.  T: 0–40 °C,  RH: 0–90 %
# Mostly a temperature effect; humidity has a minor role.
_LDAC_ELEC_REQ = np.array([
    # RH: 0     10    20    30    40    50    60    70    80    90
    [2.200, 2.200, 2.190, 2.180, 2.170, 2.165, 2.155, 2.145, 2.135, 2.125],  # T=0
    [2.170, 2.160, 2.150, 2.140, 2.130, 2.120, 2.110, 2.100, 2.090, 2.080],  # T=5
    [2.130, 2.120, 2.110, 2.100, 2.090, 2.080, 2.070, 2.060, 2.050, 2.040],  # T=10
    [2.090, 2.080, 2.070, 2.060, 2.050, 2.040, 2.030, 2.020, 2.010, 2.000],  # T=15
    [2.050, 2.040, 2.030, 2.020, 2.010, 2.000, 1.990, 1.980, 1.970, 1.960],  # T=20
    [2.010, 2.000, 1.990, 1.980, 1.970, 1.960, 1.950, 1.940, 1.930, 1.920],  # T=25
    [1.970, 1.960, 1.950, 1.940, 1.930, 1.920, 1.910, 1.900, 1.890, 1.880],  # T=30
    [1.940, 1.930, 1.920, 1.910, 1.900, 1.890, 1.880, 1.880, 1.880, 1.880],  # T=35
    [1.900, 1.890, 1.880, 1.880, 1.880, 1.880, 1.880, 1.880, 1.880, 1.880],  # T=40
], dtype=float)

# --- S-DAC specific electrical requirement (Row B, left panel) --------------
# MWh_el / tCO2.  T: 5–50 °C,  RH: 0–100 %
# Increases with temperature; nearly flat with humidity.
_SDAC_ELEC_T = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
_SDAC_ELEC_RH = np.array([0, 20, 40, 60, 80, 100], dtype=float)
_SDAC_ELEC_REQ = np.array([
    # RH: 0     20    40    60    80   100
    [0.300, 0.300, 0.300, 0.300, 0.300, 0.300],  # T=5
    [0.300, 0.300, 0.300, 0.300, 0.300, 0.300],  # T=10
    [0.330, 0.330, 0.335, 0.340, 0.350, 0.360],  # T=15
    [0.360, 0.362, 0.370, 0.380, 0.400, 0.420],  # T=20
    [0.400, 0.408, 0.420, 0.438, 0.458, 0.478],  # T=25
    [0.445, 0.456, 0.470, 0.490, 0.516, 0.540],  # T=30
    [0.495, 0.508, 0.526, 0.546, 0.568, 0.596],  # T=35
    [0.546, 0.560, 0.578, 0.600, 0.622, 0.648],  # T=40
    [0.582, 0.600, 0.620, 0.640, 0.660, 0.682],  # T=45
    [0.618, 0.638, 0.658, 0.678, 0.694, 0.710],  # T=50
], dtype=float)

# --- S-DAC specific thermal requirement (Row B, middle panel) ---------------
# MWh_th / tCO2.  T: 5–50 °C,  RH: 0–100 %
# Increases strongly with temperature; also increases with humidity.
_SDAC_HEAT_REQ = np.array([
    # RH: 0     20    40    60    80   100
    [2.000, 2.000, 2.000, 2.000, 2.100, 2.200],  # T=5
    [2.000, 2.000, 2.000, 2.050, 2.150, 2.250],  # T=10
    [2.050, 2.100, 2.200, 2.350, 2.500, 2.650],  # T=15
    [2.200, 2.300, 2.450, 2.600, 2.750, 2.900],  # T=20
    [2.400, 2.540, 2.680, 2.840, 3.000, 3.150],  # T=25
    [2.600, 2.750, 2.900, 3.050, 3.200, 3.350],  # T=30
    [2.800, 2.950, 3.100, 3.250, 3.400, 3.550],  # T=35
    [3.000, 3.150, 3.300, 3.450, 3.600, 3.750],  # T=40
    [3.200, 3.350, 3.500, 3.650, 3.750, 3.750],  # T=45
    [3.400, 3.550, 3.650, 3.750, 3.750, 3.750],  # T=50
], dtype=float)

# --- Build interpolators (linear, extrapolate by clamping to boundary) ------
_ldac_prod_interp = RegularGridInterpolator(
    (_LDAC_T, _LDAC_RH), _LDAC_PROD,
    method="linear", bounds_error=False, fill_value=None,
)
_sdac_prod_interp = RegularGridInterpolator(
    (_SDAC_T, _SDAC_RH), _SDAC_PROD,
    method="linear", bounds_error=False, fill_value=None,
)
_ldac_elec_interp = RegularGridInterpolator(
    (_LDAC_T, _LDAC_RH), _LDAC_ELEC_REQ,
    method="linear", bounds_error=False, fill_value=None,
)
_sdac_elec_interp = RegularGridInterpolator(
    (_SDAC_ELEC_T, _SDAC_ELEC_RH), _SDAC_ELEC_REQ,
    method="linear", bounds_error=False, fill_value=None,
)
_sdac_heat_interp = RegularGridInterpolator(
    (_SDAC_ELEC_T, _SDAC_ELEC_RH), _SDAC_HEAT_REQ,
    method="linear", bounds_error=False, fill_value=None,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

DEFAULT_WEATHER_FILE = Path("data/co2stop/39eb6960b9e615be9a8b70871bf9a450.grib")
TEMPERATURE_CANDIDATES = ("t2m", "2t", "temperature", "temp")
DEWPOINT_CANDIDATES = ("d2m", "2d", "dewpoint", "dew_point_temperature")


def open_weather(path: Path) -> xr.Dataset:
    engine = "cfgrib" if path.suffix.lower() in {".grib", ".grib2", ".grb", ".grb2"} else None
    kwargs = {"engine": engine} if engine else {}
    if engine == "cfgrib":
        kwargs["backend_kwargs"] = {"indexpath": ""}
    return xr.open_dataset(path, **kwargs)


def pick_variable(ds: xr.Dataset, candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        if explicit not in ds.data_vars:
            raise KeyError(f"{explicit!r} not found. Available: {list(ds.data_vars)}")
        return explicit
    lower = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Could not infer variable from {candidates}. Available: {list(ds.data_vars)}")


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"} or float(da.mean().values) > 100.0:
        return da - 273.15
    return da


def sonntag_log_saturation_vapour_pressure(t_k: xr.DataArray) -> xr.DataArray:
    """Return ln(e_w) from the Sonntag formula for temperature in Kelvin."""
    return (
        -6096.9385 / t_k
        - 2.711193e-2 * t_k
        + 1.673952e-5 * t_k**2
        + 2.433502 * np.log(t_k)
        + 21.2409642
    )


def relative_humidity_from_dewpoint(t_c: xr.DataArray, td_c: xr.DataArray) -> xr.DataArray:
    """Calculate RH [%] from air temperature and dewpoint using Sonntag.

    The caller passes Celsius because ERA5 data are normalised by
    ``kelvin_to_celsius`` first; Sonntag itself is evaluated in Kelvin.
    """
    t_k = t_c + 273.15
    td_k = td_c + 273.15
    rh = 100.0 * np.exp(
        sonntag_log_saturation_vapour_pressure(td_k)
        - sonntag_log_saturation_vapour_pressure(t_k)
    )
    return rh.clip(min=0.0, max=100.0)


def mean_over_non_spatial_dims(da: xr.DataArray) -> xr.DataArray:
    spatial = {"latitude", "lat", "longitude", "lon"}
    dims = [d for d in da.dims if d not in spatial]
    if dims:
        da = da.mean(dims)
    return da.squeeze(drop=True)


# ---------------------------------------------------------------------------
# Spatial aggregation
# ---------------------------------------------------------------------------

def country_grid_index(da: xr.DataArray, countries_path: Path) -> pd.DataFrame:
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    lon = da[lon_name].values
    lat = da[lat_name].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    grid = pd.DataFrame({
        "row": np.repeat(np.arange(len(lat)), len(lon)),
        "col": np.tile(np.arange(len(lon)), len(lat)),
        "lon": lon2d.ravel(),
        "lat": lat2d.ravel(),
        "weight": np.cos(np.deg2rad(lat2d.ravel())),
    })
    points = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    )
    countries = gpd.read_file(countries_path)[["name", "geometry"]].to_crs("EPSG:4326")
    joined = gpd.sjoin(points, countries, how="inner", predicate="within")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def weighted_country_mean(da: xr.DataArray, grid_index: pd.DataFrame) -> pd.Series:
    values = np.asarray(da.values)
    df = grid_index.copy()
    df["value"] = values[df["row"].to_numpy(), df["col"].to_numpy()]
    df = df[np.isfinite(df["value"])]
    return df.groupby("name")[["value", "weight"]].apply(
        lambda x: np.average(x["value"], weights=x["weight"])
    )


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

def _lookup(interp, t_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    pts = np.column_stack([t_c.values, rh_pct.values])
    return pd.Series(interp(pts), index=t_c.index)


def build_factors(
    t_c: pd.Series,
    rh_pct: pd.Series,
    t_ref: float,
    rh_ref: float,
) -> pd.DataFrame:
    out = pd.DataFrame(index=t_c.index.rename("country"))
    out["t_avg_c"] = t_c
    out["rh_avg_pct"] = rh_pct

    # Overall productivity from digitised right panels (for reference / reporting)
    out["ldac_productivity"] = _lookup(_ldac_prod_interp, t_c, rh_pct).clip(0.5, 1.5)
    out["sdac_productivity"] = _lookup(_sdac_prod_interp, t_c, rh_pct).clip(0.5, 1.5)
    out["ldac_cost_mult"] = 1.0 / out["ldac_productivity"]
    out["sdac_cost_mult"] = 1.0 / out["sdac_productivity"]

    # --- Separate electricity and heat scaling factors -----------------------
    # Each factor = (energy demand at country climate) / (energy demand at reference)
    # Values > 1 → country needs more energy per tCO2 than the reference site.
    ref_t = pd.Series([t_ref], index=["ref"])
    ref_rh = pd.Series([rh_ref], index=["ref"])

    ldac_elec_at_country = _lookup(_ldac_elec_interp, t_c, rh_pct)
    ldac_elec_at_ref = float(_lookup(_ldac_elec_interp, ref_t, ref_rh).iloc[0])
    out["ldac_elec_factor"] = (ldac_elec_at_country / ldac_elec_at_ref).clip(0.7, 1.3)

    sdac_elec_at_country = _lookup(_sdac_elec_interp, t_c, rh_pct)
    sdac_elec_at_ref = float(_lookup(_sdac_elec_interp, ref_t, ref_rh).iloc[0])
    out["sdac_elec_factor"] = (sdac_elec_at_country / sdac_elec_at_ref).clip(0.7, 1.3)

    sdac_heat_at_country = _lookup(_sdac_heat_interp, t_c, rh_pct)
    sdac_heat_at_ref = float(_lookup(_sdac_heat_interp, ref_t, ref_rh).iloc[0])
    out["sdac_heat_factor"] = (sdac_heat_at_country / sdac_heat_at_ref).clip(0.7, 1.3)

    return out.reset_index()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-file", type=Path, default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--countries", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/dac_country_weather_factors.csv"))
    parser.add_argument("--t-var", default=None)
    parser.add_argument("--d-var", default=None)
    parser.add_argument(
        "--t-ref", type=float, default=15.0,
        help="Reference temperature (°C) matching the cost table assumptions",
    )
    parser.add_argument(
        "--rh-ref", type=float, default=60.0,
        help="Reference relative humidity (%%) matching the cost table assumptions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = open_weather(args.weather_file)
    t_name = pick_variable(ds, TEMPERATURE_CANDIDATES, args.t_var)
    d_name = pick_variable(ds, DEWPOINT_CANDIDATES, args.d_var)

    t_c = kelvin_to_celsius(ds[t_name])
    td_c = kelvin_to_celsius(ds[d_name])

    t_avg = mean_over_non_spatial_dims(t_c)
    rh_avg = mean_over_non_spatial_dims(relative_humidity_from_dewpoint(t_c, td_c))

    grid_index = country_grid_index(t_avg, args.countries)
    factors = build_factors(
        t_c=weighted_country_mean(t_avg, grid_index),
        rh_pct=weighted_country_mean(rh_avg, grid_index),
        t_ref=args.t_ref,
        rh_ref=args.rh_ref,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  (reference: T={args.t_ref}°C, RH={args.rh_ref}%)")
    cols = ["country", "t_avg_c", "rh_avg_pct",
            "ldac_elec_factor", "sdac_elec_factor", "sdac_heat_factor",
            "ldac_productivity", "sdac_productivity"]
    print(factors[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
