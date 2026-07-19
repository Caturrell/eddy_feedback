"""
Collate JRA55 tau_fit_3 lag values (from the phase-difference cospectrum
fit in compute_phase_difference(), functions/SIT_functions/SIT_eddy_feedback_functions.py)
into a single long-format CSV.

Source: power_spec.nc in
b-parameter/b_methodology/all_plots_true/jra55_850_sit_plots/1979_2014/6hourly/level_250_500_850hPa/

Extracted across:
- season: all_time + the 12 rolling 3-month seasons (DJF, JFM, ..., NDJ)
- hemisphere: n, s
- wind_variant: native ("full" ucomp) and va (vertically-averaged 250/500/850hPa ucomp_va)

Only div1_QG (all wavenumbers) is extracted - the div1_QG_123 (wavenumbers 1-3) and
div1_QG_gt3 (wavenumbers >3) variants are intentionally excluded.

The 500hPa-only ucomp_500 variant (present in b_dataset.nc) is intentionally excluded -
tau was never computed for it in power_spec.nc.
"""

from pathlib import Path

import pandas as pd
import xarray as xr

SOURCE_NC = Path(
    "/home/links/ct715/eddy_feedback/b-parameter/b_methodology/all_plots_true/"
    "jra55_850_sit_plots/1979_2014/6hourly/level_250_500_850hPa/power_spec.nc"
)
OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "jra55_tau_fit_3.csv"

SEASONS = [
    "all_time",
    "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
    "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
]
HEMISPHERES = ["n", "s"]
QG_MODES = ["div1_QG"]
WIND_VARIANTS = {"native": "", "va": "_va"}


def var_name(qg_mode: str, variant_suffix: str, hemi: str, season: str) -> str:
    ucomp = f"ucomp{variant_suffix}"
    return (
        f"{qg_mode}{variant_suffix}_PCs_from_{ucomp}_{hemi}_{season}_"
        f"{ucomp}_PCs_{hemi}_{season}_phase_diff_tau_fit_3"
    )


def main():
    ds = xr.open_dataset(SOURCE_NC)

    rows = []
    missing = []
    for season in SEASONS:
        for hemi in HEMISPHERES:
            for qg_mode in QG_MODES:
                for variant, suffix in WIND_VARIANTS.items():
                    name = var_name(qg_mode, suffix, hemi, season)
                    if name not in ds:
                        missing.append(name)
                        continue
                    rows.append(
                        {
                            "dataset": "jra55",
                            "season": season,
                            "hemisphere": hemi,
                            "qg_mode": qg_mode,
                            "wind_variant": variant,
                            "tau_fit_3": float(ds[name].values),
                        }
                    )

    if missing:
        raise RuntimeError(
            f"{len(missing)} expected tau_fit_3 variables not found in {SOURCE_NC}, "
            f"e.g. {missing[:5]}"
        )

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
