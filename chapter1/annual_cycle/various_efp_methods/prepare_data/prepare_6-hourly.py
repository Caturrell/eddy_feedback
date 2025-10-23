import os
from pathlib import Path

import xarray as xr
import pandas as pd  # kept because you import it elsewhere, but not strictly needed now
import functions.data_wrangling as dw


def split_netcdf_by_year(
    input_path,
    output_dir=None,
    time_dim="time",
    var_name="",
    chunks=None,
    encoding=None,
    combine="by_coords",
    drop_global_attrs=True,
):
    """
    Open a NetCDF dataset (single file or multi-file glob), rename the variable,
    split into yearly datasets, and save each year individually.

    Parameters
    ----------
    input_path : str | Path
        Path or glob pattern for the NetCDF file(s) to open, e.g. '/path/ucomp_*.nc'.
    output_dir : str | Path | None
        Directory in which to save yearly NetCDFs. If None, results are returned but not saved.
    time_dim : str
        Name of the time dimension in the dataset.
    var_name : str
        Desired short name of the variable (used for renaming and filenames).
        If empty, no renaming is performed.
    chunks : dict | None
        Optional dask chunking, e.g. {'time': 96}.
    encoding : dict | None
        Optional NetCDF encoding dict for compression.
    combine : str
        Passed to xarray.open_mfdataset when using multiple files. Default 'by_coords'.
    drop_global_attrs : bool
        If True, clears ds.attrs (keeps variable attrs/encoding).

    Returns
    -------
    dict[int, xr.Dataset]
        Mapping from year -> xarray.Dataset (views of the original dataset graph).
    """
    input_path = Path(input_path) if not isinstance(input_path, str) else input_path
    is_pattern = isinstance(input_path, str) and any(ch in input_path for ch in "*?[]")
    is_dir = isinstance(input_path, (str, Path)) and Path(input_path).is_dir()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    open_kwargs = {}
    if chunks:
        open_kwargs["chunks"] = chunks

    print("=== split_netcdf_by_year: START ===")
    print(f"Input: {input_path}")
    if output_dir:
        print(f"Output directory: {output_dir}")

    # 1) Open dataset (single or multi-file)
    if is_pattern or is_dir:
        pattern = input_path if is_pattern else str(Path(input_path) / "*.nc")
        print(f"Opening multiple files with pattern: {pattern}")
        ds = xr.open_mfdataset(pattern, combine=combine, parallel=True, **open_kwargs)
        print("Opened multi-file dataset.")
    else:
        print(f"Opening single file: {input_path}")
        ds = xr.open_dataset(input_path, **open_kwargs)
        print("Opened single-file dataset.")

    try:
        # Basic checks
        if time_dim not in ds.coords and time_dim not in ds.dims:
            raise ValueError(
                f"Time dimension '{time_dim}' not found in dataset. "
                f"Found dims: {list(ds.dims)}"
            )

        # 2) Rename variable (if requested)
        if len(ds.data_vars) == 0:
            raise ValueError("No data variables found in the dataset.")

        internal_var = list(ds.data_vars)[0]  # choose first if generic name like 'var39'
        print(f"Detected data variable: '{internal_var}'")
        if var_name:
            if internal_var != var_name:
                print(f"Renaming variable '{internal_var}' -> '{var_name}'")
                ds = ds.rename({internal_var: var_name})
                target_var = var_name
            else:
                print(f"Variable already named '{var_name}', no rename needed.")
                target_var = var_name
        else:
            print("No 'var_name' provided; leaving variable name unchanged.")
            target_var = internal_var

        # Optional: drop global attrs
        if drop_global_attrs:
            ds.attrs = {}
            print("Cleared global attributes on dataset (kept variable attrs).")

        # Project-specific checker
        print("Running project data checker (dw.data_checker1000)...")
        ds = dw.data_checker1000(ds, check_vars=False)
        print("Project data checker complete.")

        # 3) Split into years
        print(f"Splitting by year along '{time_dim}'...")
        groups = ds.groupby(f"{time_dim}.year")
        years = list(groups.groups.keys())
        print(f"Found {len(years)} year(s): {sorted([int(y) for y in years])}")

        yearly_datasets = {}

        # 4) Save each year (and build mapping)
        for year, ds_year in groups:
            # ensure a clean coordinate (optional)
            ds_year = ds_year.assign_coords(year=int(year)).drop_vars(
                ["year"], errors="ignore"
            ).assign_coords(year=int(year))

            yearly_datasets[int(year)] = ds_year

            if output_dir is not None:
                # Keep your original naming convention with suffix
                out_name = f"{int(year)}_{target_var}_6hourly_uvtw.nc"
                out_path = output_dir / out_name
                print(f"Writing year {int(year)} -> {out_path}")
                ds_year.to_netcdf(out_path, encoding=encoding or {})
                print(f"Saved: {out_path}")

        print("=== split_netcdf_by_year: DONE ===")
        return yearly_datasets

    finally:
        # Keep ds open if returning lazy graphs? In practice, closing is fine because
        # xarray/dask will reopen as needed when computing on ds_year fragments.
        try:
            ds.close()
            print("Closed dataset handle.")
        except Exception:
            pass


if __name__ == "__main__":
    # set paths
    source_path = Path("/disca/share/sit204/jra_55/1958_2016_6hourly_data_efp/full_6hourly_snapshots")
    dest_path = Path("/home/links/ct715/data_storage/reanalysis/jra55_daily/split_years_1958-2016/6h_uvtw")
    dest_path.mkdir(parents=True, exist_ok=True)

    # variables to process (omega, temp, ucomp, vcomp)
    variables = ["omega", "temp", "ucomp", "vcomp"]

    # optional: compression for modest size savings
    def make_encoding(var):
        return {var: {"zlib": True, "complevel": 4}}

    for var in variables:
        print("\n" + "=" * 60)
        print(f"Processing variable: {var}")
        output_folder = dest_path / var
        file_pattern = str(source_path / f"{var}*.nc")
        
        if output_folder.exists():
            print(f"Output folder '{output_folder}' already exists. Skipping variable '{var}'.")
            continue

        yearly_data = split_netcdf_by_year(
            input_path=file_pattern,        # open directly from the source pattern
            output_dir=output_folder,       # save each year into var-specific folder
            time_dim="time",
            var_name=var,                   # rename internal var (e.g., var39) -> var
            chunks={"time": 96},            # chunking for performance
            encoding=make_encoding(var),    # per-var compression
            combine="by_coords",
            drop_global_attrs=True,
        )

        print(f"Split '{var}' into {len(yearly_data)} years: {sorted(yearly_data.keys())}")
