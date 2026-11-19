# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # SST anomalies

# %%
import numpy as np
import xarray as xr


def monthly_anomaly(da):
    """Remove the monthly climatology."""
    clim = da.groupby("time.month").mean("time")
    return da.groupby("time.month") - clim


# %%
sst = xr.DataArray(np.random.rand(24, 4, 4), dims=("time", "lat", "lon"), coords={"time": xr.date_range("2020-01-01", periods=24, freq="MS")})
anom = monthly_anomaly(sst)
