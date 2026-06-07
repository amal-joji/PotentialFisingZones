# ============================================================
# PFZ PROBABILITY MAP PIPELINE (FINAL WORKING VERSION)
# ============================================================

import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER INPUTS ----------------
EARTHDATA_TOKEN = "TOKEN"


LON_MIN, LON_MAX = 79.5, 82.5   # degrees East
LAT_MIN, LAT_MAX = 10.5, 14.0

CURRENTS_FILE = "currents_tn_coast.nc"
HEADERS = {"Authorization": f"Bearer {EARTHDATA_TOKEN}"}

# ---------------- HELPER FUNCTIONS ----------------
def download_file(url, filename):
    r = requests.get(url, headers=HEADERS, stream=True)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"Downloaded: {filename}")


def standardize_coords(ds):
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    return ds


def adaptive_clip(ds):
    """
    Handles:
    - longitude: [-180,180] or [0,360]
    - latitude: ascending or descending (MODIS!)
    """
    # ---- longitude ----
    lon_min_file = float(ds.lon.min())
    if lon_min_file < 0:
        lon_slice = slice(LON_MIN, LON_MAX)
    else:
        lon_slice = slice(LON_MIN % 360, LON_MAX % 360)

    # ---- latitude (CRITICAL FIX) ----
    lat_vals = ds.lat.values
    if lat_vals[0] < lat_vals[-1]:   # ascending
        lat_slice = slice(LAT_MIN, LAT_MAX)
    else:                            # descending (MODIS)
        lat_slice = slice(LAT_MAX, LAT_MIN)

    ds = ds.sel(lon=lon_slice, lat=lat_slice)

    if ds.sizes["lat"] == 0 or ds.sizes["lon"] == 0:
        raise RuntimeError(
            f"❌ Empty dataset after clipping\n"
            f"Lon range in file: {float(ds.lon.min())} to {float(ds.lon.max())}"
        )
    return ds


def normalize(x):
    xmin = x.min(skipna=True)
    xmax = x.max(skipna=True)
    return (x - xmin) / (xmax - xmin)

# ============================================================
# 1. CHLOROPHYLL-a (MODIS)
# ============================================================
chl_url = (
    "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
    "AQUA_MODIS.20240101.L3m.DAY.CHL.chlor_a.4km.nc"
)

download_file(chl_url, "chlorophyll.nc")

chl_ds = xr.open_dataset("chlorophyll.nc")
chl_ds = standardize_coords(chl_ds)
chl_ds = adaptive_clip(chl_ds)

chl = chl_ds["chlor_a"]

print("Chl shape:", chl.shape)

# ============================================================
# 2. SST (MODIS)
# ============================================================
sst_url = (
    "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
    "AQUA_MODIS.20240101.L3m.DAY.SST.sst.4km.nc"
)

download_file(sst_url, "sst.nc")

sst_ds = xr.open_dataset("sst.nc")
sst_ds = standardize_coords(sst_ds)
sst_ds = adaptive_clip(sst_ds)

sst = sst_ds["sst"]

print("SST shape:", sst.shape)

# ============================================================
# 3. CMEMS OCEAN CURRENTS
# ============================================================
curr_ds = xr.open_dataset(CURRENTS_FILE)

# surface layer + single time
curr_ds = curr_ds.isel(depth=0, time=0)

curr_ds = curr_ds.rename({"latitude": "lat", "longitude": "lon"})
curr_ds = adaptive_clip(curr_ds)

u = curr_ds["uo"]
v = curr_ds["vo"]

# regrid currents to MODIS grid (robust)
u_i = u.interp_like(chl, method="nearest")
v_i = v.interp_like(chl, method="nearest")

# ============================================================
# 4. PFZ PROBABILITY (NO ML)
# ============================================================
chl_n = normalize(chl)
sst_n = normalize(sst)

current_speed = np.sqrt(u_i**2 + v_i**2)
cur_n = normalize(current_speed)

pfz_prob = (
    0.5 * chl_n +
    0.3 * (1 - sst_n) +
    0.2 * cur_n
).clip(0, 1)

print("PFZ probability range:",
      float(pfz_prob.min()), float(pfz_prob.max()))

# ============================================================
# 5. PFZ HEATMAP
# ============================================================
plt.figure(figsize=(8, 6))
plt.pcolormesh(
    chl.lon,
    chl.lat,
    pfz_prob,
    shading="auto",
    cmap="jet",
    vmin=0,
    vmax=1
)
plt.colorbar(label="PFZ Probability")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title("PFZ Probability Map (Tamil Nadu Coast)")
plt.tight_layout()
plt.show()

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================
pfz_prob.to_netcdf("pfz_probability_tn_coast.nc")
plt.imsave("pfz_probability_map.png", pfz_prob, cmap="jet")

print("PFZ probability map generated successfully")

# ============================================================
# PFZ DRIFT PREDICTION (24 / 48 / 72 HOURS)
# ============================================================

import scipy.ndimage as nd

# --- Earth radius for conversion ---
R = 6371000  # meters

# --- Compute grid spacing in meters ---
dlat = np.abs(chl.lat[1] - chl.lat[0]).values
dlon = np.abs(chl.lon[1] - chl.lon[0]).values

# Convert degree spacing to meters
dy = dlat * (np.pi/180) * R
dx = dlon * (np.pi/180) * R * np.cos(np.deg2rad(chl.lat))

# Average dx across latitudes for simplicity
dx_mean = float(dx.mean())

# ------------------------------------------------------------
# Function to advect PFZ probability
# ------------------------------------------------------------
def advect_field(field, u, v, hours):
    dt = hours * 3600  # convert hours to seconds
    
    # displacement in grid cells
    shift_x = (u * dt) / dx_mean
    shift_y = (v * dt) / dy
    
    # average displacement
    mean_shift_x = float(shift_x.mean())
    mean_shift_y = float(shift_y.mean())
    
    drifted = nd.shift(
        field,
        shift=(-mean_shift_y, -mean_shift_x),
        order=1,
        mode="nearest"
    )
    
    return drifted

# ------------------------------------------------------------
# Generate drift predictions
# ------------------------------------------------------------
pfz_24h = advect_field(pfz_prob, u_i, v_i, 24)
pfz_48h = advect_field(pfz_prob, u_i, v_i, 48)
pfz_72h = advect_field(pfz_prob, u_i, v_i, 72)

# ------------------------------------------------------------
# Plot 24h Drift
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.pcolormesh(chl.lon, chl.lat, pfz_24h,
               shading="auto", cmap="jet", vmin=0, vmax=1)
plt.colorbar(label="PFZ Probability (24h Forecast)")
plt.title("PFZ Drift Prediction (24 Hours)")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 48h Drift
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.pcolormesh(chl.lon, chl.lat, pfz_48h,
               shading="auto", cmap="jet", vmin=0, vmax=1)
plt.colorbar(label="PFZ Probability (48h Forecast)")
plt.title("PFZ Drift Prediction (48 Hours)")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot 72h Drift
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.pcolormesh(chl.lon, chl.lat, pfz_72h,
               shading="auto", cmap="jet", vmin=0, vmax=1)
plt.colorbar(label="PFZ Probability (72h Forecast)")
plt.title("PFZ Drift Prediction (72 Hours)")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.tight_layout()
plt.show()

print("PFZ drift prediction generated (24/48/72 hours)")

# ============================================================
# CONVERT PFZ DATA INTO ML TRAINING DATASET
# ============================================================

import pandas as pd

# --- Flatten all feature grids ---
chl_flat = chl.values.flatten()
sst_flat = sst.values.flatten()
u_flat = u_i.values.flatten()
v_flat = v_i.values.flatten()
pfz_flat = pfz_prob.values.flatten()

# Derived feature
current_speed_flat = np.sqrt(u_flat**2 + v_flat**2)

# Latitude & longitude grids
lon_grid, lat_grid = np.meshgrid(chl.lon.values, chl.lat.values)

lon_flat = lon_grid.flatten()
lat_flat = lat_grid.flatten()

# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------
df = pd.DataFrame({
    "chlorophyll": chl_flat,
    "sst": sst_flat,
    "u_current": u_flat,
    "v_current": v_flat,
    "current_speed": current_speed_flat,
    "latitude": lat_flat,
    "longitude": lon_flat,
    "pfz_probability": pfz_flat
})

# ------------------------------------------------------------
# Remove NaNs
# ------------------------------------------------------------
df = df.dropna().reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# XGBOOST TRAINING
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb


MODE = "regression"   

if MODE == "regression":
    y = df["pfz_probability"]
else:
    df["pfz_zone"] = (df["pfz_probability"] > 0.6).astype(int)
    y = df["pfz_zone"]

X = df.drop(columns=["pfz_probability"], errors="ignore")
X = X.drop(columns=["pfz_zone"], errors="ignore")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
if MODE == "regression":
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )
else:
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )

model.fit(X_train, y_train)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
y_pred = model.predict(X_test)

if MODE == "regression":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
else:
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

# ============================================================
# GENERATE 30 DAYS OF PFZ FILES (AUTO LOOP)
# ============================================================

from datetime import datetime, timedelta

START_DATE = datetime(2024, 1, 1)
DAYS = 30

for i in range(DAYS):

    date = START_DATE + timedelta(days=i)
    date_str = date.strftime("%Y%m%d")

    print("Processing:", date_str)

    # ------------------------------------------------------------
    # MODIS URLs (date-based)
    # ------------------------------------------------------------
    chl_url = (
        f"https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
        f"AQUA_MODIS.{date_str}.L3m.DAY.CHL.chlor_a.4km.nc"
    )

    sst_url = (
        f"https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
        f"AQUA_MODIS.{date_str}.L3m.DAY.SST.sst.4km.nc"
    )

    # ------------------------------------------------------------
    # Download daily files
    # ------------------------------------------------------------
    download_file(chl_url, f"chlorophyll_{date_str}.nc")
    download_file(sst_url, f"sst_{date_str}.nc")

    # ------------------------------------------------------------
    # Load + clip (reuse your functions)
    # ------------------------------------------------------------
    chl_ds = adaptive_clip(
        standardize_coords(
            xr.open_dataset(f"chlorophyll_{date_str}.nc")
        )
    )

    sst_ds = adaptive_clip(
        standardize_coords(
            xr.open_dataset(f"sst_{date_str}.nc")
        )
    )

    chl = chl_ds["chlor_a"]
    sst = sst_ds["sst"]

    # ------------------------------------------------------------
    # Load CMEMS currents (same file for now)
    # ------------------------------------------------------------
    curr_ds = adaptive_clip(
        xr.open_dataset(CURRENTS_FILE)
        .isel(depth=0, time=0)
        .rename({"latitude": "lat", "longitude": "lon"})
    )

    u = curr_ds["uo"]
    v = curr_ds["vo"]

    u_i = u.interp_like(chl, method="nearest")
    v_i = v.interp_like(chl, method="nearest")

    # ------------------------------------------------------------
    # Compute PFZ probability
    # ------------------------------------------------------------
    chl_n = normalize(chl)
    sst_n = normalize(sst)
    current_speed = np.sqrt(u_i**2 + v_i**2)
    cur_n = normalize(current_speed)

    pfz_prob = (
        0.5 * chl_n +
        0.3 * (1 - sst_n) +
        0.2 * cur_n
    ).clip(0, 1)

# ------------------------------------------------------------
# Save processed daily outputs (NEW FILE NAMES)
# ------------------------------------------------------------

    chl_ds.to_netcdf(f"chl_processed_{date_str}.nc")
    sst_ds.to_netcdf(f"sst_processed_{date_str}.nc")
    curr_ds.to_netcdf(f"currents_processed_{date_str}.nc")
    pfz_prob.name = "pfz_probability"
    pfz_prob.to_netcdf(f"pfz_{date_str}.nc")



print("30-day PFZ dataset generated")


# ============================================================
# 30-DAY TIME SERIES STACKING FOR DYNAMIC PFZ MODEL
# ============================================================

import os
import pandas as pd
from datetime import datetime, timedelta

# ------------------------------------------
# CONFIG
# ------------------------------------------
START_DATE = datetime(2024, 1, 1)
DAYS = 30

all_days_data = []

# ------------------------------------------
# LOOP THROUGH 30 DAYS
# ------------------------------------------
for i in range(DAYS):

    date = START_DATE + timedelta(days=i)
    date_str = date.strftime("%Y%m%d")

    chl_ds = xr.open_dataset(f"chl_processed_{date_str}.nc")
    sst_ds = xr.open_dataset(f"sst_processed_{date_str}.nc")
    curr_ds = xr.open_dataset(f"currents_processed_{date_str}.nc")
    pfz_ds = xr.open_dataset(f"pfz_{date_str}.nc")

    chl = chl_ds["chlor_a"]
    sst = sst_ds["sst"]
    pfz = pfz_ds["pfz_probability"]

    u = curr_ds["uo"]
    v = curr_ds["vo"]

    #  Align currents to chlorophyll grid
    u = u.interp_like(chl, method="nearest")
    v = v.interp_like(chl, method="nearest")

    current_speed = np.sqrt(u**2 + v**2)

    df_day = pd.DataFrame({
        "chlorophyll": chl.values.flatten(),
        "sst": sst.values.flatten(),
        "u": u.values.flatten(),
        "v": v.values.flatten(),
        "current_speed": current_speed.values.flatten(),
        "pfz_probability": pfz.values.flatten(),
        "date": date
    })

    df_day = df_day.dropna()
    all_days_data.append(df_day)

df_all = pd.concat(all_days_data).reset_index(drop=True)

print("Total dataset shape:", df_all.shape)

df_all = pd.concat(all_days_data).reset_index(drop=True)

print("Total dataset shape:", df_all.shape)

# ============================================================
# CREATE LAG FEATURES  ← ADD HERE
# ============================================================

df_all = df_all.sort_values("date")

points_per_day = len(df_all) // DAYS

lags = [1, 2, 3, 5, 7]

for lag in lags:
    df_all[f"chl_lag_{lag}"] = df_all.groupby(
        df_all.index % points_per_day
    )["chlorophyll"].shift(lag)

    df_all[f"sst_lag_{lag}"] = df_all.groupby(
        df_all.index % points_per_day
    )["sst"].shift(lag)

    df_all[f"pfz_lag_{lag}"] = df_all.groupby(
        df_all.index % points_per_day
    )["pfz_probability"].shift(lag)

df_all = df_all.dropna().reset_index(drop=True)

print("Lagged dataset shape:", df_all.shape)

# ============================================================
# TRAIN DYNAMIC MODEL
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

X = df_all.drop(columns=["pfz_probability", "date"])
y = df_all["pfz_probability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.03
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Dynamic Model RMSE:", rmse)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# R² Score
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R² Score:", r2)

import pandas as pd
import numpy as np

# ------------------------------------------------
# Extract feature importance
# ------------------------------------------------
importance = model.feature_importances_

feature_names = X.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance Score": importance
})

# Sort by importance
fi_df = fi_df.sort_values(by="Importance Score", ascending=False)
fi_df = fi_df.reset_index(drop=True)

# Add Rank column
fi_df["Rank"] = fi_df.index + 1

# Reorder columns
fi_df = fi_df[["Rank", "Feature", "Importance Score"]]

print(fi_df.head(10))