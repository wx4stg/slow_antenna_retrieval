#!/usr/bin/env python3
# Helper to get sensor lat/lon/alt from directory of sensor data netCDF files

import argparse
from glob import glob

import pandas as pd
import xarray as xr
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save sensor lat/lon/alt to CSV from directory of sensor data netCDF files')
    parser.add_argument('-i', '--input', required=True, help='Directory containing processed sensor data subdirectories of netCDF files')
    args = parser.parse_args()

    # Open the dataset using xarray
    sensor_dirs = sorted(glob(f"{args.input}/processed/sensor_*/"))
    df = pd.DataFrame(columns=["sensor_num", "lat", "lon", "alt"])
    for s in sensor_dirs:
        all_netcdfs = glob(f'{s}/*.nc')
        for f in all_netcdfs:
            ds = xr.open_dataset(f)
            print(ds)
            this_lat = np.nanmean(ds['lat'].values)
            this_lon = np.nanmean(ds['lon'].values)
            this_alt = np.nanmean(ds['alt'].values)
            sensor_num = ds.sensor_num.values.item()
            if np.isnan(this_lat) or np.isnan(this_lon) or np.isnan(this_alt):
                continue
            if this_lat == 0 or this_lon == 0 or this_alt == 0:
                continue
            else:
                break
        df = pd.concat([df, pd.DataFrame({"sensor_num": [sensor_num],
                                          "lat": [this_lat],
                                          "lon": [this_lon],
                                          "alt": [this_alt]})], ignore_index=True)
    df.to_csv('sensor_locations.csv', index=False)