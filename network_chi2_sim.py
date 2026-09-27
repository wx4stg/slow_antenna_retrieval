#!/usr/bin/env python3
# Simulate a network of slow antenna stations and get location-dependent error distribution
# Created 29 April 2026 by Sam Gardner <samuel.gardner@ttu.edu>


import argparse
from sa_retrieve import monopole_E_change, multi_monopole_retrieval_dask
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import chi2
from pyxlma import coords
from pyhigh import get_elevation_batch
from shapely import wkt
import re


def station_geometry_to_x_y_z(station_df):
    """
    Convert station latitude, longitude, and altitude to x, y, z coordinates in a tangent plane Cartesian system.

    Parameters
    ----------
    station_df : pd.DataFrame
        A DataFrame containing the station information, including 'lat', 'lon', and 'alt' columns for the station coordinates.

    Returns
    -------
    station_df : pd.DataFrame
        The input DataFrame with additional 'x', 'y', and 'z' columns for the station coordinates in the tangent plane Cartesian system.
    """
    station_df['geometry'] = station_df['WKT'].apply(wkt.loads)
    station_df['lon'] = station_df['geometry'].apply(lambda x: x.x)
    station_df['lat'] = station_df['geometry'].apply(lambda x: x.y)
    station_df['alt'] = get_elevation_batch(list(zip(station_df['lat'].values, station_df['lon'].values)))
    grid_ctr_lat = station_df['lat'].mean()
    grid_ctr_lon = station_df['lon'].mean()
    grid_ctr_alt = station_df['alt'].mean()
    geosys = coords.GeographicSystem()
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=grid_ctr_lat, ctrLon=grid_ctr_lon, ctrAlt=grid_ctr_alt)
    station_X, station_Y, station_Z = geosys.toECEF(station_df['lon'].values, station_df['lat'].values, station_df['alt'].values)
    station_x, station_y, station_z = tpcs.fromECEF(station_X, station_Y, station_Z)
    station_df['x'] = station_x
    station_df['y'] = station_y
    station_df['z'] = station_z
    return station_df


def read_google_maps_df(path):
    """
    Read a CSV file containing station information from Google Maps and return a DataFrame.

    Parameters
    ----------
    path : str
        The path to the CSV file containing the station information.

    Returns
    -------
    station_df : pd.DataFrame
        A DataFrame containing the station information, including 'WKT', 'name', 'lat', 'lon', and 'alt' columns.
    """
    rows = []
    with open(path) as f:
        next(f)  # skip header line
        for line in f:
            line = line.rstrip('\n')
            # WKT is always the first quoted field — capture it and everything after the comma
            m = re.match(r'^"([^"]+)",(.*)', line)
            if m:
                rows.append({'WKT': m.group(1), 'name': m.group(2)})

    station_df = pd.DataFrame(rows)
    station_df[['name', 'description']] = station_df['name'].str.split(',', n=1, expand=True)
    station_df = station_df.drop(columns=['description'])
    return station_geometry_to_x_y_z(station_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate a network of slow antenna stations and get location-dependent error distribution.')
    parser.add_argument('--station_csv', type=str, default='station_locations.csv', help='Path to the CSV file containing station information.')
    parser.add_argument('--L_x', type=float, default=30, help='Length of the grid in the x direction (km).')
    parser.add_argument('--L_y', type=float, default=30, help='Length of the grid in the y direction (km).')
    parser.add_argument('--L_z', type=float, default=10, help='Length of the grid in the z direction (km).')
    parser.add_argument('--n_x', type=int, default=31, help='Number of grid points in the x direction.')
    parser.add_argument('--n_y', type=int, default=31, help='Number of grid points in the y direction.')
    parser.add_argument('--n_z', type=int, default=10, help='Number of grid points in the z direction.')
    parser.add_argument('-q', '--Q_coloumbs', type=float, default=3, help='Charge of the monopole in coloumbs.')
    parser.add_argument('--error_scale', type=float, default=150, help='Scale of the Gaussian noise to be added to the delta E values (V/m).')
    parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size for Dask array processing.')
    args = parser.parse_args()
    L_x, n_x = args.L_x, args.n_x
    L_y, n_y = args.L_y, args.n_y
    L_z, n_z = args.L_z, args.n_z
    Q_coloumbs = args.Q_coloumbs
    error_scale = args.error_scale # V/m
    chunk_size = args.chunk_size
    # Step 0: read in station dataframe
    # station_df = read_google_maps_df('station_locations.csv')
    # station_df = station_geometry_to_x_y_z(station_df)
    station_df = pd.read_csv(args.station_csv)
    # step 1 -- create grids of X, Y, Z, Q in shape (Nx, Ny, Nz)
    x_vals = np.linspace(-L_x/2, L_x/2, n_x)*1e3
    y_vals = np.linspace(-L_y/2, L_y/2, n_y)*1e3
    z_vals = np.linspace(0, L_z, n_z)*1e3
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals)
    q_grid = np.full_like(x_grid, Q_coloumbs)

    x_grid_flat, y_grid_flat, z_grid_flat = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    q_grid_flat = q_grid.flatten()
    # step 2 -- calculate "perfect" delta E at station locations shape Nx*Ny*Nz, N_stations
    delta_e_ideal = monopole_E_change(x_grid_flat, y_grid_flat, z_grid_flat, q_grid_flat, station_df['x'].values, station_df['y'].values, station_df['z'].values)
    # step 3 -- add gaussian noise to the delta E values to simulate instrument error
    delta_e_with_error = delta_e_ideal + np.random.normal(loc=0, scale=error_scale, size=delta_e_ideal.shape)
    # step 4 -- use noisy delta E values to retrieve x, y, z, and q of shape (Nx, Ny, Nz)
    retrieved_x, retrieved_y, retrieved_z, retrieved_q = multi_monopole_retrieval_dask(delta_e_with_error, station_df, initial_guess_q=0, chunk_size=chunk_size)
    # step 5 -- use retrieved grid to simulate delta E at the station locations
    delta_e_retrieved = monopole_E_change(retrieved_x.flatten(), retrieved_y.flatten(), retrieved_z.flatten(), retrieved_q.flatten(), station_df['x'].values, station_df['y'].values, station_df['z'].values)
    # step 6 -- calculate and fit a chi2 distribution
    err_dist_normalized = (delta_e_retrieved - delta_e_ideal) / error_scale
    reduced_chi2_observed = np.sum(err_dist_normalized**2, axis=1)
    reduced_chi2_fit = chi2.fit(reduced_chi2_observed)
    print(f'Fitted Chi2 parameters: DoF={reduced_chi2_fit[0]:.2f}, loc={reduced_chi2_fit[1]:.2f}, scale={reduced_chi2_fit[2]:.2f}')
    expected_chi2 = chi2.pdf(np.arange(0, np.max(reduced_chi2_observed), 0.1), df=4, loc=0, scale=1)
    fitted_chi2 = chi2.pdf(np.arange(0, np.max(reduced_chi2_observed), 0.1), df=reduced_chi2_fit[0], loc=reduced_chi2_fit[1], scale=reduced_chi2_fit[2])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(reduced_chi2_observed, bins=50, density=True, alpha=0.5, label='Observed Reduced Chi2')
    ax.plot(np.arange(0, np.max(reduced_chi2_observed), 0.1), expected_chi2, label='Expected Chi2 PDF', color='tab:red')
    ax.plot(np.arange(0, np.max(reduced_chi2_observed), 0.1), fitted_chi2, label='Fitted Chi2 PDF', color='tab:green')
    ax.set_xlabel('Error Distribution')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.set_title('Chi2 Distribution of Retrieval Errors')
    fig.savefig('chi2_distribution.png', dpi=300)

