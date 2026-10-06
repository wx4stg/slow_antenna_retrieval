#!/usr/bin/env python3
# Support functions for slow antenna charge retrieval
# Created 29 April 2026 by Sam Gardner <samuel.gardner@ttu.edu>


import numpy as np
from scipy.optimize import least_squares
from functools import partial


EPSILON_0 = 8.8541e-12 # F/m


def monopole_E_change(x, y, z, q, xi, yi, zi):
    """
    Calculate the change in electric field due to a monopole charge at a given position.

    Parameters
    ----------
    x, y, z, q : array-like
        The x, y, z coordinates of the charge (stroke) positions. Shape should be (N_strokes,).
    xi, yi, zi : array-like
        The x, y, z coordinates of the station positions. Shape should be (N_stations,).

    Returns
    -------
    delta_E : ndarray
        The change in electric field at each station due to each charge. Shape will be (N_strokes, N_stations).
    """
    # reshape so that all 'stroke' parameters are on axis 0
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    q = np.array(q).reshape(-1, 1)
    # reshape so that all 'station' parameters are on axis 1
    xi = np.array(xi).reshape(1, -1)
    yi = np.array(yi).reshape(1, -1)
    zi = np.array(zi).reshape(1, -1)
    # compute r vector from stroke to station
    dx = x - xi
    dy = y - yi
    dz = z - zi
    r = np.array((dx, dy, dz))
    # calculate E-field change using monopole formula, \frac{1}{4\pi\epsilon_0} \frac{2qz}{r^3}
    k_const = 1/(4*np.pi*EPSILON_0)
    delta_E = k_const * (2*q*dz / (np.linalg.norm(r, axis=0)**3) )
    return delta_E # V/m


def monopole_delta_E_error(params, observed_E, station_xyz, restrict_x=None, restrict_y=None, restrict_z=None):
    """
    Calculate the error between the observed electric field changes and the predicted electric field changes for a given set of monopole parameters.

    Parameters
    ----------
    params : array-like
        An array containing the monopole parameters (q, x, y, z) to be optimized. Shape should be (4,).
    observed_E : array-like
        A 1D array of shape (N_stations,) containing the observed electric field changes at each station for the stroke.
    station_xyz : array-like
        A 2D array of shape (3, N_stations) containing the x, y, and z coordinates of each station.

    Returns
    -------
    error : ndarray
        A 1D array of shape (N_stations,) containing the error between the observed electric field changes and the predicted electric field changes for the given monopole parameters.
    """
    if restrict_x is not None:
        x = restrict_x
    else:
        x = params[1]
    if restrict_y is not None:
        y = restrict_y
    else:
        y = params[2]
    if restrict_z is not None:
        z = restrict_z
    else:
        z = params[3]
    q = params[0]
    predicted_E = monopole_E_change(x, y, z, q, station_xyz[0], station_xyz[1], station_xyz[2])
    return observed_E - np.reshape(predicted_E, observed_E.shape)


def monopole_delta_E_jacobian(params, observed_E, station_xyz):
    q, x, y, z = params
    dx = x - station_xyz[0]
    dy = y - station_xyz[1]
    dz = z - station_xyz[2]
    r_mag_squared = dx**2 + dy**2 + dz**2
    k = 1/(4*np.pi*EPSILON_0)
    partialE_partialq = k * (2*dz / r_mag_squared**(3/2))
    partialE_partialx = k * (-3*q*dz*r_mag_squared**(-5/2) * 2 * dx)
    partialE_partialy = k * (-3*q*dz*r_mag_squared**(-5/2) * 2 * dy)
    partialE_partialz = k * (2*q) * (r_mag_squared**(-3/2) - 3 * dz**2 * r_mag_squared**(-5/2))
    return -np.vstack((partialE_partialq, partialE_partialx, partialE_partialy, partialE_partialz)).T


def monopole_charge_retrieval(initial_guess_q, initial_guess_x, initial_guess_y, initial_guess_z, stroke_obs, station_xyz, bounds=None, return_residual=False):
    """
    Retrieve the location and charge removed by a monopole discharge stroke.

    Parameters
    ----------
    initial_guess_q : float
        An initial guess for the charge of the stroke.
    initial_guess_x : float
        An initial guess for the x coordinate of the stroke.
    initial_guess_y : float
        An initial guess for the y coordinate of the stroke.
    initial_guess_z : float
        An initial guess for the z coordinate of the stroke.
    stroke_obs : np.ndarray
        A 1D array of shape (N_stations,) containing the observed electric field changes at each station for the stroke.
    station_xyz : array-like
        A 2D array of shape (3, N_stations) containing the x, y, and z coordinates of each station.
    bounds : tuple, optional
        A tuple of the form (lower_bounds, upper_bounds) where lower_bounds and upper_bounds are arrays of the same shape as the parameter vector.
        Default is None (-inf to inf for x,y,q; lowest station height to inf for z).
    return_residual : bool, optional
        If True, return the residuals of the least squares optimization. Default is False.

    Returns
    -------
    retrieved_opt : np.ndarray
        A 1D array of shape (4,) containing the retrieved parameters (q, x, y, z) of the stroke. Values will be NaN if the optimization fails.
    residuals : np.ndarray, optional
        A 1D array of shape (N_stations,) containing the residuals of the least squares optimization. Only returned if return_residual is True.
    """
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
    else:
        lower_bounds = [-np.inf, -np.inf, -np.inf, station_xyz[2].min()]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
    try:
        retrieved_opt = least_squares(monopole_delta_E_error,
                    x0=np.array([initial_guess_q, initial_guess_x, initial_guess_y, initial_guess_z]),
                    bounds=(lower_bounds, upper_bounds),
                    args=(stroke_obs, station_xyz), jac=monopole_delta_E_jacobian).x
    except ValueError as e:
                print(f"Initial Guess: q={initial_guess_q}, x={initial_guess_x}, y={initial_guess_y}, z={initial_guess_z}")
                print(f"Observations: {stroke_obs}")
                print(f"Error: {e}")
                if return_residual:
                    return np.array([np.nan, np.nan, np.nan, np.nan]), np.array([np.nan]*len(stroke_obs))
                return np.array([np.nan, np.nan, np.nan, np.nan])
    if return_residual:
        residuals = monopole_delta_E_error(retrieved_opt, stroke_obs, station_xyz)
        return retrieved_opt, residuals
    return retrieved_opt


def monopole_q_analytic(x, y, z, stroke_obs, station_xyz):
    """
    Analytically retrieve the charge of a monopole discharge stroke given its location and the observed electric field changes at each station.

    Parameters
    ----------
    x, y, z : float
        The x, y, z coordinates of the stroke to be retrieved.
    stroke_obs : np.ndarray
            A 1D array of shape (N_stations,) containing the observed electric field changes at each station for the stroke.
    station_xyz : array-like
        A 2D array of shape (3, N_stations) containing the x, y, and z coordinates of each station.

    Returns
    -------
    q : float
        The analytically retrieved charge of the stroke.
    """
    dx = x - station_xyz[0]
    dy = y - station_xyz[1]
    dz = z - station_xyz[2]
    r_mag_squared = dx**2 + dy**2 + dz**2
    k = 1/(4*np.pi*EPSILON_0)
    g = 2*k*dz / r_mag_squared**(3/2) # dE per coulomb at each station
    q = np.sum(stroke_obs * g) / np.sum(g**2) # (dE * (dE/C)) / ((dE/C)^2) = C least squares solution
    return q



def monopole_retrieval_geographic(stroke_obs, station_df, bounds=None, return_residual=False):
    """
    Retrieve the location and charge removed by a monopole discharge stroke using geographic coordinates.

    Parameters
    ----------
    stroke_obs : np.ndarray
        A 1D array of shape (N_stations,) containing the observed electric field changes at each station for the stroke.
    station_df : pd.DataFrame
        A DataFrame containing the station information, including 'lon', 'lat', and 'alt' columns for the station coordinates.
    bounds : tuple, optional
        A tuple of the form (lower_bounds, upper_bounds) where lower_bounds and upper_bounds are (4,) arrays of q, lon, lat, alt.
        Default is None (-inf to inf for q, lon, lat; lowest station height to inf for alt). Specifying None for either the upper or lower bound for
        a parameter will set that bound to its default. If any of the three geographic coordinates are bounded, bounds for the other two geographic coordinates
        are required.
    return_residual : bool, optional
        If True, return the residuals of the least squares optimization. Default is False.
    
    Returns
    -------
    retrieved_lon : float
        The retrieved longitude of the stroke.
    retrieved_lat : float
        The retrieved latitude of the stroke.
    retrieved_alt : float
        The retrieved altitude of the stroke.
    retrieved_q : float
        The retrieved charge of the stroke.
    residuals : np.ndarray, optional
        A 1D array of shape (N_stations,) containing the residuals of the least squares optimization. Only returned if return_residual is True.
    """
    from pyxlma import coords
    station_lon, station_lat, station_alt = station_df['lon'].values, station_df['lat'].values, station_df['alt'].values
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=station_lat.mean(), ctrLon=station_lon.mean(), ctrAlt=station_alt.min())
    geosys = coords.GeographicSystem()
    station_ECEF = geosys.toECEF(station_lon, station_lat, station_alt)
    station_xyz = np.array(tpcs.fromECEF(*station_ECEF))
    bounds_qxyz = None
    station_largest_ob_idx = np.argmax(np.abs(stroke_obs))
    initial_guess = np.array([1.0, *station_xyz[:, station_largest_ob_idx]]) + [0, 0, 0, 1000] # initial guess 1C, 1km above the station with the largest observation
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        q_bounds = (lower_bounds[0], upper_bounds[0]) if lower_bounds[0] is not None and upper_bounds[0] is not None else (-np.inf, np.inf)
        lon_bounds = (lower_bounds[1], upper_bounds[1])
        lat_bounds = (lower_bounds[2], upper_bounds[2])
        alt_bounds = (lower_bounds[3], upper_bounds[3]) 
        if np.any(np.array([lon_bounds, lat_bounds, alt_bounds]) == None):
            x_bounds = (-np.inf, np.inf)
            y_bounds = (-np.inf, np.inf)
            z_bounds = (station_xyz[2].min(), np.inf)
        else:
            corners_lla = np.meshgrid([lon_bounds[0], lon_bounds[1]], [lat_bounds[0], lat_bounds[1]], [alt_bounds[0], alt_bounds[1]], indexing='ij')
            corners_lla = np.array(corners_lla).reshape(3, -1)
            corners_ecef = geosys.toECEF(*corners_lla)
            corners_xyz = np.array(tpcs.fromECEF(*corners_ecef))
            x_bounds = (corners_xyz[0].min(), corners_xyz[0].max())
            y_bounds = (corners_xyz[1].min(), corners_xyz[1].max())
            z_bounds = (corners_xyz[2].min(), corners_xyz[2].max())
            initial_guess[1:] = corners_xyz.mean(axis=1) # center of the bounding box
        if np.all(np.isfinite(q_bounds)):
            initial_guess[0] = np.mean(q_bounds)
        bounds_qxyz = tuple(np.array([q_bounds, x_bounds, y_bounds, z_bounds], dtype=float).T)
        if np.all(bounds_qxyz[0][1:] == bounds_qxyz[1][1:]):
            # if the x, y, z bounds are all equal, then analytically solve for the charge
            retrieved_q = monopole_q_analytic(*initial_guess[1:], stroke_obs, station_xyz)
            if return_residual:
                residuals = monopole_delta_E_error([retrieved_q, *initial_guess[1:]], stroke_obs, station_xyz)
                return lon_bounds[0], lat_bounds[0], alt_bounds[0], retrieved_q, residuals
            return lon_bounds[0], lat_bounds[0], alt_bounds[0], retrieved_q
    retrieved_opt = monopole_charge_retrieval(*initial_guess, stroke_obs, station_xyz, bounds=bounds_qxyz, return_residual=return_residual)
    retrieved_q = retrieved_opt[0]
    retrieved_xyz = retrieved_opt[1:4]
    retrieved_lon, retrieved_lat, retrieved_alt = geosys.fromECEF(*tpcs.toECEF(retrieved_xyz[0], retrieved_xyz[1], retrieved_xyz[2]))
    if return_residual:
        residuals = retrieved_opt[4]
        return retrieved_lon, retrieved_lat, retrieved_alt, retrieved_q, residuals
    return retrieved_lon, retrieved_lat, retrieved_alt, retrieved_q


def multi_monopole_retrieval_dask(all_monopoles, station_df, initial_guess_q=1, chunk_size=50):
    """
    Retrieve the location and charge removed by multiple monopole discharge strokes.

    
    Parameters
    ----------
    all_monopoles : np.ndarray
        A 2D array of shape (N_strokes, N_stations) containing the observed electric field changes at each station for each stroke.
    station_df : pd.DataFrame
        A DataFrame containing the station information, including 'x' and 'y' and 'z' columns for the station coordinates.
    initial_guess_q : float, optional
        An optional initial guess for the charge of each stroke. Default is 1.

    Returns
    -------
    retrieved_x : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved x coordinates of each stroke.
    retrieved_y : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved y coordinates of each stroke.
    retrieved_z : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved z coordinates of each stroke.
    retrieved_q : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved charges of each stroke.
    """
    import dask.array as da
    this_charge_obs = da.from_array(all_monopoles, chunks=(chunk_size, -1)) # observations for all charges, shape (N_strokes, N_stations)
    station_largest_ob_idx = np.argmax(np.abs(all_monopoles), axis=1)
    initial_guess = da.from_array(station_df[['x', 'y', 'z']].to_numpy()[station_largest_ob_idx], chunks=(chunk_size, -1)) # station x, y, z for each initial guess, shape (N_strokes, 3)
    station_xyz = np.array([station_df['x'].values, station_df['y'].values, station_df['z'].values]) # shape (3, N_stations)
    # retrieve the charge parameters for every stroke in parallel using the charge retrieval function
    all_retrievals = da.apply_gufunc(partial(monopole_charge_retrieval, station_xyz=station_xyz), '(),(),(),(),(n)->(p)',
                                        initial_guess_q, # initial_guess_q
                                        initial_guess[:, 0], # initial_guess_x
                                        initial_guess[:, 1], # initial_guess_y
                                        initial_guess[:, 2] + 1000, # initial_guess_z, 1km above station location
                                        this_charge_obs, # stroke_obs
                                        vectorize=True, output_dtypes=float, output_sizes={'p': 4}).persist()
    retrieved_x = all_retrievals[:, 1]
    retrieved_y = all_retrievals[:, 2]
    retrieved_z = all_retrievals[:, 3]
    retrieved_q = all_retrievals[:, 0]
    return retrieved_x, retrieved_y, retrieved_z, retrieved_q

def dipole_E_change(x, y, z, q, xi, yi, zi, dr):
    """
    Calculate the change in electric field due to a dipole charge at a given position.

    Parameters
    ----------
    x, y, z, q : array-like
        The x, y, z coordinates of the charge (stroke) positions. Shape should be (N_strokes,).
    xi, yi, zi : array-like
        The x, y, z coordinates of the station positions. Shape should be (N_stations,).
    dr : array-like
        The dipole moment vector components (dr_x, dr_y, dr_z). Shape should be (3, N_strokes).

    Returns
    -------
    delta_E : ndarray
        The change in electric field at each station due to each charge. Shape will be (N_strokes, N_stations).
    """
    # reshape so that all 'stroke' parameters are on axis 0
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    q = np.array(q).reshape(-1, 1)
    drx = dr[0, :].reshape(-1, 1)
    dry = dr[1, :].reshape(-1, 1)
    drz = dr[2, :].reshape(-1, 1)
    # reshape so that all 'station' parameters are on axis 1
    xi = np.array(xi).reshape(1, -1)
    yi = np.array(yi).reshape(1, -1)
    zi = np.array(zi).reshape(1, -1)
    # compute r vector from stroke to station
    dx = x - xi
    dy = y - yi
    dz = z - zi
    # compute charge moment vector, p = dr * q
    p = np.array((drx, dry, drz)) * q
    r = np.array((dx, dy, dz))
    r_mag = np.linalg.norm(r, axis=0)
    # dot product of r and p, r_dot_p = r_x*p_x + r_y*p_y + r_z*p_z
    r_dot_p = np.sum(r*p, axis=0)
    # calculate E-field change using dipole formula, \frac{1}{4\pi\epsilon_0} (\frac{2p_z}{r^3} - \frac{6z}{r^5} (r \cdot p))
    k_const = 1/(4*np.pi*EPSILON_0)
    delta_E = k_const * ((2*p[2, :]/(r_mag**3)) - (6*dz)/(r_mag**5) * r_dot_p)
    return delta_E # V/m
