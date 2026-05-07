import torch
import numpy as np
import matplotlib.pyplot as plt
from LightCurveDataset import LightCurveDataset, DATA_DIRECTORY
from astropy.io import fits
from scipy.interpolate import interp1d
import os

# Function to preprocess the light curve (center the transit, normalize, resample, and truncate/pad)
def preprocess_lightcurve(flux, time, max_length=2000):

    # Center the light curve around the transit (assumes transit happens at time = 0)
    min_index = np.argmin(flux)  # Find the dip (minimum flux, representing the transit)
    half_length = max_length // 2

    # Center the transit
    start = min_index - half_length
    end = min_index + half_length

    if start < 0:
        start = 0
        end = min(max_length, len(flux))

    if end > len(flux):
        end = len(flux)
        start = max(0, end - max_length)

    flux = flux[start:end]
    time = time[start:end]

    # Normalize the flux
    flux = (flux - np.mean(flux)) / np.std(flux)

    # Resample the flux to the target length (max_length)
    resampled_time = np.linspace(np.min(time), np.max(time), max_length)
    interp_flux = interp1d(time, flux, kind='linear', fill_value="extrapolate")
    resampled_flux = interp_flux(resampled_time)

    return resampled_flux, resampled_time

# Function to apply data augmentation (shifts and noise)
def augment_data(time, flux, shift_range=0.05, flux_noise_range=0.02, time_noise_range=0.01):

    # Random time shift (slightly shift the x-axis)
    time_shift = np.random.uniform(-shift_range, shift_range) * np.ptp(time)  # Random shift within a range
    time = time + time_shift

    # Random flux shift and noise (slightly shift and add noise to the y-axis)
    flux_shift = np.random.uniform(-flux_noise_range, flux_noise_range)
    flux_noise = np.random.normal(0, flux_noise_range, size=flux.shape)
    flux = flux + flux_shift + flux_noise

    return time, flux

def plot_lightcurve():
    # Load dataset
    dataset = LightCurveDataset(DATA_DIRECTORY)
    
    # Get the files for the specific exoplanet "kplr006937402"
    exoplanet_name = "kplr006937402"
    
    if exoplanet_name not in dataset.planet_files:
        print(f"Error: Exoplanet {exoplanet_name} not found in the dataset.")
        return
    
    files = dataset.planet_files[exoplanet_name]

    # Load the time and flux data from one of the FITS files for the selected exoplanet
    # (assuming the exoplanet has at least one associated FITS file)
    file_path = os.path.join(DATA_DIRECTORY, files[0])
    with fits.open(file_path) as hdul:
        time = hdul[1].data['TIME']
        flux = hdul[1].data['PDCSAP_FLUX']

    # Remove NaN values from flux and time arrays
    valid_mask = ~np.isnan(flux)  # Mask of valid (non-NaN) values
    time = time[valid_mask]
    flux = flux[valid_mask]

    # Preprocess the light curve (center, normalize, resample, and truncate/pad)
    resampled_flux, resampled_time = preprocess_lightcurve(flux, time)

    # Apply data augmentation
    augmented_time, augmented_flux = augment_data(resampled_time, resampled_flux)

    # Plot the light curve data with augmentation
    plt.figure(figsize=(10, 6))
    plt.scatter(augmented_time, augmented_flux, color='blue', label='Augmented Light Curve Data', s=10)
    
    # Customize plot
    plt.title(f"Augmented Exoplanet Light Curve for {exoplanet_name}")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_lightcurve()
