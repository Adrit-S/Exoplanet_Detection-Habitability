import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from scipy.interpolate import interp1d

# Define Dataset Directory
DATA_DIRECTORY = "D:\Exoplanet_Dataset"

class LightCurveDataset(Dataset):
    def __init__(self, directory, max_length=2000):
        self.directory = directory

        # Checks if file is a fits file before using

        self.files = [files for files in os.listdir(directory) if files.endswith(".fits")]
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, n):
        # Function to retrieve data from NASA Kepler bulk dataset script (Confirmed Exoplanet Detections)

        file_path = os.path.join(self.directory, self.files[n])

        # Opens FITS File and uses PDCSAP_FLUX for semi-processed data

        with fits.open(file_path) as file:
            time = file[1].data['TIME']
            flux = file[1].data['PDCSAP_FLUX']

        # Further process data by removing NaN values in flux

        mask = ~np.isnan(flux)
        time, flux = time[mask], flux[mask]

        # Normalize flux using mean and standard deviation

        flux = (flux - np.mean(flux)) / np.std(flux)

        # Center's Exoplanet Transit Dip for truncation or padding 

        flux, time = self.center_transit(flux, time, self.max_length)

        # Further process data by Truncating or Padding to fixed length 

        flux = self.resample_lightcurve(time, flux, self.max_length)

        # Convert to a tensor for CNN usability

        flux_tensor = torch.tensor(flux, dtype=torch.float32)

        return flux_tensor

    def center_transit(self, flux, time, max_length):
        #Centers the Exoplanet Transit Dip
        min_index = np.argmin(flux)
        half_length = max_length // 2
        start = max(0, min_index - half_length)
        end = start + max_length

        if end > len(flux):  
            start = max(0, len(flux) - max_length)
            end = len(flux)

        return flux[start:end], time[start:end]

    def resample_lightcurve(self, time, flux, target_length):
        #Truncates or pads data to a fixed length for CNN
        if len(flux) == target_length:
            return flux  
        
        interp_func = interp1d(np.linspace(0, 1, len(flux)), flux, kind='linear')
        return interp_func(np.linspace(0, 1, target_length))
