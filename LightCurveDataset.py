import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from scipy.interpolate import interp1d
from collections import defaultdict

# Define Dataset Directory
DATA_DIRECTORY = os.path.abspath(r"D:\Exoplanet Dataset")

class LightCurveDataset(Dataset):
    def __init__(self, directory, max_length=2000):
        self.directory = directory
        self.max_length = max_length

        # Organizes files by category: Confirmed Exoplanets and False Positives
        self.planet_files = defaultdict(list)
        self.false_positive_planets = {}  # This will map false exoplanet IDs to their FITS files
        self.labels = {}  # Dictionary to store labels (1=exoplanet, 0=false positive)

        # Organize confirmed exoplanets and false positives
        for filename in os.listdir(directory):
            if filename.endswith(".fits") and "llc" in filename.lower():
                if filename.startswith("FALSE"):
                    false_planet_id = filename.split("-")[0]  # Extract false exoplanet name (e.g., FALSE_kplr001026032)
                    self.false_positive_planets.setdefault(false_planet_id, []).append(filename)
                    self.labels[filename] = 0  # Label as False Positive
                else:
                   planet_name = filename.split("-")[0]  # Extract planet/system name
                   self.labels[filename] = 1  # Label as Confirmed Exoplanet
                   self.planet_files[planet_name].append(filename)

        # Store planet names for confirmed exoplanets
        self.planet_names = list(self.planet_files.keys())

        # Limit to first 500 unique false exoplanets
        self.false_positive_planets = dict(list(self.false_positive_planets.items())[:500])  # Keep first 500 false exoplanets
        self.false_positive_names = list(self.false_positive_planets.keys())  # Extract just the planet names

        # Generate additional false positives if needed to balance dataset
        while len(self.false_positive_names) < len(self.planet_names):
            self.false_positive_names.append(None)  # Mark that we need to generate a new one

        # Combine real exoplanets and false positives into one list
        self.planet_names += self.false_positive_names

    def __len__(self):
        return len(self.planet_names)

    def __getitem__(self, n):
        num_exoplanets = len(self.planet_files)  # Number of real exoplanets (label 1)
        num_real_false_positives = len(self.false_positive_names)  # Number of real false positives (label 0)

        # If `n` is less than `num_exoplanets`, we handle exoplanets (label 1)
        if n < num_exoplanets:
            planet_name = self.planet_names[n]
            flux_tensor, label = self.load_real_lightcurve(planet_name), 1  # Label as exoplanet
            print(f"Batching real exoplanet: {planet_name}")

        # Otherwise, handle false positives (label 0)
        else:
            false_pos_index = n - num_exoplanets  # Adjust index to false positives

            # Case 1: Use real false positives first
            if false_pos_index < num_real_false_positives:
                planet_name = self.false_positive_names[false_pos_index]  # Get the false exoplanet name
                if planet_name is None:
                    flux_tensor, label = self.generate_false_positive_lightcurve(), 0
                    print(f"Batching synthetic false positive (generated): Index {n}")
                else:
                    flux_tensor, label = self.load_real_lightcurve(planet_name), 0
                    print(f"Batching real false positive: {planet_name}")

            # Case 2: Generate synthetic false positives only if needed
            else:
                flux_tensor, label = self.generate_false_positive_lightcurve(), 0  # Label as false positive
                print(f"Batching synthetic false positive (generated): Index {n}")

        return torch.tensor(flux_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long).squeeze()





    def load_real_lightcurve(self, planet_name):
        files = self.planet_files[planet_name]
        all_time, all_flux = [], []
        label = None

        # Opens FITS Files and uses PDCSAP_FLUX for semi-processed data
        for file in files:
            file_path = os.path.join(self.directory, file)
            with fits.open(file_path) as hdul:
                time = hdul[1].data['TIME']
                flux = hdul[1].data['PDCSAP_FLUX']

            # Remove NaN values
            mask = ~np.isnan(flux)
            time, flux = time[mask], flux[mask]

            all_time.append(time)
            all_flux.append(flux)

            if label is None:
                label = self.labels[file]  # Set label (should be 1 for exoplanet)

        # Flatten merged data
        if not all_time or not all_flux:
            print(f"Warning: No valid data for {planet_name}")
            return torch.zeros(self.max_length, dtype=torch.float32), torch.tensor(1, dtype=torch.long)  # Return zero tensor with label 1

        all_time = np.concatenate(all_time)
        all_flux = np.concatenate(all_flux)


        # Sorts by time to ensure chronological order
        sorted_indices = np.argsort(all_time)
        all_time = all_time[sorted_indices]
        all_flux = all_flux[sorted_indices]

        # Normalize flux using mean and standard deviation
        all_flux = (all_flux - np.mean(all_flux)) / np.std(all_flux)

        # Centers the transit dip in the middle of the light curve
        all_flux, all_time = self.center_transit(all_flux, all_time, self.max_length)

        # Further process data by truncating or padding to fixed length
        all_flux = self.resample_lightcurve(all_time, all_flux, self.max_length)

        # Convert to a tensor for CNN usability
        flux_tensor = torch.tensor(all_flux, dtype=torch.float32)  # Ensure it stays 1D

        return flux_tensor, torch.tensor(label, dtype=torch.long)  # Ensure label is long for classification

    def generate_false_positive_lightcurve(self):
        length = self.max_length

        # Generates random stellar noise to simulate instrument variability
        noise = np.random.normal(0, 0.1, length)

        # Adds sinusoidal variations to mimic stellar activity (random period)
        time = np.linspace(0, 10, length)
        variability = 0.2 * np.sin(2 * np.pi * time / np.random.uniform(1, 5))

        # Introduces a fake transit dip (but not an actual exoplanet transit)
        dip_center = np.random.randint(int(0.3 * length), int(0.7 * length))  # Random transit location
        dip_width = np.random.randint(50, 75)  # Transit Length
        dip_depth = np.random.uniform(0.05, 0.1)  # Transit Depth

        artificial_flux = 1 + variability + noise  # Baseline normalized at 1
        artificial_flux[dip_center - dip_width // 2: dip_center + dip_width // 2] -= dip_depth  # Fake dip

        # Normalize to ensure consistent scaling with real light curves
        artificial_flux = (artificial_flux - np.mean(artificial_flux)) / np.std(artificial_flux)

        return torch.tensor(artificial_flux, dtype=torch.float32)  # Ensure it stays 1D



    def center_transit(self, flux, time, max_length):
        min_index = np.argmin(flux)  
        half_length = max_length // 2

        start = min_index - half_length
        end = min_index + half_length

        if start < 0:  
            start = 0
            end = min(max_length, len(flux))

        if end > len(flux):  
            end = len(flux)
            start = max(0, end - max_length)

        return flux[start:end], time[start:end]

    def resample_lightcurve(self, time, flux, target_length):
        if len(flux) == target_length:
            return flux  

        interp_func = interp1d(np.linspace(0, 1, len(flux)), flux, kind='linear')
        return interp_func(np.linspace(0, 1, target_length))
