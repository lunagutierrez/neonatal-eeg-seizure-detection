"""
Data loading utilities for the Neonatal EEG project.
This file handles reading HDF5 data files, constructing paths to results.
"""

import h5py # Library to read and write HDF5 files (hierarchical data format)
import numpy as np
from pathlib import Path
from  config import INPUT_DIR

def make_data_filename(which_expert: str, window: int, chunks: int, freq: int) -> str:
    """Constructs the HDF5 filename based on experiment parameters."""
    
    return f"expert_{which_expert}_{window}sec_{chunks}chunk_{freq}Hz.hdf5"

def load_hdf5_data(file_name: str) -> np.ndarray:
    """Reads raw EEG data from HDF5 and ensures correct orientation."""

    data_path = INPUT_DIR / file_name
    if not data_path.exists():
        raise FileNotFoundError(f"EEG file not found: {data_path}")

    with h5py.File(data_path, "r") as file_h5:
        temp = np.array(file_h5["FINAL_mtx"])
    
    # Ensure (Samples, Channels) orientation
    # Some datasets might have shape (Channels, Samples), so we transpose if needed
    return temp.T if temp.shape[0] < temp.shape[1] else temp 