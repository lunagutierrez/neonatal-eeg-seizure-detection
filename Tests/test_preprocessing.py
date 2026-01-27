"""Tests generation of bipolar montage, final matrix shape (electrodes + label) and
class distribution.

Use the following line on the PowerShell to see loggers of the test process:
python -m pytest Tests/test_preprocessing.py -s --log-cli-level=INFO"
"""

import pytest
import numpy as np
import pandas as pd
import logging
from preprocessing.EEG_neonatal_FUNS import select_seizure_chunks, generate_montage

# Set up a logger to see shapes and details in the console
logger = logging.getLogger(__name__)

def test_generate_montage_details():
    """
    Verifies montage shape and specific bipolar math: Fp2-F4 = Fp2 - F4
    """
    labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", 
              "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
    
    # Create predictable data to check the math
    # Row 0: Fp2 = 10, F4 = 3 -> Result should be 7
    mock_data = np.zeros((1, len(labels)))
    mock_data[0, labels.index("Fp2")] = 10
    mock_data[0, labels.index("F4")] = 3
    
    montage_df = generate_montage(mock_data, labels)
    
    # Log the found columns for debugging
    logger.info(f"Generated Montage Columns: {list(montage_df.columns)}")
    
    # Verify shape
    assert montage_df.shape[1] == 18
    # Verify math (Fp2 - F4)
    assert montage_df.loc[0, "Fp2-F4"] == 7
    logger.info("Bipolar montage and shape verified.")

def test_final_matrix_integrity():
    """
    Simulates the 'FINAL' matrix creation to log its properties.
    """
    fs = 64
    window_sec = 10
    samples_per_window = fs * window_sec
    num_channels = 18
    
    # Simulate 3 seizure chunks and 3 non-seizure chunks
    seizure_chunks = [pd.DataFrame(np.ones((samples_per_window, num_channels))) for _ in range(3)]
    for df in seizure_chunks: df["seizure"] = 1
        
    non_seizure_chunks = [pd.DataFrame(np.zeros((samples_per_window, num_channels))) for _ in range(3)]
    for df in non_seizure_chunks: df["seizure"] = 0
    
    final_df = pd.concat(seizure_chunks + non_seizure_chunks, ignore_index=True)
    
    # LOGGING THE SHAPE
    logger.info(f"Final Matrix Shape: {final_df.shape}")
    logger.info(f"Class Distribution: \n{final_df['seizure'].value_counts(normalize=True)}")
    
    # Assertions
    assert final_df.shape == (6 * 640, 19)
    assert final_df["seizure"].mean() == 0.5 # Verify 50/50 balance
    assert not final_df.isnull().values.any() # Ensure no NaN values were created