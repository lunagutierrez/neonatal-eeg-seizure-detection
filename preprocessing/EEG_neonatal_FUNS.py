
import numpy as np 
import pandas as pd
import h5py
import os

from scipy.signal import lfilter
import mne
import matplotlib.pyplot as plt

def select_seizure_chunks(data, f, k): # Info annot, sampling freq, patient_index
    """
    Identify seizure periods for a specific patient and convert to sample indices (annotation_matrix, freq after downsampling, patient_index)
    Returns: dict {seizures: df start/end of e/ seizure in s and samples, total_seizures : int}
    """

    # Identify all seconds where the annotation equals 1 for patient k
    sec_ones = np.where(data.iloc[:, k - 1] == 1)[0] + 1 # (k-1 because DataFrame is 0-indexed and for patient k i need the k-1 col, +1 to convert to 1-based seconds)

    if len(sec_ones) == 0:  # If no seizure seconds exist for this patient
        return {"seizures": None, "total_seizures": 0} # Empty result

    # find continuous segments
    breaks = np.where(np.diff(sec_ones) != 1)[0] # Find indices where the seizure annotation is NOT consecutive
    segments = np.split(sec_ones, breaks + 1) # Split seizure sec into continuous segments

    records = []

    for seg in segments: # Process each continuous seizure segment
        from_sec = seg[0]
        to_sec = seg[-1]
        duration = to_sec - from_sec + 1

        # Convert second-based indices to sample-based indices
        from_sample = (from_sec - 1) * f + 1
        to_sample = (from_sec - 1) * f + duration * f

        records.append([
            k, duration, from_sec, to_sec, from_sample, to_sample
        ])

    seizures_df = pd.DataFrame(
        records,
        columns=[
            "patient", "seizure_duration",
            "from_sec", "to_sec",
            "from_sample", "to_sample"
        ]
    )

    return {
        "seizures": seizures_df,
        "total_seizures": len(records)
    }


def generate_montage(matrix, labels): # EEG data matrix, channel labels
    """
    Bipolar EEG montage construction by subtracting predefined electrode pairs.
    """

    # Predef bipolar electrode pairs
    pairs = [
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("Fz", "Cz"), ("Cz", "Pz")
    ]

    montage_data = []

    for a, b in pairs:
        # Find channel indices corresponding to electrode names
        idx_a = next(i for i, l in enumerate(labels) if a in l) #i,l = index, label (str) next() finds the first one
        idx_b = next(i for i, l in enumerate(labels) if b in l)
        montage_data.append(matrix[:, idx_a] - matrix[:, idx_b]) # Subtract signals

    montage_data = np.vstack(montage_data).T # Stack and transpose to get (samples x channels)

    df = pd.DataFrame(
        montage_data,
        columns=[f"{a}-{b}" for a, b in pairs]
    )

    return df


def generate_samples(
    which_expert,
    annotations_file,
    seizure_IDs,
    non_seizure_IDs,
    window,
    chunks,
    down_sampling_factor=4,
    preprocessing=False,
    base_dir="",
    random=True,
    write_hdf5=True
):
    """
    1. Loads EEG recordings and annotations
    2. Applies downsampling and bipolar montage
    3. Extracts seizure and non-seizure windows
    4. Balances the dataset
    5. Saves the final dataset to HDF5
    """

    # Load annotations
    ann = pd.read_csv(
        os.path.join(base_dir, "annotations", annotations_file)
    )

    # Read first EDF to get metadata
    raw = mne.io.read_raw_edf(
        os.path.join(base_dir, "edf", "eeg1.edf"),
        preload=True,
        verbose=False
    )

    fs = int(raw.info["sfreq"] / down_sampling_factor) # Effective sampling frequency after downsampling
    channel_names = raw.ch_names

    SEIZURE = []
    NON_SEIZURE = []

    # ----------------
    # Seizure patients
    # ----------------
    S = 0 # # of seizure windows

    for pid in seizure_IDs:
        # Load EEG for seizure patient
        raw = mne.io.read_raw_edf(
            os.path.join(base_dir, "edf", f"eeg{pid}.edf"),
            preload=True,
            verbose=False
        )

        data = raw.get_data().T # Transpose to (samples × channels)
        data = data[::down_sampling_factor, :] # Downsample signal

        montage_df = generate_montage(data, channel_names) # Apply bipolar montage
        montage_df *= 1e6  # Convert from V to microV

        out = select_seizure_chunks(ann, fs, pid) # Identify seizure intervals

        if out["total_seizures"] == 0: # Skip patients with no seizures
            continue

        for _, row in out["seizures"].iterrows(): # Extract windowed seizure chunks
            r = row["seizure_duration"] // window # # of full windows in this seizure
            r = min(r, chunks) # Limit to  max chunks

            for k in range(r): # Sample indices for this window
                start = int(row["from_sample"] + k * fs * window)
                end = int(start + fs * window)

                chunk = montage_df.iloc[start:end].copy() # Extract EEG window
                chunk["seizure"] = 1 # Label as seizure
                SEIZURE.append(chunk)

                S += 1

    # --------------------
    # Non-seizure patients
    # --------------------
    nn = int(np.ceil(S / len(non_seizure_IDs))) # # of non-seizure windows per patient (to balance classes)

    for pid in non_seizure_IDs:
        raw = mne.io.read_raw_edf(
            os.path.join(base_dir, "edf", f"eeg{pid}.edf"),
            preload=True,
            verbose=False
        )

        data = raw.get_data().T
        data = data[::down_sampling_factor, :]

        montage_df = generate_montage(data, channel_names)
        montage_df *= 1e6  # Convert from V to mV

        max_start = len(montage_df) - fs * window # Max valid starting index for a full window

        for _ in range(nn): # Randomly sample non-seizure window
            r = np.random.randint(0, max_start)
            chunk = montage_df.iloc[r:r + fs * window].copy()
            chunk["seizure"] = 0
            NON_SEIZURE.append(chunk)

    FINAL = pd.concat(SEIZURE + NON_SEIZURE, ignore_index=True)

    # ---------
    # Save HDF5
    # ---------
    if write_hdf5:
        out_file = os.path.join(
            base_dir,
            "working",
            "inputs",
            f"expert_{which_expert}_{window}sec_{chunks}chunk_{fs}Hz.hdf5"
        )

        with h5py.File(out_file, "w") as f:
            f.create_dataset("FINAL.mtx", data=FINAL.values)

    return {
        "SEIZURE": SEIZURE,
        "NON_SEIZURE": NON_SEIZURE,
        "FINAL": FINAL
    }