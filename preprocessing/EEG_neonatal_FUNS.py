
import numpy as np 
import pandas as pd
import h5py
import os

from scipy.signal import lfilter
import mne
import matplotlib.pyplot as plt

def select_seizure_chunks(data, f, k):
    """
    Parameters
    ----------
    data : pandas.DataFrame
        Annotation matrix (seconds × patients)
    f : int
        Sampling frequency
    k : int
        Patient index (1-based, as in original R code)

    Returns
    -------
    dict with:
        seizures : DataFrame
        total_seizures : int
    """

    sec_ones = np.where(data.iloc[:, k - 1] == 1)[0] + 1

    if len(sec_ones) == 0:
        return {"seizures": None, "total_seizures": 0}

    # find continuous segments
    breaks = np.where(np.diff(sec_ones) != 1)[0]
    segments = np.split(sec_ones, breaks + 1)

    records = []

    for seg in segments:
        from_sec = seg[0]
        to_sec = seg[-1]
        duration = to_sec - from_sec + 1

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


def generate_montage(matrix, labels):
    """
    Bipolar montage construction (same as R version)
    """

    pairs = [
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("Fz", "Cz"), ("Cz", "Pz")
    ]

    montage_data = []

    for a, b in pairs:
        idx_a = next(i for i, l in enumerate(labels) if a in l)
        idx_b = next(i for i, l in enumerate(labels) if b in l)
        montage_data.append(matrix[:, idx_a] - matrix[:, idx_b])

    montage_data = np.vstack(montage_data).T

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
    Python equivalent of generate_samples() from R
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

    fs = int(raw.info["sfreq"] / down_sampling_factor)
    channel_names = raw.ch_names

    SEIZURE = []
    NON_SEIZURE = []

    # ---------------------------
    # Seizure patients
    # ---------------------------
    S = 0

    for pid in seizure_IDs:
        raw = mne.io.read_raw_edf(
            os.path.join(base_dir, "edf", f"eeg{pid}.edf"),
            preload=True,
            verbose=False
        )

        data = raw.get_data().T
        data = data[::down_sampling_factor, :]

        montage_df = generate_montage(data, channel_names)

        out = select_seizure_chunks(ann, fs, pid)

        if out["total_seizures"] == 0:
            continue

        for _, row in out["seizures"].iterrows():
            r = row["seizure_duration"] // window
            r = min(r, chunks)

            for k in range(r):
                start = int(row["from_sample"] + k * fs * window)
                end = int(start + fs * window)

                chunk = montage_df.iloc[start:end].copy()
                chunk["seizure"] = 1
                SEIZURE.append(chunk)

                S += 1

    # ---------------------------
    # Non-seizure patients
    # ---------------------------
    nn = int(np.ceil(S / len(non_seizure_IDs)))

    for pid in non_seizure_IDs:
        raw = mne.io.read_raw_edf(
            os.path.join(base_dir, "edf", f"eeg{pid}.edf"),
            preload=True,
            verbose=False
        )

        data = raw.get_data().T
        data = data[::down_sampling_factor, :]

        montage_df = generate_montage(data, channel_names)

        max_start = len(montage_df) - fs * window

        for _ in range(nn):
            r = np.random.randint(0, max_start)
            chunk = montage_df.iloc[r:r + fs * window].copy()
            chunk["seizure"] = 0
            NON_SEIZURE.append(chunk)

    FINAL = pd.concat(SEIZURE + NON_SEIZURE, ignore_index=True)

    # ---------------------------
    # Save HDF5
        # ---------------------------
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