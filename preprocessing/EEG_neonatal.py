from EEG_neonatal_FUNS import generate_samples
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

EXPERTS = ["A", "B", "C"]

ANNOTATION_FILES = [
    "annotations_2017_A_fixed.csv",
    "annotations_2017_B.csv",
    "annotations_2017_C.csv"
]

# Infant IDs
SEIZURE_IDS = [
    1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22,
    25, 31, 34, 36, 38, 39, 40, 41, 44, 47, 50, 51, 52,
    62, 63, 66, 67, 69, 71, 73, 75, 76, 77, 78, 79
]

NON_SEIZURE_IDS = [
    3, 10, 18, 27, 28, 29, 30, 32, 35, 37, 42, 45, 48,
    49, 53, 55, 57, 58, 59, 60, 70, 72
]

# Parameters
WINDOWS = [1, 2, 5]
CHUNKS = [1, 2, 5, 10, 20]
MAX_CHUNKS = 20

DOWNSAMPLING = 4 # Down-sampling factor (from 256Hz to 64Hz)

def timestamp():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

timestamp()

for expert, ann_file in zip(EXPERTS, ANNOTATION_FILES):

    for window in WINDOWS:
        for chunks in CHUNKS:
            generate_samples(
                which_expert=expert,
                annotations_file=ann_file,
                seizure_IDs=SEIZURE_IDS,
                non_seizure_IDs=NON_SEIZURE_IDS,
                window=window, # in seconds
                chunks=chunks, # # chunks/window
                down_sampling_factor=DOWNSAMPLING,
                preprocessing=False,
                base_dir=BASE_DIR,
                random=False, # Random sampling
                write_hdf5=True # Save to HDF5 file
            )

timestamp()
