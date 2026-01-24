from pathlib import Path
# Paths for folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#__file__ is the path to the current script (this file)
# .resolve() gives the absolute path, resolving any symbolic links
# .parents[1] goes two levels up from the current file's location
WORKING_DIR = PROJECT_ROOT / "working"
OUTPUT_DIR = WORKING_DIR / "outputs"
INPUT_DIR = WORKING_DIR / "inputs"

# Paper parameters
EXPERTS = ["A", "C"]
WINDOWS = [1,2,5]
CHUNKS = [1, 2, 5, 10, 20]

# Training parameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.1
KFOLDS = 5
SEED = 1024

# Data parameters
N_CLASSES = 2

NO_OF_EEG_CHANNELS = 18
SEIZURE_INDICATOR = 20
FREQS = 64
COMPLETE_CALCULATIONS = False

