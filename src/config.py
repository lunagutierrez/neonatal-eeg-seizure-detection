from pathlib import Path

# Paths for folders
PROJECT_ROOT = Path(__file__).parent.parent
WORKING_DIR = PROJECT_ROOT / "working"
OUTPUT_DIR = WORKING_DIR / "outputs"
INPUT_DIR = WORKING_DIR / "inputs"

# Paper parameters
EXPERTS = ["A", "B", "C"]
WINDOWS = [1]
CHUNKS = [1]
#WINDOWS = [1, 2, 5]
#CHUNKS = [1, 2, 5, 10, 20]

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
KFOLDS = 5
SEED = 1024

# Data parameters
N_CLASSES = 2

NO_OF_EEG_CHANNELS = 18
SEIZURE_INDICATOR = 20
FREQS = 64
COMPLETE_CALCULATIONS = False