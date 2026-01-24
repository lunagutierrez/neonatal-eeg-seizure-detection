
# Neonatal EEG Seizure Detection Project

This project implements a pipeline to preprocess neonatal EEG data, segment it into windows, and train a convolutional neural network (CNN) to classify seizure vs. non-seizure segments. The pipeline supports k-fold cross-validation, per-expert processing, and logging of metrics per epoch. It corresponds to the **code implementation** of the paper:

## [Gramacki, A., Gramacki, J. A deep learning framework for epileptic seizure detection based on neonatal EEG signals. Sci Rep 12, 13010 (2022)](https://www.nature.com/articles/s41598-022-15830-2#Sec15)  
**By:** Karla Arzate & Luna Gutiérrez

---

The dataset is available at: [Zenodo Dataset](https://zenodo.org/records/4940267) includes:  
- Raw EEG files in **EDF format**  
- Annotation files in **CSV format**  

This repository does not contain the edf files' folder due to their size. 

## Directory Structure


```
BASE_DIR/
├── edf/
│   ├── eeg1.edf
│   ├── eeg2.edf
│   └── ...
├── annotations/
│   ├── annotations_2017_A_fixed.csv
│   ├── annotations_2017_B.csv
│   └── annotations_2017_C.csv
├── working/                  # Inputs and outputs for experiments
│   ├── inputs/ # Resulting preprocessed HDF5 files
│   │   ├── results_expert_A/
│   │   ├── results_expert_B/
│   │   └── results_expert_C/
│   └── outputs/ # Experiment results
│       ├── A/
│       ├── B/
│       └── C/
├── preprocessing/
│   ├── EEG_neonatal_FUNS.py
│   ├── EEG_neonatal.py # Main data creation runner
├── src/                      # Core Python code
│   ├── main_mult.py          # Main experiment runner
│   ├── config.py             # Configuration (paths, parameters, experts)
│   ├── logger.py             # Logging utility
│   ├── loaders.py            # Data loading functions
│   ├── cnn.py                # CNN architecture definition
│   ├── engine.py             # Training / K-fold evaluation
│   ├── processor.py          # Preprocessing & windowing functions
│   ├── evaluation.py         # Model evaluation metrics
```

> Note: `working/inputs` **must exist** before writing HDF5 files, or the code will automatically create it.

---

## **Folder contents**

* **edf/**: raw EEG files per patient (e.g., `eeg1.edf`, `eeg2.edf`, …).
* **annotations/**: CSV files with seizure annotations per expert.
* **working/inputs/**: preprocessed segmented windows saved as HDF5 files (one file per expert, window, and chunk).
* **outputs/**: contains experiment results per expert, per window, and chunk, including:

  * Average metrics CSV (`accuracy`, `loss`, `false positives`, `false negatives`, ...).
  * Optional epoch-level metrics CSV. 

---

## **Pipeline Files**

| File                   | Purpose                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `EEG_neonatal_FUNS.py` | Preprocessing utilities: montage generation, seizure segmentation, and sample generation.                           |
| `EEG_neonatal.py`      | Main preprocessing script. Generates HDF5 segmented windows per expert.                                             |
| `engine.py`            | Runs CNN training with k-fold cross-validation, learning rate scheduling, and evaluation.                           |
| `cnn.py`               | Defines the CNN architecture.                                                                                       |
| `evaluation.py`        | Computes metrics like accuracy, loss, false positives, false negatives per epoch.                                   |
| `main_mult.py`         | Main execution script. Calls preprocessing and CNN training for all experts, windows, and chunks.                   |
| `config.py`            | Configuration file with parameters: windows, chunks, frequencies, batch size, number of epochs, learning rate, etc. |
| `logger.py`            | Logging utility for tracking pipeline progress.                                                                     |
| `processor.py`         | Processes raw HDF5 EEG data into features and labels suitable for CNN input.                                        |
| `loaders.py`           | Utility to construct filenames and load HDF5 files.                                                                 |

---

## **Execution Order**

1. **Preprocessing** – Generate segmented EEG windows:

```bash
python EEG_neonatal.py
```

* This reads raw EDF files and annotations, creates bipolar montages, segments seizures and non-seizures into windows and chunks, and writes HDF5 files into `working/inputs/`.

2. **CNN Training & Evaluation** – Run experiments per expert:

```bash
python main_mult.py
```

* For each expert, window, and chunk combination:

  * Loads preprocessed HDF5 data from `working/inputs/`.
  * Segments into training and validation sets.
  * Trains CNN using k-fold cross-validation.
  * Logs metrics per epoch (`accuracy`, `loss`, `false positives`, `false negatives`).
  * Stores average metrics per window and chunk in `outputs/<expert>/`.

---

## **Outputs**

* **Per-expert folder** (e.g., `outputs/A/`):

  * `results_W{window}_C{chunk}.csv` → contains **average metrics across k folds** (accuracy, loss, false positives, false negatives).
  * `epoch_metrics_W{window}_C{chunk}.csv` → optional, per-epoch metrics for training and validation sets.

---

## **Configuration**

* Modify `config.py` to adjust parameters:

```python
WINDOWS = [1, 2, 5]        # Window sizes in seconds
CHUNKS = [1, 2, 5, 10, 20] # Maximum number of chunks per window
FREQS = 64                 # Sampling frequency after downsampling
EXPERTS = ["A", "B", "C"]
EPOCHS = 50
BATCH_SIZE = 32
KFOLDS = 5
LEARNING_RATE = 0.1
SEED = 42
INPUT_DIR = Path("working/inputs")
OUTPUT_DIR = Path("outputs")
```

---

## **Notes**

* The pipeline currently assumes **balanced seizure and non-seizure segments** by replicating non-seizure windows.
* Epoch-level metrics include both training and validation sets.
* HDF5 files must be present before CNN training. Missing files will cause `main_mult.py` to skip that window/chunk.
* All logs are stored using `logger.py`, printed to console for tracking.

---
