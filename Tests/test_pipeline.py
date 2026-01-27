"""
Verifies the CNN architecture, ensures the model is unpacked correctly 
from the tuple returned by build_cnn, validates that the processor segments raw data 
into windows and maintains the 4D shape required for Conv2D layers, verifies the 
calculation of the performance metrics. 

Use the following line on the PowerShell to see loggers of the test process:
python -m pytest Tests/test_preprocessing.py -s --log-cli-level=INFO"
pytest Tests/test_pipeline.py -s --log-cli-level=INFO
"""


import os
import sys
import logging
import numpy as np
import pytest
from pathlib import Path

# --- ENVIRONMENT SETUP ---
# Identify the root project directory by going up one level from the Tests folder
root = Path(__file__).resolve().parents[1]
# Define the path to the 'src' directory where the project modules are located
src_path = str(root / "src")

# Manually insert the src directory into sys.path to ensure Python can import local modules
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the functions from src directory
from cnn import build_cnn
from processor import process_and_segment
from evaluation import MetricsPerEpoch
from config import NO_OF_EEG_CHANNELS, FREQS

# Set up a logger to provide real-time feedback in the terminal during testing
logger = logging.getLogger("PIPELINE_VERIFIER")

def test_logger_shape_and_montage():
    """
    Verifies the CNN architecture and ensures the model is unpacked correctly 
    from the tuple returned by build_cnn.
    """
    # Define an input window: 640 samples (10s at 64Hz), 18 EEG channels, 1 filter depth
    input_shape = (640, 18, 1)
    
    # build_cnn returns a tuple: (tensorflow_model, string_description)
    model, description = build_cnn(input_shape)
    
    # Log the configuration and the model's expected input shape for manual verification
    logger.info(f"\n[MONTAGE] Channels configured in config.py: {NO_OF_EEG_CHANNELS}")
    logger.info(f"[SHAPE] Input detected by the CNN: {model.input_shape}")
    
    # Keras adds a 'None' dimension at index 0 to represent the Batch Size
    assert model.input_shape == (None, 640, 18, 1)
    logger.info("Architecture: Model built and unpacked correctly.")


def test_logger_segmentation_data():
    """
    Validates that the processor correctly segments raw data into windows 
    and maintains the 4D shape required for Conv2D layers.
    """
    window_size = 640
    # Simulate raw data: 2 full windows, 19 columns (18 EEG channels + 1 Label column)
    mock_data = np.random.rand(window_size * 2, 19)
    
    # The processor should separate the 18 channels from the label and normalize
    x, y = process_and_segment(mock_data, window_size)
    
    # Log the resulting matrix shapes to verify dimensions
    logger.info(f"\n[MATRIX] Shape of X (Features): {x.shape}")
    logger.info(f"[MATRIX] Shape of Y (Labels): {y.shape}")
    
    # Confirm X is 4D: (Number of windows, Time samples, Channels, Depth)
    assert x.shape == (2, 640, 18, 1)
    # Confirm Y contains one label per window
    assert len(y) == 2
    logger.info("Processor: 4D Matrix and labels verified.")

def test_logger_metrics():
    """
    Verifies that the confusion matrix and accuracy logic in evaluation.py
    calculates percentages correctly.
    """
    # Create controlled labels and predictions (75% accuracy)
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    
    # Calculate True Positives, True Negatives, False Positives, False Negatives, and Accuracy
    tp, tn, fp, fn, acc = MetricsPerEpoch.compute_confusion(y_true, y_pred)
    
    logger.info(f"\n[METRICS] Accuracy: {acc*100}%")
    
    # Assert that the math correctly results in 0.75
    assert acc == 0.75
    logger.info("Evaluation: Math logic verified.")